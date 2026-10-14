"""
experiments/benchmark.py — how long a training run (and its CoReBench evals) will take on
the GPUs it is launched on: a few minutes of the real code paths, extrapolated.

    accelerate launch --multi_gpu --num_processes 8 experiments/benchmark.py experiment=std120k_thinker_minit2i \
        train.batch_size=32 train.gradient_accumulation_steps=3

Training: `bench.warmup_steps` untimed train steps (torch.compile, allocator
warm-up), then `bench.train_steps` timed ones, each one effective batch of the
config's real data and model on all launched GPUs (as train_trm.py runs them,
EMA included); projected to train.num_steps.

CoReBench: the config's CoReBenchEvalCallback (needs its judge_python, e.g.
COREBENCH_JUDGE_PYTHON) on the first `bench.corebench_prompts_per_dimension`
prompts of each dimension, generated on all GPUs and judged by the real judge;
projected to the full 1,080 prompts — generation from the slowest GPU's time
per image, judging from the judge's load time plus its time per judgment. The
per-judgment time of a small subset includes more batch tail, so the judging
projection errs long.

Options (Hydra +overrides): +bench.warmup_steps=2 +bench.train_steps=5
+bench.corebench_prompts_per_dimension=2 (0 skips CoReBench).
Writes <run.output_dir>/benchmark.json.
"""

import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.corebench_eval import corebench_prompts
from factory import build_datasets, build_model
from models.eval_callbacks import CoReBenchEvalCallback
from models.trm.ema import EMAHelper
from train_trm import build_train_dataloader


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def benchmark_training(cfg, accelerator, bench) -> tuple[dict, object]:
    torch.manual_seed(cfg.train.seed + accelerator.process_index)
    train_ds, _ = build_datasets(cfg)
    train_dl = build_train_dataloader(cfg, train_ds)
    model = build_model(cfg, instantiate(cfg.diffusion))
    if cfg.train.compile:
        model.compile_submodules()
    world = accelerator.num_processes
    optimizers = model.build_optimizers(world, cfg.train.num_steps)
    model, train_dl = accelerator.prepare(model, train_dl)
    unwrapped = accelerator.unwrap_model(model)
    ema = EMAHelper(mu=cfg.ema.rate) if cfg.ema.enabled else None
    if ema is not None:
        ema.register(unwrapped)

    accum = cfg.train.get("gradient_accumulation_steps", 1)
    global_batch_size = cfg.train.batch_size * world
    batches = iter(train_dl)
    global_step, times, increments = 0, [], []
    unwrapped.train()
    for i in range(bench.warmup_steps + bench.train_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        micro_batches = [next(batches) for _ in range(accum)]
        _, _, new_step = unwrapped.train_step(micro_batches, accelerator, optimizers, ema, global_batch_size, global_step)
        torch.cuda.synchronize()
        if i >= bench.warmup_steps:
            times.append(time.perf_counter() - t0)
            increments.append(new_step - global_step)
        global_step = new_step
        if accelerator.is_main_process:
            print(f"[bench] train step {i + 1}: {time.perf_counter() - t0:.2f}s"
                  + (" (warm-up)" if i < bench.warmup_steps else ""), flush=True)

    seconds = statistics.median(times)
    samples = global_batch_size * accum
    calls = cfg.train.num_steps / statistics.mean(increments)  # train_step calls in the whole run
    result = {
        "gpus": world,
        "effective_batch": samples,
        "seconds_per_train_step": seconds,
        "samples_per_second": samples / seconds,
        "train_steps_per_run": calls,
        "projected_training_seconds": calls * seconds,
        "peak_memory_gb_rank0": torch.cuda.max_memory_allocated() / 1e9,
    }
    for opt in optimizers:  # the optimizer state stays allocated, as during training evals
        opt.zero_grad()
    return result, unwrapped


def benchmark_corebench(cfg, accelerator, model, bench) -> dict:
    callbacks = [cb for cb in model.eval_callbacks if isinstance(cb, CoReBenchEvalCallback)]
    if bench.corebench_prompts_per_dimension <= 0 or not callbacks:
        return {}
    callback = callbacks[0]
    if callback.judge_python is None:
        if accelerator.is_main_process:
            print("[bench] CoReBench skipped: no judge_python (set COREBENCH_JUDGE_PYTHON)", flush=True)
        return {}
    callback.prompts_per_dimension = bench.corebench_prompts_per_dimension
    callback.out_dir = str(Path(cfg.run.output_dir) / "benchmark_corebench")
    callback.keep_images = False
    if accelerator.is_main_process:
        shutil.rmtree(callback.out_dir, ignore_errors=True)  # no stale, cached results
    accelerator.wait_for_everyone()

    model.eval()
    with torch.no_grad():
        callback.run(model, accelerator, step=0)
    if not accelerator.is_main_process:
        return {}

    outcome = json.loads((Path(callback.out_dir) / "step00000000" / "judged.json").read_text())
    if "error" in outcome:
        raise RuntimeError(f"CoReBench benchmark failed: {outcome['error']}")
    spp = callback.gen_kwargs["samples_per_prompt"]
    world = accelerator.num_processes
    slowest = max(outcome["generation"], key=lambda g: g["seconds"] / max(g["images"], 1))
    seconds_per_image = slowest["seconds"] / slowest["images"]
    full_images_per_gpu = math.ceil(len(corebench_prompts()) / world) * spp  # prompts are split round-robin
    generation = full_images_per_gpu * seconds_per_image

    timing = outcome["scores"]["timing"]
    init = timing.get("judge_init_seconds", 0.0)
    subset_judgments = spp * sum(len(p.metadata["Checklist"]) for p in corebench_prompts(callback.prompts_per_dimension))
    full_judgments = spp * sum(len(p.metadata["Checklist"]) for p in corebench_prompts())
    per_judgment = (timing["total_seconds"] - init) / subset_judgments
    judging = init + full_judgments * per_judgment
    return {
        "subset_prompts_per_dimension": callback.prompts_per_dimension,
        "generation_seconds_per_image_per_gpu": seconds_per_image,
        "judge_init_seconds": init,
        "judge_seconds_per_judgment": per_judgment,
        "projected_generation_seconds": generation,
        "projected_judging_seconds": judging,
        "projected_corebench_seconds": generation + judging,
        "every_n_steps": callback.every_n_steps,
        "evals_per_run": cfg.train.num_steps // callback.every_n_steps,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    bench = type("Bench", (), {"warmup_steps": 2, "train_steps": 5, "corebench_prompts_per_dimension": 2,
                               **dict(cfg.get("bench") or {})})
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)

    train, model = benchmark_training(cfg, accelerator, bench)
    core = benchmark_corebench(cfg, accelerator, model, bench)
    if not accelerator.is_main_process:
        return

    total = train["projected_training_seconds"]
    lines = [
        f"GPUs: {train['gpus']}   effective batch: {train['effective_batch']}   "
        f"peak memory (rank 0): {train['peak_memory_gb_rank0']:.1f} GB",
        f"training: {train['seconds_per_train_step']:.2f} s per train step "
        f"({train['samples_per_second']:.1f} samples/s) -> {train['train_steps_per_run']:.0f} train steps "
        f"for train.num_steps={cfg.train.num_steps}: {_hours(total)}",
    ]
    if core:
        total += core["evals_per_run"] * core["projected_corebench_seconds"]
        lines += [
            f"CoReBench (projected from {core['subset_prompts_per_dimension']} prompts/dimension): "
            f"generation {_hours(core['projected_generation_seconds'])} "
            f"({core['generation_seconds_per_image_per_gpu']:.2f} s/image/GPU), "
            f"judging {_hours(core['projected_judging_seconds'])} "
            f"(load {core['judge_init_seconds']:.0f} s + {core['judge_seconds_per_judgment']:.3f} s/judgment) "
            f"= {_hours(core['projected_corebench_seconds'])} per eval",
            f"  x {core['evals_per_run']} evals (every {core['every_n_steps']} steps)",
        ]
    lines.append(f"total (training + CoReBench evals; other evals not included): {_hours(total)}")
    print("\n[bench] " + "\n[bench] ".join(lines), flush=True)
    out = Path(cfg.run.output_dir) / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"training": train, "corebench": core, "projected_total_seconds": total}, indent=2))
    print(f"[bench] -> {out}", flush=True)


if __name__ == "__main__":
    main()
