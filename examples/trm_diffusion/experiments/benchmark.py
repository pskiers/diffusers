"""
experiments/benchmark.py — how long a training run (and its CoReBench evals)
will take on the GPUs it is launched on, and which batch sizes to use: minutes
of the real code paths, extrapolated.

    accelerate launch --multi_gpu --num_processes 8 experiments/benchmark.py experiment=std120k_thinker_minit2i \
        train.batch_size=32 train.gradient_accumulation_steps=3 \
        '+bench.train_batch_sizes=[16,32,48,64]' '+bench.sample_batch_sizes=[8,16,32,64]'

Training (`bench.train_batch_sizes`, per-GPU micro-batches; default: the
config's): for each size, the gradient accumulation that keeps the launched
effective batch (train.batch_size x accumulation x GPUs, e.g. 768); then
`bench.warmup_steps` untimed train steps (torch.compile, allocator warm-up) and
`bench.train_steps` timed ones, as train_trm.py runs them (real data, EMA).
A size that runs out of memory is reported as such (all GPUs agree on it) and
larger ones are skipped. The fastest size is projected to train.num_steps.

Sampling (`bench.sample_batch_sizes`, images per GPU batch; default: the
CoReBench callback's): `bench.sample_steps` timed denoising steps per size
(after one warm-up step), with the training state still on the GPU as during a
training eval; per-step cost doesn't depend on the step, so seconds per image
= per-step time x the sampler's steps / batch. The fastest size is used for
the CoReBench run below and its generation projection.

CoReBench (`bench.corebench_prompts_per_dimension` prompts of each dimension;
0 skips): the config's CoReBenchEvalCallback (needs judge_python, e.g.
COREBENCH_JUDGE_PYTHON) generated on all GPUs and judged by the real judges,
projected to the full benchmark: judging from the slowest parallel judge's load
time plus its time per judgment. Few prompts leave the judges' batches part
empty, so the judging projection errs long; for tuning the judges (overrides
eval_callbacks.1.judges_per_node / judge_tensor_parallel_size /
'judge_args=[--batch-size,1024]'), use ~8 prompts per dimension and
+bench.skip_training=true.

Options (+bench.<name>=...): train_batch_sizes, warmup_steps (2), train_steps
(5), sample_batch_sizes, sample_steps (3), corebench_prompts_per_dimension (2),
skip_training (false). Writes <run.output_dir>/benchmark.json.
"""

import datetime
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
import torch.distributed as dist
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_sample import DataSample
from eval.corebench_eval import corebench_prompts
from factory import build_datasets, build_model
from models.eval_callbacks import CoReBenchEvalCallback
from models.sampling import CFGPredictor, SamplingPipeline
from models.trm.ema import EMAHelper
from train_trm import build_train_dataloader

DEFAULTS = dict(train_batch_sizes=None, warmup_steps=2, train_steps=5, sample_batch_sizes=None, sample_steps=3,
                corebench_prompts_per_dimension=2, skip_training=False)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


class _Agreement:
    """Whether every rank got through a trial without running out of memory —
    over a CPU (gloo) group, so it can't pair up with a pending NCCL op."""

    def __init__(self, world: int):
        self.group = dist.new_group(backend="gloo", timeout=datetime.timedelta(minutes=30)) if world > 1 else None

    def all_ok(self, ok: bool) -> bool:
        if self.group is None:
            return ok
        failures = torch.tensor([0 if ok else 1])
        dist.all_reduce(failures, group=self.group)
        return failures.item() == 0


def _free_after_oom(optimizers=()) -> None:
    for opt in optimizers:
        opt.zero_grad()
    torch.cuda.empty_cache()


def benchmark_training(cfg, accelerator, bench, agreement) -> tuple[dict, object]:
    world = accelerator.num_processes
    accum0 = cfg.train.get("gradient_accumulation_steps", 1)
    effective = cfg.train.batch_size * accum0 * world  # the launched effective batch, kept for every size
    sizes = sorted(bench.train_batch_sizes or [cfg.train.batch_size])

    torch.manual_seed(cfg.train.seed + accelerator.process_index)
    train_ds, _ = build_datasets(cfg)
    model = build_model(cfg, instantiate(cfg.diffusion))
    if cfg.train.compile:
        model.compile_submodules()
    optimizers = model.build_optimizers(world, cfg.train.num_steps)
    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    ema = EMAHelper(mu=cfg.ema.rate) if cfg.ema.enabled else None
    if ema is not None:
        ema.register(unwrapped)
    unwrapped.train()

    trials = []
    for size in sizes:
        if effective % (size * world):
            trials.append({"batch_size": size, "skipped": f"{effective} is not a multiple of {size} x {world} GPUs"})
            continue
        accum = effective // (size * world)
        with open_dict(cfg):
            cfg.train.batch_size = size
        loader = accelerator.prepare(build_train_dataloader(cfg, train_ds))
        batches = iter(loader)

        def next_batch():  # a new epoch when the data runs out, as in train_trm.py
            nonlocal batches
            try:
                return next(batches)
            except StopIteration:
                batches = iter(loader)
                return next(batches)

        torch.cuda.reset_peak_memory_stats()
        times, increments, global_step, ok = [], [], 0, True
        try:
            for i in range(bench.warmup_steps + bench.train_steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                micro_batches = [next_batch() for _ in range(accum)]
                _, _, new_step = unwrapped.train_step(micro_batches, accelerator, optimizers, ema, size * world, global_step)
                torch.cuda.synchronize()
                if i >= bench.warmup_steps:
                    times.append(time.perf_counter() - t0)
                    increments.append(new_step - global_step)
                global_step = new_step
        except torch.cuda.OutOfMemoryError:
            ok = False
        del batches, loader
        ok = agreement.all_ok(ok)
        if not ok:
            _free_after_oom(optimizers)
            trials.append({"batch_size": size, "accumulation": accum, "out_of_memory": True})
            if accelerator.is_main_process:
                print(f"[bench] train batch {size}: out of memory — skipping larger sizes", flush=True)
            break
        seconds = statistics.median(times)
        trial = {"batch_size": size, "accumulation": accum, "seconds_per_train_step": seconds,
                 "samples_per_second": effective / seconds, "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                 "train_steps_per_run": cfg.train.num_steps / statistics.mean(increments)}
        trials.append(trial)
        if accelerator.is_main_process:
            print(f"[bench] train batch {size} x accumulation {accum}: {seconds:.2f} s/step, "
                  f"{trial['samples_per_second']:.1f} samples/s, peak {trial['peak_memory_gb']:.1f} GB", flush=True)

    for opt in optimizers:  # the optimizer state stays allocated, as during training evals
        opt.zero_grad()
    timed = [t for t in trials if "seconds_per_train_step" in t]
    best = max(timed, key=lambda t: t["samples_per_second"]) if timed else None
    result = {"gpus": world, "effective_batch": effective, "trials": trials, "best": best}
    if best:
        result["projected_training_seconds"] = best["train_steps_per_run"] * best["seconds_per_train_step"]
    return result, unwrapped


@torch.no_grad()
def benchmark_sampling(model, accelerator, callback, bench, agreement) -> dict:
    """Seconds per image (per GPU) of the CoReBench sampler at each batch size."""
    from transformers import AutoTokenizer

    sizes = sorted(bench.sample_batch_sizes or [callback.gen_kwargs["batch_size"]])
    steps = callback.gen_kwargs["steps"]
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(callback.tokenizer_name, model_max_length=callback.prompt_length)
    texts = [p.text for p in corebench_prompts()]
    model.eval()
    trials = []
    for size in sizes:
        tokens = tokenizer(texts[:size], max_length=callback.prompt_length, padding="max_length", truncation=True,
                           return_tensors="pt")
        conditions = DataSample(token_conditions=tokens["input_ids"], token_mask=tokens["attention_mask"]).to(device)

        def sample(n_steps):
            pipeline = SamplingPipeline(num_inference_steps=n_steps, predictor=CFGPredictor(callback.gen_kwargs["cfg_scale"]))
            pipeline.sample_one_batch(model, conditions, device)
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        ok = True
        try:
            sample(1)  # warm-up (compilation, allocator)
            t0 = time.perf_counter()
            sample(bench.sample_steps)
            per_step = (time.perf_counter() - t0) / bench.sample_steps
        except torch.cuda.OutOfMemoryError:
            ok = False
        ok = agreement.all_ok(ok)
        if not ok:
            _free_after_oom()
            trials.append({"batch_size": size, "out_of_memory": True})
            if accelerator.is_main_process:
                print(f"[bench] sample batch {size}: out of memory — skipping larger sizes", flush=True)
            break
        trial = {"batch_size": size, "seconds_per_image": per_step * steps / size,
                 "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
        trials.append(trial)
        if accelerator.is_main_process:
            print(f"[bench] sample batch {size}: {trial['seconds_per_image']:.2f} s/image/GPU ({steps} steps, CFG), "
                  f"peak {trial['peak_memory_gb']:.1f} GB", flush=True)
    torch.cuda.empty_cache()
    timed = [t for t in trials if "seconds_per_image" in t]
    return {"trials": trials, "best": min(timed, key=lambda t: t["seconds_per_image"]) if timed else None}


def benchmark_corebench(cfg, accelerator, model, callback, bench, sampling) -> dict:
    if bench.corebench_prompts_per_dimension <= 0:
        return {}
    if callback.judge_python is None:
        if accelerator.is_main_process:
            print("[bench] CoReBench skipped: no judge_python (set COREBENCH_JUDGE_PYTHON)", flush=True)
        return {}
    callback.prompts_per_dimension = bench.corebench_prompts_per_dimension
    callback.out_dir = str(Path(cfg.run.output_dir) / "benchmark_corebench")
    callback.keep_images = False
    if sampling.get("best"):
        callback.gen_kwargs["batch_size"] = sampling["best"]["batch_size"]
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
    if sampling.get("best"):
        seconds_per_image = sampling["best"]["seconds_per_image"]
    else:
        slowest = max(outcome["generation"], key=lambda g: g["seconds"] / max(g["images"], 1))
        seconds_per_image = slowest["seconds"] / slowest["images"]
    generation = math.ceil(len(corebench_prompts()) / world) * spp * seconds_per_image  # prompts split round-robin

    # The judges run in parallel, each on 1/num_shards of the judgments: the
    # slowest one (load + its share at its own pace) sets the judging time.
    shards = outcome["judge_shards"]
    full_judgments = spp * sum(len(p.metadata["Checklist"]) for p in corebench_prompts())
    shard_times = []
    for sh in shards:
        init = sh["timing"].get("judge_init_seconds", 0.0)
        per_judgment = (sh["timing"]["total_seconds"] - init) / sh["judgments"]
        shard_times.append((init + full_judgments / len(shards) * per_judgment, init, per_judgment))
    judging, init, per_judgment = max(shard_times)
    return {
        "subset_prompts_per_dimension": callback.prompts_per_dimension,
        "generation_batch_size": callback.gen_kwargs["batch_size"],
        "generation_seconds_per_image_per_gpu": seconds_per_image,
        "judges": len(shards),
        "judge_tensor_parallel_size": callback.judge_tensor_parallel_size,
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
    bench = type("Bench", (), {**DEFAULTS, **dict(cfg.get("bench") or {})})
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    agreement = _Agreement(accelerator.num_processes)

    if bench.skip_training:
        train, model = {}, build_model(cfg, instantiate(cfg.diffusion)).to(accelerator.device)
    else:
        train, model = benchmark_training(cfg, accelerator, bench, agreement)
    callbacks = [cb for cb in model.eval_callbacks if isinstance(cb, CoReBenchEvalCallback)]
    callback = callbacks[0] if callbacks else None
    sampling = benchmark_sampling(model, accelerator, callback, bench, agreement) if callback else {}
    core = benchmark_corebench(cfg, accelerator, model, callback, bench, sampling) if callback else {}
    if not accelerator.is_main_process:
        return

    lines, total = [], 0.0
    if train:
        lines.append(f"GPUs: {train['gpus']}   effective batch: {train['effective_batch']}")
        for t in train["trials"]:
            if "skipped" in t:
                lines.append(f"  train batch {t['batch_size']}: skipped ({t['skipped']})")
            elif t.get("out_of_memory"):
                lines.append(f"  train batch {t['batch_size']} x accumulation {t['accumulation']}: out of memory")
            else:
                lines.append(f"  train batch {t['batch_size']} x accumulation {t['accumulation']}: "
                             f"{t['seconds_per_train_step']:.2f} s/step, {t['samples_per_second']:.1f} samples/s, "
                             f"peak {t['peak_memory_gb']:.1f} GB")
        if train["best"]:
            b, total = train["best"], train["projected_training_seconds"]
            lines.append(f"training: best batch {b['batch_size']} -> {b['train_steps_per_run']:.0f} train steps for "
                         f"train.num_steps={cfg.train.num_steps}: {_hours(total)}")
    for t in sampling.get("trials", []):
        lines.append(f"  sample batch {t['batch_size']}: " + ("out of memory" if t.get("out_of_memory") else
                     f"{t['seconds_per_image']:.2f} s/image/GPU, peak {t['peak_memory_gb']:.1f} GB"))
    if sampling.get("best"):
        lines.append(f"sampling: best batch {sampling['best']['batch_size']} (eval_callbacks.1.batch_size)")
    if core:
        total += core["evals_per_run"] * core["projected_corebench_seconds"]
        lines += [
            f"CoReBench (projected from {core['subset_prompts_per_dimension']} prompts/dimension): "
            f"generation {_hours(core['projected_generation_seconds'])} "
            f"({core['generation_seconds_per_image_per_gpu']:.2f} s/image/GPU at batch {core['generation_batch_size']}), "
            f"judging {_hours(core['projected_judging_seconds'])} "
            f"({core['judges']} parallel judges x {core['judge_tensor_parallel_size']} GPUs; slowest: load "
            f"{core['judge_init_seconds']:.0f} s + {core['judge_seconds_per_judgment']:.3f} s/judgment) "
            f"= {_hours(core['projected_corebench_seconds'])} per eval",
            f"  x {core['evals_per_run']} evals (every {core['every_n_steps']} steps)",
        ]
    if train:
        lines.append(f"total (training + CoReBench evals; other evals not included): {_hours(total)}")
    print("\n[bench] " + "\n[bench] ".join(lines), flush=True)
    out = Path(cfg.run.output_dir) / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"training": train, "sampling": sampling, "corebench": core,
                               "projected_total_seconds": total if train else None}, indent=2))
    print(f"[bench] -> {out}", flush=True)


if __name__ == "__main__":
    main()
