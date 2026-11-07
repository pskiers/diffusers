"""
ABOUTME: Rebuttal exp 06: parameters, inference FLOPs and latency per sample of PaTh, the DM baselines and the TRM
ABOUTME: denoiser on every task, each sampled exactly as its eval does (its own config's sampling pipeline).

Per entry (task/model): the model is built from the run's Hydra overrides (frozen painters load their checkpoint; the
compute does not depend on the trained weights, so thinker/DM checkpoints are not loaded: no data-dependent control
flow is enabled, e.g. the EXTREME thinker's halt head is off, as in its eval). Conditions are the first B items of the
val set through `model._batch_to_sample`, as the eval callbacks do.
- params: per top-level module (painter / thinker / encoder / translator ...), trainable and frozen.
- FLOPs/sample: torch's FlopCounterMode around one real `sampling_pipeline.sample_one_batch` call (B=1), so CFG
  doubling, CLEVR's extra x0-hint painter forward and every recursion are counted as run; plus decode_for_eval (VAE
  decode on CLEVR, a no-op elsewhere) in its own column.
- latency: CUDA-synchronized wall clock of sample_one_batch + decode_for_eval, ms per sample at B=1 and at B=64
  (one warm-up call, then the median of --reps calls), fp32 + TF32 matmul as eval.py; a model's own painter_dtype
  (bf16 in the Sudoku DMs) applies as in its eval; compiled (compile_submodules) as the run's in-training eval was.
Accuracy: the paper-table number of the run (not re-measured).

Usage (from examples/trm_diffusion, on the H100 for the reported latencies; a GPU is needed either way):
    python experiments/compute_table.py                               # every entry
    python experiments/compute_table.py --only extreme/path,extreme/dm --batch-sizes 1 --reps 2   # smoke
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

PAINTER = "runs/mnist_unet_painter/checkpoint_final.pt"
SUDOKU_DM = ["experiment=mnist_unet_concat_painter_srm_1000t", "diffusion.num_train_timesteps=4800",
             "eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt"]
CLEVR = ["data.include_centroid_mask=true", "data.centroid_mask_presence_only=true"]
# task/model -> run (name (id)), paper-table accuracy, Hydra overrides (docs/PROJECT.md reproduce commands)
ENTRIES: dict[str, dict] = {
    "extreme/path": dict(
        run="mnist_painter_halt_pairhead_singlestage_nsup16 (y86cr9av)", acc="71.2 (puzzle acc)",
        overrides=["experiment=mnist_thinker_v1_controlnet", f"painter.checkpoint={PAINTER}",
                   "condition_encoder=x0_hint_v1", "condition_encoder.threshold=80", "+condition_encoder.enabled=false",
                   "condition_encoder.inner.with_timestep_emb=false", "mode=thinker_act",
                   "thinker.with_halt_head=true", "thinker.n_sup=16"]),
    "extreme/dm": dict(run="mnist_unet_concat_painter_extr_4800t (o1zzl5bf)", acc="2.9 (puzzle acc)",
                       overrides=[*SUDOKU_DM, "data=mnist_sudoku"]),
    "extreme/trm_denoiser": dict(run="mnist_trm_diffusion_backbone (vu78kfn4)", acc="8.1 (puzzle acc, best eval)",
                                 overrides=["experiment=mnist_trm_diffusion_backbone"]),
    "hard/path": dict(
        run="sudoku_hard_trm_0_26_x0_hint_repro (pvpx8kbs)", acc="92.5 (given-consistent)",
        overrides=["experiment=mnist_thinker_controlnet", f"painter.checkpoint={PAINTER}",
                   "translator.bridge_mode=normalized", "condition_encoder=x0_hint_v1",
                   "+condition_encoder.enabled=false", "condition_encoder.inner.with_timestep_emb=false",
                   "data=mnist_sudoku_srm", "data.train_dataset.given_cells_range=[0,26]"]),
    "hard/dm": dict(run="mnist_unet_concat_painter_srm_4800t (vl9lpx06)", acc="8.3 (given-consistent)",
                    overrides=SUDOKU_DM),
    "hard/trm_denoiser": dict(run="mnist_trm_diffusion_backbone_hard (exp 03)", acc="pending",
                              overrides=["experiment=mnist_trm_diffusion_backbone", "data=mnist_sudoku_srm"]),
    "maze/path": dict(
        run="pt_maze_final_thinker (m10k4xhx)", acc="75.1 (Exact@1, faint scorer)",
        overrides=["experiment=amaze_thinker_v2_controlnet", "data.amaze_root=data/amaze/maze", "data.cell_size=12",
                   "thinker.seq_len=144", "translator.grid=12",
                   "painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt"]),
    "maze/dm": dict(run="dit_maze_final (cb7sx7no)", acc="19.0 (Exact@1)",
                    overrides=["experiment=amaze_dit_maze", "data.amaze_root=data/amaze/maze"]),
    "maze/trm_denoiser": dict(run="amaze_trm_diffusion_backbone_maze (exp 03)", acc="pending",
                              overrides=["experiment=amaze_trm_diffusion_backbone", "data.amaze_root=data/amaze/maze"]),
    "queens/path": dict(
        run="pt_queens_final_thinker (jvvoqptb)", acc="100.0 (Exact@1)",
        overrides=["experiment=amaze_thinker_v2_controlnet", "data.amaze_root=data/amaze/queens", "data.cell_size=12",
                   "thinker.seq_len=144", "translator.grid=12",
                   "painter.checkpoint=runs/queens_painter_v2/checkpoint_final.pt"]),
    "queens/dm": dict(run="queens_dit_baseline (lhj4v8rg)", acc="90.9 (Exact@1)",
                      overrides=["experiment=amaze_dit_queens", "data.amaze_root=data/amaze/queens"]),
    "queens/trm_denoiser": dict(run="amaze_trm_diffusion_backbone_queens (exp 03)", acc="pending",
                                overrides=["experiment=amaze_trm_diffusion_backbone",
                                           "data.amaze_root=data/amaze/queens"]),
    "clevr/path": dict(
        run="clevr_thinker_v1_centroidmaskfused_x0hint_t80_sliced_biggrid_forceuncond (0il6x2cq)",
        acc="89.4 (spatial)",
        overrides=["experiment=clevr_thinker_v1_controlnet", "painter.checkpoint=runs/clevr_unet_big/checkpoint_step-80000.pt",
                   "backbone.block_out_channels=[128,256,512]",
                   "condition_encoder=object_feature_v1_centroidmask_presence_fused_x0hint",
                   "condition_encoder.threshold=80", *CLEVR, "translator=controlnet_clevr_v1_sliced",
                   "translator.painter_channels=[128,256,512]", "thinker.seq_len=266",
                   "+train.force_unconditional_painter=true"]),
    "clevr/dm": dict(run="clevr_unet_concat_painter_very_big (4ia9nlq6)", acc="68.8 (spatial)",
                     overrides=["experiment=clevr_unet_concat_painter", "backbone.block_out_channels=[256,512,512]",
                                "backbone.layers_per_block=3", "backbone.cross_attention_dim=1024"]),
    "clevr/trm_denoiser": dict(run="clevr_trm_diffusion_backbone (exp 03)", acc="pending",
                               overrides=["experiment=clevr_trm_diffusion_backbone"]),
}


def build(entry: dict, extra: list[str], device: torch.device):
    from factory import build_datasets, build_model

    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[*entry["overrides"], *extra])
    _, eval_ds = build_datasets(cfg)
    model = build_model(cfg, instantiate(cfg.diffusion)).to(device).eval()
    return cfg, model, eval_ds


def conditions(model, eval_ds, n: int, device: torch.device):
    collate = getattr(type(eval_ds), "collate_fn", None)
    if collate is None:
        from datasets.data_sample import collate_data_samples as collate
    batch = collate([eval_ds[i] for i in range(n)])
    return model._batch_to_sample(batch, device)


def param_groups(model) -> dict:
    out = {}
    for name, child in model.named_children():
        params = list(child.parameters())
        if params:
            out[name] = {"total_M": sum(p.numel() for p in params) / 1e6,
                         "trainable_M": sum(p.numel() for p in params if p.requires_grad) / 1e6}
    return out


@torch.no_grad()
def measure(model, eval_ds, device, batch_sizes: list[int], reps: int, compile_model: bool) -> dict:
    pipe = model.sampling_pipeline
    cond1 = conditions(model, eval_ds, 1, device)
    with FlopCounterMode(display=False) as fc:
        x = pipe.sample_one_batch(model, cond1, device)
    sampler_flops = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        img = model.decode_for_eval(x)
    decode_flops = fc.get_total_flops()
    res = {"sampler_gflops": sampler_flops / 1e9, "decode_gflops": decode_flops / 1e9,
           "steps": pipe.num_inference_steps, "cfg_scale": pipe.cfg_scale,
           "output_shape": list(img.shape[1:]), "compiled": compile_model, "latency": {}}
    if compile_model:  # after the FLOP count (FlopCounterMode does not see inside compiled graphs)
        model.compile_submodules()
    for B in batch_sizes:
        cond = conditions(model, eval_ds, B, device)
        times = []
        for i in range(reps + 1):  # dynamic=False compile: the warm-up call also compiles for this B
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.decode_for_eval(pipe.sample_one_batch(model, cond, device))
            torch.cuda.synchronize()
            if i > 0:  # first call is warm-up
                times.append(time.perf_counter() - t0)
        res["latency"][B] = {"ms_per_sample": 1e3 * statistics.median(times) / B, "times_s": times}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", default=None, help="comma-separated task/model keys (default: all)")
    p.add_argument("--batch-sizes", default="1,64")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--extra", nargs="*", default=[], help="Hydra overrides added to every entry (e.g. data roots)")
    p.add_argument("--no-compile", action="store_true", help="time eager (default: as run, compile_submodules() "
                   "when the run's train.compile is true; every table run trained, and so evaluated, compiled)")
    args = p.parse_args()
    keys = args.only.split(",") if args.only else list(ENTRIES)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    out_dir = Path(args.out_dir or HERE / "results" / "compute_table" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")  # as eval.py
    device = torch.device("cuda")

    results = {}
    for key in keys:
        entry = ENTRIES[key]
        t0 = time.time()
        cfg, model, eval_ds = build(entry, args.extra, device)
        r = {"run": entry["run"], "acc": entry["acc"], "params": param_groups(model),
             "painter_dtype": str(cfg.painter.get("painter_dtype", None)),
             **measure(model, eval_ds, device, batch_sizes, args.reps,
                       compile_model=bool(cfg.train.get("compile", False)) and not args.no_compile)}
        r["params_total_M"] = sum(g["total_M"] for g in r["params"].values())
        results[key] = r
        mods = ", ".join(f"{k} {v['total_M']:.1f}M" for k, v in r["params"].items())
        print(f"[{key}] {r['run']}: params {r['params_total_M']:.1f}M ({mods}), "
              f"{r['sampler_gflops']:.0f} GFLOPs/sample (+{r['decode_gflops']:.0f} decode), "
              + ", ".join(f"B={b}: {v['ms_per_sample']:.1f} ms/sample" for b, v in r["latency"].items())
              + f" ({time.time() - t0:.0f} s)", flush=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        del model
        torch.cuda.empty_cache()

    lines = ["# Exp 06: compute per task", "",
             f"GPU {torch.cuda.get_device_name(device)}; FLOPs counted on one real sampler call (B=1); latency = "
             "sample + decode, median, ms per sample. Accuracy: paper tables.", "",
             "| task | model | run | acc | params (M) | modules | steps | CFG | sampler GFLOPs | decode GFLOPs | "
             "compiled | " + " | ".join(f"ms/sample B={b}" for b in batch_sizes) + " |",
             "|---" * (11 + len(batch_sizes)) + "|"]
    for key, r in results.items():
        task, model_name = key.split("/")
        mods = ", ".join(f"{k} {v['total_M']:.1f}" + ("" if v["trainable_M"] else " (frozen)")
                         for k, v in r["params"].items())
        lines.append(f"| {task} | {model_name} | {r['run']} | {r['acc']} | {r['params_total_M']:.1f} | {mods} | "
                     f"{r['steps']} | {r['cfg_scale']:g} | {r['sampler_gflops']:.0f} | {r['decode_gflops']:.0f} | "
                     f"{r['compiled']} | "
                     + " | ".join(f"{r['latency'][b]['ms_per_sample']:.1f}" for b in batch_sizes) + " |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "git": git, "dirty": dirty, "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device), "time": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print("\n".join(lines))
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
