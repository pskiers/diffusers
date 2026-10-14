"""experiments/sweep_dm_flops_vs_acc.py — puzzle accuracy vs. real FLOPs for
ONE plain-DM (painter-only) checkpoint, sweeping DDIM inference-step count.

One of the two data sources for the paper's 6-curve compute/accuracy plot
(paper_plots/plot_flops_vs_puzzle_acc.py): run this once per DM checkpoint
(small 2M / medium 20M / big 80M / large 200M), each producing a JSON file
with (flops, puzzle_acc) points; the other 2 curves (TRM flat/wise) come
from experiments/sweep_trm_flops_vs_acc.py.

Reuses ablate_trm_loop_budget.py's `_build_cached_batches`/`_load_checkpoint`
and denoising_budget_sweep.py's `_run_sampling`/`evaluate_grids` pattern
verbatim (see their docstrings) rather than reimplementing the sampling
loop — this script only adds the FLOPs axis (flops_utils.measure_dm_flops,
profiled once, then flops = num_steps * (2 if cfg else 1) * F_unet for
every sweep point) on top of that machinery.

IMPORTANT — two overrides the 20M/80M/200M checkpoints' real training
commands used that must be replicated exactly at eval time, or accuracy
silently collapses:
  - diffusion.num_train_timesteps=4800 — hand-edited into the config on the
    cluster (NOT what's checked into this repo's yaml files, which still
    read 100 for 20M/200M and 1000 for 80M). Omitting it samples with the
    wrong noise schedule.
  - data=mnist_sudoku — the 20M/80M/200M training commands explicitly
    overrode data to the plain mnist_sudoku dataset, even though those
    experiments' own defaults point at data: mnist_sudoku_srm (a different
    dataset). Confirmed empirically: evaluating the 80M checkpoint under
    the experiment's default (mnist_sudoku_srm) gave puzzle_acc=0.0/384 at
    20 steps (statistically ~impossible if the true rate were the expected
    ~3% — P(0 successes | p=0.03, n=384) ~ 1e-5); adding data=mnist_sudoku
    fixed it to 1.8%, matching the sanity check. The small (2M) painter and
    the TRM checkpoint don't need this override — their experiments already
    default to data: mnist_sudoku.

Usage (one call per checkpoint — the small 2M painter is deliberately NOT
included here; see paper_plots/plot_flops_vs_puzzle_acc.py's docstring for
why):
    python experiments/sweep_dm_flops_vs_acc.py \\
      experiment=mnist_unet_concat_painter_srm_20m +checkpoint=runs/sudoku_extr_20M.pt \\
      data=mnist_sudoku diffusion.num_train_timesteps=4800 \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +sweep.num_samples=256 +sweep.out=runs/flops_sweep_dm_20m.json

    python experiments/sweep_dm_flops_vs_acc.py \\
      experiment=mnist_unet_concat_painter_srm_1000t +checkpoint=runs/unet_etreme.pt \\
      data=mnist_sudoku diffusion.num_train_timesteps=4800 \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +sweep.num_samples=256 +sweep.out=runs/flops_sweep_dm_80m.json

    python experiments/sweep_dm_flops_vs_acc.py \\
      experiment=mnist_unet_concat_painter_srm_200m +checkpoint=runs/sudoku_extr_200M.pt \\
      data=mnist_sudoku diffusion.num_train_timesteps=4800 \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +sweep.num_samples=256 +sweep.out=runs/flops_sweep_dm_200m.json

Options (all under +sweep.*):
    num_samples — puzzles evaluated per grid point (default 256)
    steps       — list of num_inference_steps to sweep (default a log-ish
                  grid up to 4800; pass your own to go higher, e.g. for the
                  small model: +sweep.steps=[1,2,5,10,20,50,100,250,500,
                  1000,2000,4800,6000,8000] — note anything above
                  diffusion.num_train_timesteps repeats/degenerates DDIM
                  timestep indices, so treat those points as explicit
                  extrapolation, not a "more real steps" data point)
    seed        — default 0
    out         — output JSON path
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import dataclasses

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from eval.mnist_eval import evaluate_grids
from export_sampling_animation import _slice_sample
from factory import build_datasets, build_model
from flops_utils import measure_dm_flops

logger = get_logger(__name__, log_level="INFO")

DEFAULT_STEPS = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 4800]


@torch.no_grad()
def _run_point(model, classifier, cell_size: int, cached_batches: list, num_steps: int, cfg_scale: float) -> dict:
    """Mirrors denoising_budget_sweep.py's _run_sampling, but sets up the
    scheduler via set_timesteps(num_inference_steps=...) only (not an
    explicit `timesteps=` list) — DDIMScheduler's set_timesteps in the
    installed diffusers version only accepts num_inference_steps, unlike
    DDPMScheduler, so this script stays scheduler-agnostic (uniform step
    counts are all we ever need here, unlike denoising_budget_sweep.py's
    own non-uniform "bump" step-placement axis, which does need that kwarg)."""
    all_puzzle = []
    t0 = time.time()
    for cb in cached_batches:
        device = cb["x_init"].device
        model.scheduler.set_timesteps(num_inference_steps=num_steps, device=device)
        x = cb["x_init"].clone()
        for t in model.scheduler.timesteps:
            t_batch = t.expand(x.shape[0]).to(device)
            step_sample = dataclasses.replace(cb["conditions"], x_noisy=x, timesteps=t_batch)
            noise_pred = model(step_sample).pred
            if cfg_scale != 1.0:
                null_sample = model.null_condition_sample(step_sample)
                pred_u = model(null_sample).pred
                noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
            x = model.scheduler.step(noise_pred, t, x).prev_sample
        generated = model.decode_for_eval(x)
        acc = evaluate_grids(generated, cb["solutions"], classifier, cell_size, given_masks=cb["given_masks"])
        all_puzzle.append(acc["puzzle_acc"])
    return {"puzzle_acc": float(np.mean(all_puzzle)), "wall_time_sec": time.time() - t0}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/sweep_dm_flops_vs_acc.py experiment=<name> "
            "+checkpoint=<path/to/checkpoint.pt> diffusion.num_train_timesteps=4800 [+sweep.xxx=...]"
        )

    sw = cfg.get("sweep", {})
    num_samples: int = sw.get("num_samples", 256)
    seed: int = sw.get("seed", 0)
    steps: list[int] = list(sw.get("steps", DEFAULT_STEPS))
    out_path: str = sw.get("out", str(Path(checkpoint).parent / "flops_sweep_dm.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds, batch_size=cfg.eval.get("batch_size", cfg.train.batch_size), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    num_train_timesteps = model.scheduler.config.num_train_timesteps
    pipeline = model.sampling_pipeline
    cfg_scale: float = sw.get("cfg_scale", pipeline.cfg_scale)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(f"num_train_timesteps={num_train_timesteps}  cfg_scale={cfg_scale}  steps={steps}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    flops_sample = _slice_sample(cached_batches[0]["conditions"], 0)
    flops_sample = dataclasses.replace(
        flops_sample, x_noisy=cached_batches[0]["x_init"][:1], timesteps=torch.zeros(1, device=device),
    )
    f_unet = measure_dm_flops(model, flops_sample)
    logger.info(f"F_unet (one denoiser forward) = {f_unet:,} FLOPs")

    points = []
    for n in steps:
        logger.info(f"Running num_inference_steps={n} ...")
        r = _run_point(model, classifier, cell_size, cached_batches, n, cfg_scale)
        flops = n * (2 if cfg_scale != 1.0 else 1) * f_unet
        point = {"num_steps": n, "flops": flops, **r}
        points.append(point)
        logger.info(f"  → {point}")

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print(f"{'num_steps':>10}{'flops':>18}{'puzzle_acc':>12}")
        print("=" * 70)
        for p in points:
            print(f"{p['num_steps']:>10}{p['flops']:>18,}{p['puzzle_acc']:>12.4f}")
        print("=" * 70)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "checkpoint": str(checkpoint), "num_samples": num_samples, "cfg_scale": cfg_scale,
                "num_train_timesteps": num_train_timesteps, "F_unet": f_unet, "points": points,
            }, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return points


if __name__ == "__main__":
    main()
