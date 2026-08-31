"""
experiments/denoising_budget_sweep.py — Ablation A: denoising budget sweep
for the plain (non-recursive) DM baseline.

Sweeps the plain diffusion baseline's OWN compute budget — number of
denoising steps, non-uniform step placement, and CFG scale — to make sure
a comparison against TRM accuracy isn't a strawman: if some setting of
this baseline's own knobs reaches TRM-level accuracy, the recursion claim
is dead and step-scheduling/compute becomes the finding instead. Works on
ANY PainterBase model (plain UNetPainter, DepthMatchedFeedforwardBackbone,
or even a TRM-based model for a same-axis comparison) since it only uses
model(sample) / model.null_condition_sample / model.scheduler /
model.decode_for_eval — no forward_with_carry needed.

Two axes:
  uniform — standard evenly-spaced DDIM/DDPM steps at probe.uniform_steps
            (default [100, 250, 1000]).
  bump    — non-uniform step PLACEMENT (not a reasoning budget — the
            underlying denoiser call is the same either way) concentrating
            steps at HIGH NOISE, mirroring the Gaussian-bump budget
            allocation idea from experiments/ablate_trm_loop_budget.py's
            "schedule" axis, applied here to WHICH timesteps get sampled
            rather than how much reasoning each step gets. Built via
            inverse-CDF sampling from a uniform-baseline + Gaussian-bump
            density over the full [0, num_train_timesteps) range (see
            _bump_density_timesteps) — a real non-uniform timestep
            schedule handed to scheduler.set_timesteps(timesteps=...), not
            a uniform grid with steps skipped. probe.bump_total_steps sets
            the target step count (default: the middle uniform_steps
            value); probe.bump_peaks (default [0.05, 0.1]) and
            probe.bump_width set where/how sharply steps concentrate.

Every axis, every step-count/schedule config is run across
probe.cfg_scales (default [1.0, 1.5, 2.0, 3.0, 4.0]) — CFG-swept so the
baseline is tuned, not strawmanned — and reports both the full grid and,
per config, the BEST (highest puzzle_acc) CFG scale found.

Logs accuracy vs. total_denoiser_calls (= step count x2 under CFG, a
direct FLOPs proxy for a fixed-size single-shot denoiser).

Usage:
    python experiments/denoising_budget_sweep.py \\
      experiment=mnist_unet_painter checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      +probe.num_samples=256

    python experiments/denoising_budget_sweep.py \\
      experiment=mnist_depth_matched_ff checkpoint=runs/mnist_depth_matched_ff/checkpoint_final.pt \\
      +probe.num_samples=256 +probe.uniform_steps=[50,100,250,1000]
"""

from __future__ import annotations

import dataclasses
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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from eval.mnist_eval import evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


def _bump_density_timesteps(
    num_inference_steps: int, num_train_timesteps: int, peak: float, width: float, concentration: float = 4.0
) -> list[int]:
    """Non-uniform, descending timestep schedule of (up to)
    num_inference_steps entries, concentrating denoising steps near
    trajectory position `peak` (0 = highest noise / first step, 1 =
    cleanest / last step) — mirrors _matched_budget_schedule's Gaussian-
    bump allocation (experiments/ablate_trm_loop_budget.py), applied to
    WHICH timesteps get sampled instead of a reasoning budget.

    Built by inverse-CDF sampling: density = uniform baseline +
    concentration * Gaussian(peak, width) over a fine grid spanning the
    full [0, num_train_timesteps) range, normalized to a CDF, then
    num_inference_steps evenly-spaced quantiles are mapped back through
    the inverse CDF to actual timesteps. `concentration` scales how much
    extra density piles up at the peak relative to uniform; `width` is
    the bump's std as a fraction of the trajectory.

    Rounding to integer timesteps can collide near a sharp peak, so the
    returned list may be shorter than num_inference_steps after
    deduplication — callers should use the ACTUAL returned length as the
    real step count for compute-proxy reporting, not the nominal target.
    """
    M = 4000
    x = np.linspace(0.0, 1.0, M)
    sigma = max(width, 1e-6)
    bump = np.exp(-0.5 * ((x - peak) / sigma) ** 2)
    density = 1.0 + concentration * bump
    density = density / density.sum()
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    quantiles = (np.arange(num_inference_steps) + 0.5) / num_inference_steps
    idx = np.clip(np.searchsorted(cdf, quantiles), 0, M - 1)
    positions = x[idx]  # ascending in [0,1]

    timesteps = np.round((1.0 - positions) * (num_train_timesteps - 1)).astype(int)
    timesteps = np.clip(timesteps, 0, num_train_timesteps - 1)
    timesteps = np.unique(timesteps)[::-1]  # descending, de-duplicated
    return timesteps.tolist()


@torch.no_grad()
def _run_sampling(model, conditions, x_init: torch.Tensor, timesteps: list, cfg_scale: float) -> torch.Tensor:
    device = x_init.device
    model.scheduler.set_timesteps(timesteps=list(timesteps), device=device)
    x = x_init.clone()

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u = model(null_sample).pred
            noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return x


def _run_config(model, classifier, cell_size: int, cached_batches: list, timesteps: list, cfg_scale: float) -> dict:
    all_cell, all_puzzle, all_constraint, all_given_consistent = [], [], [], []
    t0 = time.time()

    for cb in cached_batches:
        x = _run_sampling(model, cb["conditions"], cb["x_init"], timesteps, cfg_scale)
        generated = model.decode_for_eval(x)
        acc = evaluate_grids(generated, cb["solutions"], classifier, cell_size, given_masks=cb["given_masks"])
        all_cell.append(acc["cell_acc"])
        all_puzzle.append(acc["puzzle_acc"])
        all_constraint.append(acc.get("constraint_puzzle_acc", 0.0))
        if acc.get("given_consistent_puzzle_acc") is not None:
            all_given_consistent.append(acc["given_consistent_puzzle_acc"])

    result = {
        "cfg_scale": cfg_scale,
        "num_steps": len(timesteps),
        "total_denoiser_calls": len(timesteps) * (2 if cfg_scale != 1.0 else 1),
        "cell_acc": float(np.mean(all_cell)),
        "puzzle_acc": float(np.mean(all_puzzle)),
        "constraint_puzzle_acc": float(np.mean(all_constraint)),
        "wall_time_sec": time.time() - t0,
    }
    if all_given_consistent:
        result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent))
    return result


def _best_by_cfg(configs: list[dict]) -> dict:
    return max(configs, key=lambda r: r["puzzle_acc"])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/denoising_budget_sweep.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    uniform_steps: list[int] = list(pb.get("uniform_steps", [100, 250, 1000]))
    cfg_scales: list[float] = list(pb.get("cfg_scales", [1.0, 1.5, 2.0, 3.0, 4.0]))
    bump_total_steps = pb.get("bump_total_steps", None)
    bump_peaks: list[float] = list(pb.get("bump_peaks", [0.05, 0.1]))
    bump_width: float = pb.get("bump_width", 0.1)
    bump_concentration: float = pb.get("bump_concentration", 4.0)
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "denoising_budget_sweep.json"))

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
        eval_ds,
        batch_size=cfg.eval.get("batch_size", cfg.train.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    num_train_timesteps = model.scheduler.config.num_train_timesteps
    bump_total_steps = bump_total_steps if bump_total_steps is not None else sorted(uniform_steps)[len(uniform_steps) // 2]

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(
        f"num_train_timesteps={num_train_timesteps}  uniform_steps={uniform_steps}  cfg_scales={cfg_scales}  "
        f"bump_total_steps={bump_total_steps}  bump_peaks={bump_peaks}  bump_width={bump_width}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict = {"uniform": {}, "bump": {}}

    for steps in uniform_steps:
        model.scheduler.set_timesteps(num_inference_steps=steps, device=device)
        timesteps = model.scheduler.timesteps.tolist()
        per_cfg = []
        for scale in cfg_scales:
            key = f"steps={steps}/cfg={scale}"
            logger.info(f"Running uniform/{key} ...")
            r = _run_config(model, classifier, cell_size, cached_batches, timesteps, scale)
            per_cfg.append(r)
            logger.info(f"  → {r}")
        results["uniform"][str(steps)] = {"grid": per_cfg, "best": _best_by_cfg(per_cfg)}

    for peak in bump_peaks:
        timesteps = _bump_density_timesteps(bump_total_steps, num_train_timesteps, peak, bump_width, bump_concentration)
        if len(timesteps) < bump_total_steps:
            logger.warning(
                f"bump peak={peak}: requested {bump_total_steps} steps, got {len(timesteps)} after "
                "dedup (sharp peak collided integer timesteps) — using the actual achieved count."
            )
        per_cfg = []
        for scale in cfg_scales:
            key = f"peak={peak}/steps={len(timesteps)}/cfg={scale}"
            logger.info(f"Running bump/{key} ...")
            r = _run_config(model, classifier, cell_size, cached_batches, timesteps, scale)
            per_cfg.append(r)
            logger.info(f"  → {r}")
        results["bump"][str(peak)] = {"grid": per_cfg, "best": _best_by_cfg(per_cfg), "nominal_steps": bump_total_steps}

    if accelerator.is_main_process:
        print("\n" + "=" * 100)
        print(f"{'config':<28}{'cfg':>6}{'steps':>8}{'calls':>8}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}")
        print("=" * 100)
        for steps, blob in results["uniform"].items():
            for r in blob["grid"]:
                print(
                    f"{'uniform/steps=' + steps:<28}{r['cfg_scale']:>6}{r['num_steps']:>8}"
                    f"{r['total_denoiser_calls']:>8}{r['cell_acc']:>10.4f}{r['puzzle_acc']:>12.4f}"
                    f"{r['constraint_puzzle_acc']:>12.4f}"
                )
            b = blob["best"]
            print(f"{'  best (steps=' + steps + ')':<28}{b['cfg_scale']:>6}{'':>8}{'':>8}{'':>10}{b['puzzle_acc']:>12.4f}")
        for peak, blob in results["bump"].items():
            for r in blob["grid"]:
                print(
                    f"{'bump/peak=' + peak:<28}{r['cfg_scale']:>6}{r['num_steps']:>8}"
                    f"{r['total_denoiser_calls']:>8}{r['cell_acc']:>10.4f}{r['puzzle_acc']:>12.4f}"
                    f"{r['constraint_puzzle_acc']:>12.4f}"
                )
            b = blob["best"]
            print(f"{'  best (peak=' + peak + ')':<28}{b['cfg_scale']:>6}{'':>8}{'':>8}{'':>10}{b['puzzle_acc']:>12.4f}")
        print("=" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
