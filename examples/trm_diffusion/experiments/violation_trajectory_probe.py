"""
experiments/violation_trajectory_probe.py — does constraint-violation count
fall over the denoising trajectory (evidence of active constraint-checking),
or does it lock in early as pixels sharpen (evidence of blind pattern
completion)?

Decodes the x0 prediction at EVERY denoising step, from pure noise, for both:
  DM — the frozen painter alone (steering=None, no thinker at all).
  PT — the full TRM+painter system (real reasoning, no corruption).
and counts sudoku constraint violations (how many of the 27 units -- 9 rows,
9 cols, 9 boxes -- contain at least one duplicate digit) at each step,
averaged across the batch. Plotting the two curves on the same axes is the
direct evidence for whether the system is doing real constraint-checking
work over the trajectory (PT should trend down; a model with no evaluation
mechanism should flatten out early, once pixels are confident enough to
stop changing classification).

No corruption involved here -- this is the plain generative trajectory.

Usage:
    python experiments/violation_trajectory_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/sudoku-painter-thinker.pt \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +probe.num_samples=512 +probe.out=runs/violation_trajectory_easy.json

    # Options (all under +probe.*):
    #   num_samples — default 256
    #   seed        — default 0
    #   cfg_scale / num_inference_steps — default from model.sampling_pipeline
    #   out         — json path
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from eval.mnist_eval import _check_sudoku_constraints
from factory import build_datasets, build_model
from hydra.utils import instantiate
from perturbation_recovery_probe import _decode_cellwise

logger = get_logger(__name__, log_level="INFO")


def _count_violating_units(preds: torch.Tensor) -> torch.Tensor:
    """preds: (B, 81) int64, values 0-8. Returns (B,) int64 -- how many of
    the 27 units (9 rows, 9 cols, 9 boxes) contain at least one duplicate
    digit (i.e. are NOT a permutation of 0-8). 0 = fully valid grid, matching
    _check_sudoku_constraints's row_ok/col_ok/box_ok booleans but summed
    into a graded count instead of a single valid/invalid flag."""
    B = preds.shape[0]
    grid = preds.reshape(B, 9, 9)
    expected = torch.arange(9, device=preds.device)
    violations = torch.zeros(B, dtype=torch.int64, device=preds.device)
    for i in range(9):
        row_bad = ~grid[:, i, :].sort(dim=1).values.eq(expected).all(dim=1)
        col_bad = ~grid[:, :, i].sort(dim=1).values.eq(expected).all(dim=1)
        br, bc = (i // 3) * 3, (i % 3) * 3
        box = grid[:, br:br + 3, bc:bc + 3].reshape(B, 9)
        box_bad = ~box.sort(dim=1).values.eq(expected).all(dim=1)
        violations += row_bad.int() + col_bad.int() + box_bad.int()
    return violations


@torch.no_grad()
def _run_pt_trajectory(model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size) -> list:
    """Full TRM+painter generation from pure noise, decoding at every step.
    No halting, no carry-reset shenanigans -- the model's own default
    forward (fixed n_sup, resets every step, matching how it was trained).

    Decodes noise_pred (the model's direct output BEFORE scheduler.step)
    rather than the post-step x. Under prediction_type="sample" (this
    pipeline's config), noise_pred IS the model's clean x0 estimate;
    scheduler.step()'s returned .prev_sample is a deliberate blend of that
    estimate with the still-noisy input sample plus freshly injected noise
    for every step except the last -- decoding THAT instead reads mostly as
    noise until the trajectory nearly finishes, regardless of how good the
    model's actual prediction is. (The very last step's prev_sample happens
    to collapse exactly to noise_pred once alpha_prod_prev=1 and
    variance=0, which is why final-step accuracy numbers elsewhere in this
    codebase were never affected by this.)"""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    per_step_preds = []
    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u = model(null_sample).pred
            noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        per_step_preds.append(_decode_cellwise(noise_pred.clamp(0.0, 1.0), classifier, cell_size))
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return per_step_preds


@torch.no_grad()
def _run_dm_trajectory(model, conditions, x_init, num_inference_steps, classifier, cell_size) -> list:
    """Painter alone (steering=None, no thinker call at all), from pure
    noise, decoding at every step. See _run_pt_trajectory's docstring for
    why noise_pred (pre-scheduler.step) is decoded instead of x."""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    per_step_preds = []
    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        noise_pred = model.painter(step_sample, steering=None).pred
        per_step_preds.append(_decode_cellwise(noise_pred.clamp(0.0, 1.0), classifier, cell_size))
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return per_step_preds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/violation_trajectory_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "violation_trajectory.json"))

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

    pipeline = model.sampling_pipeline
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in model.scheduler.timesteps]

    pt_violations = [[] for _ in range(num_inference_steps)]
    dm_violations = [[] for _ in range(num_inference_steps)]
    n_done = 0
    for bi, cb in enumerate(cached_batches):
        conditions = cb["conditions"]
        torch.manual_seed(seed + bi)
        x_init = torch.randn_like(cb["x_init"])

        pt_preds = _run_pt_trajectory(model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size)
        dm_preds = _run_dm_trajectory(model, conditions, x_init, num_inference_steps, classifier, cell_size)

        for step_idx in range(num_inference_steps):
            pt_violations[step_idx].append(_count_violating_units(pt_preds[step_idx]))
            dm_violations[step_idx].append(_count_violating_units(dm_preds[step_idx]))
        n_done += x_init.shape[0]
        logger.info(f"  batch {bi+1}/{len(cached_batches)} done ({n_done} samples)")

    results = {"timesteps": timesteps, "pt_mean_violations": [], "dm_mean_violations": []}
    for step_idx in range(num_inference_steps):
        pt_cat = torch.cat(pt_violations[step_idx])
        dm_cat = torch.cat(dm_violations[step_idx])
        results["pt_mean_violations"].append(float(pt_cat.float().mean().item()))
        results["dm_mean_violations"].append(float(dm_cat.float().mean().item()))

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print(f"{'timestep':>10}{'PT violations':>16}{'DM violations':>16}")
        print("=" * 60)
        for i, t in enumerate(timesteps):
            print(f"{t:>10}{results['pt_mean_violations'][i]:>16.3f}{results['dm_mean_violations'][i]:>16.3f}")
        print("=" * 60)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": n_done, **results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
