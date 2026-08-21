"""
experiments/perturbation_recovery_probe.py — Ablation D4: per-cell
commitment points and mid-trajectory corruption recovery.

Adapts experiments/cond_swap_experiment.py's swap-and-continue idea (built
for swapping the CONDITION between samples, at a chosen denoising step) to
corrupting the model's own INTERMEDIATE PREDICTION instead, with the
condition/puzzle held fixed throughout — directly testing whether the
model can revise a corrupted committed guess, complementing D3's organic
repair-rate evidence with an externally injected perturbation.

Works uniformly on ANY PainterBase model (plain feed-forward DM,
Painter-Thinker, or TRM-alone) since it only calls model(sample) /
model.null_condition_sample / model.scheduler / model.decode_for_eval — no
forward_with_carry needed. Run it once per checkpoint and compare the two
runs' JSON outputs externally (e.g. TRM vs. experiment B's depth-matched
feedforward baseline, or TRM vs. a plain unconditional UNetPainter).

(a) Commitment points: runs the real (CFG) generation trajectory once per
cached batch, decoding + classifying the running image at EVERY denoising
step (unlike D2/D3's early/mid/late buckets — commitment detection needs
the full per-step curve). For every blank cell, the "commitment step" is
the earliest denoising step at which that cell is correct AND stays
correct through the final step (a last-flip-to-correct point, via the same
suffix-reduction trick experiments/train_halt_head.py's future-min target
uses). Cells never correct at the end have no commitment point and are
excluded from the distribution; the exclusion fraction (~= 1 - puzzle-
level accuracy's cell-level analogue) is reported alongside it.

(b) Corruption recovery: at each of probe.corrupt_fractions (positions
along the trajectory), runs the real trajectory up to that denoising step,
then for every cell that is currently (i) blank and (ii) correct,
overwrites its patch with another sample's real patch for that same cell
— borrowed via a fixed cyclic permutation across the batch, so the
corruption is a plausible real digit, not noise — keeping the swap only
where it's confirmed to actually be a different, wrong digit. Continues
the trajectory normally from this corrupted image for the remaining steps
(condition unchanged throughout) and reports the recovery rate: the
fraction of corrupted cells that are correct again by the final step.

Usage:
    python experiments/perturbation_recovery_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +probe.num_samples=256 +probe.corrupt_fractions=[0.1,0.3,0.5,0.7,0.9]

    # TRM-alone / a plain unconditional DM work identically — point
    # experiment=/checkpoint= at the other model, then diff the two JSONs.
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
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


def _decode_cellwise(images: torch.Tensor, classifier, cell_size: int) -> torch.Tensor:
    """images: (B, 1, H, W) in [0,1]. Returns (B, 81) long — per-cell
    predicted digit. Standalone (not evaluate_grids) since this needs
    calling at every denoising step and mid-corruption, not just once
    paired with accuracy aggregation."""
    B = images.shape[0]
    cells = images.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)
    return classifier(cells).argmax(dim=1).reshape(B, 81)


def _corrupt_correct_cells(
    x: torch.Tensor, solutions: torch.Tensor, given_masks, classifier, cell_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B, 1, H, W) running (pre-decode_for_eval) image. For every blank
    cell currently correct, overwrites its patch with the same cell from a
    cyclically-shifted sample in the batch — a real, plausible digit, not
    noise — keeping the swap only where the borrowed patch is confirmed a
    DIFFERENT, wrong digit (so every corruption is a genuine perturbation).

    Returns (x_corrupted, corrupted_mask) where corrupted_mask is (B, 81)
    bool — which cells were actually corrupted.
    """
    B = x.shape[0]
    sol = solutions.to(x.device)
    blank = (~given_masks.to(x.device)) if given_masks is not None else torch.ones_like(sol, dtype=torch.bool)

    preds = _decode_cellwise(x.clamp(0.0, 1.0), classifier, cell_size)
    correct_now = (preds == sol) & blank

    perm = torch.roll(torch.arange(B, device=x.device), shifts=1)
    x_perm = x[perm]
    cand_preds = _decode_cellwise(x_perm.clamp(0.0, 1.0), classifier, cell_size)
    do_swap = correct_now & (cand_preds != sol)  # (B, 81)

    x_corrupted = x.clone()
    for row in range(9):
        for col in range(9):
            cell_idx = row * 9 + col
            mask = do_swap[:, cell_idx]
            if mask.any():
                r0, r1 = row * cell_size, (row + 1) * cell_size
                c0, c1 = col * cell_size, (col + 1) * cell_size
                x_corrupted[mask, :, r0:r1, c0:c1] = x_perm[mask, :, r0:r1, c0:c1]

    return x_corrupted, do_swap


@torch.no_grad()
def _run_full_trajectory(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    classifier,
    cell_size: int,
    solutions: torch.Tensor,
    given_masks,
    corrupt_at: int | None = None,
) -> tuple[list, torch.Tensor | None]:
    """Runs the real CFG trajectory via model(sample)/null_condition_sample
    (model-agnostic — no forward_with_carry, works for any PainterBase
    subclass), decoding + classifying at EVERY denoising step.

    corrupt_at: if not None, immediately after that step's scheduler.step,
    corrupts currently-correct blank cells (see _corrupt_correct_cells) and
    continues from the corrupted x for the remaining steps — condition is
    never touched, only the running image.

    Returns (per_step_preds, corrupted_mask): per_step_preds is a length-
    num_inference_steps list of (B, 81) long tensors (per-cell predicted
    digit at every step, post-corruption for steps >= corrupt_at);
    corrupted_mask is (B, 81) bool (None if corrupt_at is None).
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    per_step_preds = []
    corrupted_mask = None

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u = model(null_sample).pred
            noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

        if corrupt_at is not None and step_idx == corrupt_at:
            x, corrupted_mask = _corrupt_correct_cells(x, solutions, given_masks, classifier, cell_size)

        per_step_preds.append(_decode_cellwise(x.clamp(0.0, 1.0), classifier, cell_size))

    return per_step_preds, corrupted_mask


def _commitment_steps(per_step_preds: list, solutions: torch.Tensor, given_masks) -> tuple[torch.Tensor, torch.Tensor]:
    """per_step_preds: length-T list of (B, 81) long. Returns
    (commitment_step, has_commitment): both (B, 81); commitment_step[b,c]
    is the earliest step index i such that preds[i:, b, c] are all correct
    (a last-flip-to-correct point); has_commitment is False for cells never
    correct at the final step (commitment_step is meaningless there, and
    they're excluded from reported distributions)."""
    device = per_step_preds[0].device
    sol = solutions.to(device)
    blank = (~given_masks.to(device)) if given_masks is not None else torch.ones_like(sol, dtype=torch.bool)

    correct = torch.stack(per_step_preds, dim=0) == sol.unsqueeze(0)  # (T, B, 81)
    T = correct.shape[0]
    # suffix_all[i] = correct[i:].all(dim=0) — same reverse-cummin trick as
    # SpatialTRM._future_min_targets (this codebase's established pattern
    # for "does everything from i to the end hold").
    suffix_all = torch.flip(torch.cummin(torch.flip(correct.int(), dims=[0]), dim=0).values, dims=[0]).bool()
    has_commitment = suffix_all.any(dim=0) & blank
    full_default = torch.full_like(suffix_all[0], T - 1, dtype=torch.int64)
    commitment_step = torch.where(has_commitment, suffix_all.int().argmax(dim=0), full_default)
    return commitment_step, has_commitment


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/perturbation_recovery_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    corrupt_fractions: list[float] = list(pb.get("corrupt_fractions", [0.1, 0.3, 0.5, 0.7, 0.9]))
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "perturbation_recovery.json"))

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

    pipeline = model.sampling_pipeline
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    corrupt_idx = sorted({round(f * (num_inference_steps - 1)) for f in corrupt_fractions})
    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  "
        f"corrupt denoise steps={corrupt_idx}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    # ── (a) Commitment points — one clean (uncorrupted) trajectory per batch ──
    all_commit_steps = []
    total_blank = 0
    for cb in cached_batches:
        per_step_preds, _ = _run_full_trajectory(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale,
            classifier, cell_size, cb["solutions"], cb["given_masks"], corrupt_at=None,
        )
        commit_step, has_commit = _commitment_steps(per_step_preds, cb["solutions"], cb["given_masks"])
        all_commit_steps.append(commit_step[has_commit].float())
        gm = cb["given_masks"]
        blank = (~gm.to(device)) if gm is not None else torch.ones_like(cb["solutions"], dtype=torch.bool, device=device)
        total_blank += int(blank.sum().item())

    pooled_commit = torch.cat(all_commit_steps) if any(t.numel() for t in all_commit_steps) else torch.zeros(0)
    total_committed = sum(t.numel() for t in all_commit_steps)
    commitment_frac = pooled_commit / max(num_inference_steps - 1, 1)

    commitment_results = {
        "mean_step": float(pooled_commit.mean().item()) if pooled_commit.numel() else None,
        "mean_frac": float(commitment_frac.mean().item()) if pooled_commit.numel() else None,
        "p50_frac": float(torch.quantile(commitment_frac, 0.5).item()) if pooled_commit.numel() else None,
        "p90_frac": float(torch.quantile(commitment_frac, 0.9).item()) if pooled_commit.numel() else None,
        "frac_never_correct": 1.0 - (total_committed / total_blank if total_blank > 0 else 0.0),
        "n_blank_cells": int(total_blank),
    }
    logger.info(f"commitment: {commitment_results}")

    # ── (b) Corruption recovery, one corrupt_idx at a time ──
    recovery_results: dict = {}
    for c_idx in corrupt_idx:
        n_corrupted, n_recovered = 0, 0
        for cb in cached_batches:
            per_step_preds, corrupted_mask = _run_full_trajectory(
                model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale,
                classifier, cell_size, cb["solutions"], cb["given_masks"], corrupt_at=c_idx,
            )
            if corrupted_mask is None or not corrupted_mask.any():
                continue
            final_preds = per_step_preds[-1]
            sol = cb["solutions"].to(device)
            recovered = (final_preds == sol) & corrupted_mask
            n_corrupted += int(corrupted_mask.sum().item())
            n_recovered += int(recovered.sum().item())

        recovery_results[str(c_idx)] = {
            "denoise_step": c_idx,
            "frac_of_trajectory": c_idx / max(num_inference_steps - 1, 1),
            "n_corrupted": n_corrupted,
            "n_recovered": n_recovered,
            "recovery_rate": (n_recovered / n_corrupted) if n_corrupted > 0 else None,
        }
        logger.info(f"corrupt_at={c_idx}  {recovery_results[str(c_idx)]}")

    results = {"commitment": commitment_results, "recovery": recovery_results}

    if accelerator.is_main_process:
        print("\n" + "=" * 90)
        print("Commitment points (blank cells, fraction of trajectory position, 0=first..1=last)")
        print("=" * 90)
        for k, v in commitment_results.items():
            print(f"  {k:<20}{v}")

        print("\n" + "=" * 90)
        print(f"{'denoise_step':>12}{'frac_traj':>11}{'n_corrupted':>13}{'n_recovered':>13}{'recovery_rate':>15}")
        print("=" * 90)
        for r in recovery_results.values():
            rr = r["recovery_rate"]
            print(
                f"{r['denoise_step']:>12}{r['frac_of_trajectory']:>11.3f}{r['n_corrupted']:>13}"
                f"{r['n_recovered']:>13}{(rr if rr is not None else float('nan')):>15.4f}"
            )
        print("=" * 90)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
