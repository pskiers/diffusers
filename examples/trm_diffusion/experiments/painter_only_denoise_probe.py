"""
experiments/painter_only_denoise_probe.py — how much can the FROZEN painter
alone (no TRM/thinker reasoning at all) recover, as a function of how much
noise it has to denoise away, and how it reacts to a wrong-digit corruption
injected at that same noise level?

Motivation: every other ablation in this session measures the TRM+painter
system end-to-end starting from pure noise. This probe isolates the painter
by itself — steering=None, no thinker call anywhere — as the "what would a
plain diffusion denoiser do with zero reasoning" baseline. Comparing this
against the TRM ablations tells us whether the TRM is doing real
puzzle-solving work or whether the painter's own denoising prior already
gets most of the way there once enough of the grid is visible.

Two conditions per starting timestep t_start:
  1. Clean: take the real solved-grid image (dataset's `images` field, i.e.
     the actual correct sudoku), noise it to t_start via scheduler.add_noise,
     then run the ordinary sampling loop (steering=None) from t_start down
     to t=0. Reports the same cell_acc/puzzle_acc/constraint_puzzle_acc/
     given_consistent_puzzle_acc as every other eval in this codebase
     (eval.mnist_eval.evaluate_grids).
  2. Corrupted: same as above, but immediately after noising to t_start,
     n_cells blank cells get a real MNIST crop of a DIFFERENT (wrong) digit
     class spliced in (noised to the same t_start via scheduler.add_noise —
     the exact mechanism from targeted_corruption_probe.py's
     _corrupt_with_wrong_digits), before denoising. Reports the same
     recover/adapt/collapse + puzzle_exact/puzzle_valid_different metrics as
     targeted_corruption_probe.py, since there the ground-truth solution IS
     the starting point (not a prior model rollout), every blank cell counts
     as a valid corruption candidate — no need to first identify which cells
     a clean rollout got right.

No TRM checkpoint is needed at all — the painter's own pretrained weights
are loaded automatically from painter.checkpoint= at model-build time, and
the thinker is instantiated but never called.

Usage:
    python experiments/painter_only_denoise_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +probe.num_samples=512 +probe.t_starts=[0,5,20,35,50,65,80,95] \\
      +probe.n_cells_values=[0,1,2,4]

    # Options (all under +probe.*):
    #   num_samples             — default 256
    #   seed                    — default 0
    #   t_starts                — denoising start points, must be members of
    #                             the num_inference_steps-step schedule
    #                             (default [0,5,20,35,50,65,80,95])
    #   n_cells_values          — 0 means the clean (uncorrupted) condition;
    #                             >0 means inject that many wrong digits
    #                             (default [0,1,2,4])
    #   crop_bank_max_per_class — real digit crops cached per class (default 32)
    #   num_inference_steps     — default from model.sampling_pipeline
    #   out                     — json path (default: alongside checkpoint)
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

from ablate_trm_loop_budget import _build_cached_batches
from eval.mnist_eval import _check_sudoku_constraints, evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate
from targeted_corruption_probe import _build_digit_crop_bank, _corrupt_with_wrong_digits, _sample_wrong_crops, _select_target_cells

logger = get_logger(__name__, log_level="INFO")


def _get_painter(model):
    """model.painter for a ThinkerFrozenPainterBase built around a frozen
    painter checkpoint; model itself when mode=painter_base built the
    painter directly as a standalone model (e.g. ConcatConditionedUNetPainter
    trained via train_trm.py experiment=mnist_unet_concat_painter_srm)."""
    return getattr(model, "painter", model)


@torch.no_grad()
def _denoise_from(model, sample, x_t: torch.Tensor, run_timesteps: torch.Tensor, cfg_scale: float = 1.0) -> torch.Tensor:
    """Runs the ordinary painter-only sampling loop (steering=None, no
    thinker involved) starting from x_t at run_timesteps[0] down through
    run_timesteps[-1] (== 0). run_timesteps must be a suffix of the full
    model.scheduler.timesteps array set by the caller's set_timesteps call,
    so scheduler.step()'s internal spacing math stays correct.

    CFG only actually engages when the painter has a droppable condition
    (condition_keys non-empty, e.g. ConcatConditionedUNetPainter's
    spatial_conditions channel-concat) -- for the original unconditional
    painter (condition_keys=[]) null_condition_sample is a no-op, so cond
    and uncond predictions are identical and the combination collapses back
    to the plain cond prediction; skipped outright there to avoid paying
    for a second, pointless forward pass."""
    painter = _get_painter(model)
    use_cfg = cfg_scale != 1.0 and len(painter.condition_keys) > 0
    x = x_t
    for t in run_timesteps:
        t_batch = t.expand(x.shape[0]).to(x.device)
        step_sample = dataclasses.replace(sample, x_noisy=x, timesteps=t_batch)
        noise_pred = painter(step_sample, steering=None).pred
        if use_cfg:
            null_sample = painter.null_condition_sample(step_sample)
            noise_pred_u = painter(null_sample, steering=None).pred
            noise_pred = noise_pred_u + cfg_scale * (noise_pred - noise_pred_u)
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return x


def _run_config(
    model, classifier, cell_size: int, cached_batches: list, crop_bank: dict,
    run_timesteps: torch.Tensor, t_start: int, n_cells: int, base_seed: int, cfg_scale: float = 1.0,
) -> dict:
    total = correct_sum = puzzle_exact_sum = constraint_sum = given_consistent_sum = 0
    blank_total = blank_correct_sum = 0
    total_corrupted = total_recovered = total_adapted = total_collapsed = 0
    total_collateral_candidates = total_collateral_broken = 0
    total_puzzles = total_puzzle_exact_corrupt = total_puzzle_valid_different = 0

    for bi, cb in enumerate(cached_batches):
        device = cb["conditions"].images.device
        images = cb["conditions"].images
        solutions = cb["solutions"].to(device)
        given_masks = cb["given_masks"].to(device) if cb["given_masks"] is not None else None
        B = images.shape[0]
        seed = base_seed + bi

        torch.manual_seed(seed)
        noise = torch.randn_like(images)
        t_start_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t = model.scheduler.add_noise(images, noise, t_start_batch)

        target_mask = None
        wrong_digits = None
        if n_cells > 0:
            all_blank_ok = torch.ones_like(given_masks) if given_masks is not None else torch.ones(B, 81, dtype=torch.bool, device=device)
            target_mask = _select_target_cells(all_blank_ok, given_masks, n_cells)
            if target_mask.any():
                clean_wrong_crops, wrong_digits = _sample_wrong_crops(crop_bank, solutions, target_mask, device)
                x_t = _corrupt_with_wrong_digits(x_t, t_start_batch, target_mask, clean_wrong_crops, cell_size, model.scheduler)
            else:
                target_mask = None

        x_final = _denoise_from(model, cb["conditions"], x_t, run_timesteps, cfg_scale=cfg_scale)
        result = evaluate_grids(x_final.clamp(0.0, 1.0), solutions, classifier, cell_size, given_masks=given_masks)
        preds = result["preds"].to(device)
        correct = preds == solutions

        total += B
        correct_sum += int(correct.all(dim=1).sum().item())
        constraint_sum += int(_check_sudoku_constraints(preds).sum().item())
        if given_masks is not None:
            blank = ~given_masks
            blank_total += int(blank.sum().item())
            blank_correct_sum += int(correct[blank].sum().item())
            given_ok = (correct | blank).all(dim=1)
            given_consistent_sum += int((_check_sudoku_constraints(preds) & given_ok).sum().item())

        if target_mask is not None:
            recovered = (preds == solutions) & target_mask
            adapted = (~recovered) & (preds == wrong_digits) & target_mask
            collapsed = target_mask & (~recovered) & (~adapted)
            total_corrupted += int(target_mask.sum().item())
            total_recovered += int(recovered.sum().item())
            total_adapted += int(adapted.sum().item())
            total_collapsed += int(collapsed.sum().item())

            collateral_candidates = (~given_masks if given_masks is not None else torch.ones_like(target_mask)) & (~target_mask)
            collateral_broken = collateral_candidates & (preds != solutions)
            total_collateral_candidates += int(collateral_candidates.sum().item())
            total_collateral_broken += int(collateral_broken.sum().item())

            blank_all = (~given_masks) if given_masks is not None else torch.ones_like(solutions, dtype=torch.bool)
            puzzle_exact = ((preds == solutions) | ~blank_all).all(dim=1)
            valid_grid = _check_sudoku_constraints(preds)
            puzzle_valid_different = valid_grid & (~puzzle_exact)
            puzzle_has_target = target_mask.any(dim=1)

            total_puzzles += int(puzzle_has_target.sum().item())
            total_puzzle_exact_corrupt += int((puzzle_exact & puzzle_has_target).sum().item())
            total_puzzle_valid_different += int((puzzle_valid_different & puzzle_has_target).sum().item())

    out = {
        "t_start": t_start,
        "n_cells": n_cells,
        "n_samples": total,
        "puzzle_acc": correct_sum / total if total else None,
        "cell_acc": (blank_correct_sum / blank_total) if blank_total else None,
        "constraint_puzzle_acc": constraint_sum / total if total else None,
        "given_consistent_puzzle_acc": (given_consistent_sum / total) if given_masks is not None else None,
    }
    if n_cells > 0:
        out.update({
            "n_corrupted": total_corrupted,
            "recovery_rate": (total_recovered / total_corrupted) if total_corrupted else None,
            "adapt_rate": (total_adapted / total_corrupted) if total_corrupted else None,
            "collapse_rate": (total_collapsed / total_corrupted) if total_corrupted else None,
            "n_collateral_candidates": total_collateral_candidates,
            "collateral_break_rate": (total_collateral_broken / total_collateral_candidates) if total_collateral_candidates else None,
            "n_puzzles": total_puzzles,
            "puzzle_exact_rate": (total_puzzle_exact_corrupt / total_puzzles) if total_puzzles else None,
            "puzzle_valid_different_rate": (total_puzzle_valid_different / total_puzzles) if total_puzzles else None,
        })
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    t_starts: list[int] = list(pb.get("t_starts", [0, 5, 20, 35, 50, 65, 80, 95]))
    n_cells_values: list[int] = list(pb.get("n_cells_values", [0, 1, 2, 4]))
    crop_bank_max_per_class: int = pb.get("crop_bank_max_per_class", 32)
    out_path: str = pb.get("out", "runs/painter_only_denoise_probe.json")

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds, batch_size=cfg.eval.get("batch_size", cfg.train.batch_size), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is not None:
        from ablate_trm_loop_budget import _load_checkpoint
        _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    full_timesteps = model.scheduler.timesteps
    valid_t = set(int(t.item()) for t in full_timesteps)
    bad = [t for t in t_starts if t not in valid_t]
    if bad:
        raise SystemExit(
            f"t_starts {bad} are not members of the {num_inference_steps}-step schedule {sorted(valid_t, reverse=True)}. "
            "Pick t_start values from that list."
        )
    logger.info(f"num_inference_steps={num_inference_steps}  cfg_scale={cfg_scale}  schedule={[int(t.item()) for t in full_timesteps]}  t_starts={t_starts}  n_cells_values={n_cells_values}")

    logger.info("Building digit crop bank from real dataset images...")
    crop_bank = _build_digit_crop_bank(eval_dl, cell_size, max_per_class=crop_bank_max_per_class)
    crop_bank = {d: v.to(device) for d, v in crop_bank.items()}

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict = {}
    for t_start in t_starts:
        run_timesteps = full_timesteps[full_timesteps <= t_start]
        for n_cells in n_cells_values:
            key = f"t_start={t_start}/n_cells={n_cells}"
            logger.info(f"Running {key} ...")
            r = _run_config(model, classifier, cell_size, cached_batches, crop_bank, run_timesteps, t_start, n_cells, seed, cfg_scale=cfg_scale)
            results[key] = r
            logger.info(f"  → {r}")

    if accelerator.is_main_process:
        print("\n" + "=" * 130)
        print(f"{'config':<24}{'puzzle_acc':>11}{'cell_acc':>10}{'constr':>9}{'given_cons':>11}{'recovery':>10}{'adapt':>8}{'collapse':>9}")
        print("=" * 130)
        for key, r in results.items():
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "n/a"
            print(
                f"{key:<24}{_fmt(r['puzzle_acc']):>11}{_fmt(r['cell_acc']):>10}{_fmt(r['constraint_puzzle_acc']):>9}"
                f"{_fmt(r['given_consistent_puzzle_acc']):>11}{_fmt(r.get('recovery_rate')):>10}"
                f"{_fmt(r.get('adapt_rate')):>8}{_fmt(r.get('collapse_rate')):>9}"
            )
        print("=" * 130)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
