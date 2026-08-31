"""
experiments/violation_sensitivity_probe.py — does the model repair
constraint-VIOLATING injections more than constraint-SAFE ones, on the hard
(sparse-givens) checkpoint where genuine alternative completions exist?

Reuses targeted_corruption_probe.py's paired same-seed clean/corrupted
trajectory machinery (crop bank, target-cell selection, corruption
injection, recovery/adapt/collapse + valid-sudoku metrics) but adds a new
axis: which WRONG digit gets injected at a targeted cell.

  "violating"     — the injected digit already appears among the GIVEN
                    (clue) cells sharing that target cell's row, column, or
                    box. This digit cannot be part of ANY valid completion
                    of this puzzle, full stop — accepting it necessarily
                    means the grid stops satisfying the puzzle's own clues.
  "non_violating" — the injected digit does NOT appear among any given cell
                    in that row/column/box. It may or may not match the
                    dataset's one stored solution, but nothing about the
                    puzzle's fixed clues rules it out — on a hard puzzle
                    with few givens, some other valid completion may
                    genuinely use this digit here.

A model doing real constraint-checking should repair "violating" injections
at a higher rate than "non_violating" ones (there's nothing to "fix" about
a non_violating digit from the puzzle's own point of view) and should be
more willing to keep a non_violating digit while leaving the rest of the
grid valid. A model that's just pattern-matching toward one memorized
answer should treat both conditions roughly the same (any deviation from
its expected output gets corrected the same way, violating or not).

Usage:
    python experiments/violation_sensitivity_probe.py \\
      experiment=mnist_thinker_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      data=mnist_sudoku_srm data.train_dataset.given_cells_range=[0,26] \\
      checkpoint=runs/sudoku-hard-painter-thinker.pt \\
      +probe.num_samples=128 +probe.n_cells_values=[1,4,16] \\
      +probe.corrupt_fractions=[0.1,0.5,0.9] +probe.conditions=[violating,non_violating]

    # Options (all under +probe.*): same as targeted_corruption_probe.py,
    # plus conditions (default [violating, non_violating]).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
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
from eval.mnist_eval import _check_sudoku_constraints
from factory import build_datasets, build_model
from hydra.utils import instantiate
from targeted_corruption_probe import (
    _build_digit_crop_bank, _corrupt_with_wrong_digits, _run_full_trajectory_targeted, _select_target_cells,
)

logger = get_logger(__name__, log_level="INFO")


def _sample_condition_wrong_crops(
    crop_bank: dict, solutions: torch.Tensor, given_mask, target_mask: torch.Tensor, device, condition: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like targeted_corruption_probe._sample_wrong_crops, but the wrong
    digit is constrained by `condition` relative to the GIVEN cells sharing
    the target cell's row/col/box (see module docstring). Returns
    (crops, wrong_digits, success) -- success is False wherever no digit
    satisfying `condition` existed (e.g. a "violating" digit needs at least
    one given cell in the unit; excluded from corruption/metrics)."""
    B, N = solutions.shape
    cs = next(iter(crop_bank.values())).shape[-1]
    num_classes = len(crop_bank)
    crops = torch.zeros(B, N, 1, cs, cs, device=device)
    wrong_digits = solutions.clone()
    success = torch.zeros_like(target_mask)
    sol_np = solutions.cpu().numpy()
    given_np = given_mask.cpu().numpy() if given_mask is not None else np.zeros_like(sol_np, dtype=bool)
    idx_b, idx_c = target_mask.nonzero(as_tuple=True)
    for b, c in zip(idx_b.tolist(), idx_c.tolist()):
        row, col = c // 9, c % 9
        true_d = int(sol_np[b, c])
        given_digits: set = set()
        for cc in range(9):
            cell = row * 9 + cc
            if given_np[b, cell]:
                given_digits.add(int(sol_np[b, cell]))
        for rr in range(9):
            cell = rr * 9 + col
            if given_np[b, cell]:
                given_digits.add(int(sol_np[b, cell]))
        br, bc0 = (row // 3) * 3, (col // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc0, bc0 + 3):
                cell = rr * 9 + cc
                if given_np[b, cell]:
                    given_digits.add(int(sol_np[b, cell]))
        given_digits.discard(true_d)

        if condition == "violating":
            candidates = list(given_digits)
        else:
            candidates = [d for d in range(num_classes) if d != true_d and d not in given_digits]

        if not candidates:
            continue
        wrong_d = random.choice(candidates)
        bank = crop_bank[wrong_d]
        crops[b, c] = bank[random.randrange(bank.shape[0])].to(device)
        wrong_digits[b, c] = wrong_d
        success[b, c] = True
    return crops, wrong_digits, success


def _run_config(
    model, classifier, cell_size: int, cached_batches: list, crop_bank: dict,
    num_inference_steps: int, cfg_scale: float, n_cells: int, corrupt_idx: int, base_seed: int, condition: str,
) -> dict:
    total_corrupted = total_recovered = total_adapted = total_collapsed = 0
    total_recovered_and_valid = total_adapted_and_valid = 0
    total_puzzles = total_puzzle_valid_different = total_puzzle_same_as_clean = 0
    total_clean = total_clean_valid = total_clean_exact = 0

    for bi, cb in enumerate(cached_batches):
        device = cb["x_init"].device
        conditions = cb["conditions"]
        solutions = cb["solutions"].to(device)
        given_mask = cb["given_masks"].to(device) if cb["given_masks"] is not None else None
        seed = base_seed + bi

        torch.manual_seed(seed)
        x_init = torch.randn_like(cb["x_init"])

        torch.manual_seed(seed + 1)
        clean_preds = _run_full_trajectory_targeted(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size
        )
        clean_final = clean_preds[-1]
        clean_correct = clean_final == solutions

        if n_cells == 0:
            # No corruption at all -- reports the plain clean-generation
            # baseline (full trajectory from pure noise, no injection) so
            # the n_cells=0 column has a real reference point instead of
            # being skipped outright. recovery/adapt/collapse and the
            # recovered_and_valid/adapted_and_valid rates below are
            # genuinely undefined here (nothing was corrupted), left as
            # None; clean_puzzle_valid_rate/clean_puzzle_exact_rate are
            # this branch's own two numbers instead.
            total_clean += clean_final.shape[0]
            total_clean_valid += int(_check_sudoku_constraints(clean_final).sum().item())
            total_clean_exact += int(clean_correct.all(dim=1).sum().item())
            continue

        target_mask = _select_target_cells(clean_correct, given_mask, n_cells)
        if not target_mask.any():
            continue

        crops, wrong_digits, success = _sample_condition_wrong_crops(
            crop_bank, solutions, given_mask, target_mask, device, condition
        )
        target_mask = target_mask & success
        if not target_mask.any():
            continue

        torch.manual_seed(seed + 1)
        corrupted_preds = _run_full_trajectory_targeted(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size,
            corrupt_at=corrupt_idx, target_mask=target_mask, clean_wrong_crops=crops,
        )
        corrupted_final = corrupted_preds[-1]

        recovered = (corrupted_final == solutions) & target_mask
        adapted = (~recovered) & (corrupted_final == wrong_digits) & target_mask
        collapsed = target_mask & (~recovered) & (~adapted)
        total_corrupted += int(target_mask.sum().item())
        total_recovered += int(recovered.sum().item())
        total_adapted += int(adapted.sum().item())
        total_collapsed += int(collapsed.sum().item())

        valid_grid_per_sample = _check_sudoku_constraints(corrupted_final)
        valid_broadcast = valid_grid_per_sample.unsqueeze(1).expand_as(target_mask)
        total_recovered_and_valid += int((recovered & valid_broadcast).sum().item())
        total_adapted_and_valid += int((adapted & valid_broadcast).sum().item())

        blank_all = (~given_mask) if given_mask is not None else torch.ones_like(solutions, dtype=torch.bool)
        puzzle_exact = ((corrupted_final == solutions) | ~blank_all).all(dim=1)
        puzzle_valid_different = valid_grid_per_sample & (~puzzle_exact)
        puzzle_same_as_clean = (corrupted_final == clean_final).all(dim=1)
        puzzle_has_target = target_mask.any(dim=1)

        total_puzzles += int(puzzle_has_target.sum().item())
        total_puzzle_valid_different += int((puzzle_valid_different & puzzle_has_target).sum().item())
        total_puzzle_same_as_clean += int((puzzle_same_as_clean & puzzle_has_target).sum().item())

    return {
        "condition": condition,
        "n_cells": n_cells,
        "corrupt_idx": corrupt_idx,
        "n_corrupted": total_corrupted,
        "recovery_rate": (total_recovered / total_corrupted) if total_corrupted else None,
        "adapt_rate": (total_adapted / total_corrupted) if total_corrupted else None,
        "collapse_rate": (total_collapsed / total_corrupted) if total_corrupted else None,
        "recovered_and_valid_rate": (total_recovered_and_valid / total_corrupted) if total_corrupted else None,
        "adapted_and_valid_rate": (total_adapted_and_valid / total_corrupted) if total_corrupted else None,
        "n_puzzles": total_puzzles,
        "puzzle_valid_different_rate": (total_puzzle_valid_different / total_puzzles) if total_puzzles else None,
        "puzzle_same_as_clean_rate": (total_puzzle_same_as_clean / total_puzzles) if total_puzzles else None,
        "n_clean": total_clean,
        "clean_puzzle_valid_rate": (total_clean_valid / total_clean) if total_clean else None,
        "clean_puzzle_exact_rate": (total_clean_exact / total_clean) if total_clean else None,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/violation_sensitivity_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 128)
    seed: int = pb.get("seed", 0)
    n_cells_values: list[int] = list(pb.get("n_cells_values", [1, 4, 16]))
    corrupt_fractions: list[float] = list(pb.get("corrupt_fractions", [0.1, 0.5, 0.9]))
    conditions: list[str] = list(pb.get("conditions", ["violating", "non_violating"]))
    crop_bank_max_per_class: int = pb.get("crop_bank_max_per_class", 32)
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "violation_sensitivity.json"))

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

    corrupt_idx = sorted({round(f * (num_inference_steps - 1)) for f in corrupt_fractions})
    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  corrupt_idx={corrupt_idx}  n_cells_values={n_cells_values}  conditions={conditions}")

    logger.info("Building digit crop bank from real dataset images...")
    crop_bank = _build_digit_crop_bank(eval_dl, cell_size, max_per_class=crop_bank_max_per_class)
    crop_bank = {d: v.to(device) for d, v in crop_bank.items()}

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict = {}
    if 0 in n_cells_values:
        # No corruption at all, no dependence on condition or corrupt_idx --
        # run once, not once per (condition, corrupt_idx) combo.
        key = "clean/n_cells=0"
        logger.info(f"Running {key} ...")
        r = _run_config(
            model, classifier, cell_size, cached_batches, crop_bank,
            num_inference_steps, cfg_scale, 0, corrupt_idx[0], seed, "clean",
        )
        results[key] = r
        logger.info(f"  → {r}")

    for condition in conditions:
        for n_cells in n_cells_values:
            if n_cells == 0:
                continue
            for c_idx in corrupt_idx:
                key = f"{condition}/n_cells={n_cells}/corrupt_idx={c_idx}"
                logger.info(f"Running {key} ...")
                r = _run_config(
                    model, classifier, cell_size, cached_batches, crop_bank,
                    num_inference_steps, cfg_scale, n_cells, c_idx, seed, condition,
                )
                results[key] = r
                logger.info(f"  → {r}")

    if accelerator.is_main_process:
        print("\n" + "=" * 130)
        print(f"{'config':<38}{'n_corrupt':>10}{'recovery':>10}{'adapt':>9}{'rec+valid':>11}{'adapt+valid':>13}{'same_as_clean':>15}")
        print("=" * 130)
        for key, r in results.items():
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "n/a"
            print(
                f"{key:<38}{r['n_corrupted']:>10}{_fmt(r['recovery_rate']):>10}{_fmt(r['adapt_rate']):>9}"
                f"{_fmt(r['recovered_and_valid_rate']):>11}{_fmt(r['adapted_and_valid_rate']):>13}{_fmt(r['puzzle_same_as_clean_rate']):>15}"
            )
        print("=" * 130)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
