"""
experiments/targeted_corruption_probe.py — paired same-seed corruption +
recovery, with controlled wrong-digit injection, collateral-damage
tracking, and recovery-latency measurement.

Refines experiments/perturbation_recovery_probe.py's corruption-recovery
axis in three ways:

  1. Paired same-seed comparison: for each cached batch, a CLEAN
     trajectory is run first (fixed seed) to identify which (sample,
     cell) pairs the model gets right on its own — the candidate pool for
     corruption. The SAME x_init AND the same RNG state going into
     scheduler.step()'s internal per-step noise draws are then replayed
     for a SECOND, corrupted trajectory (same seed reset before each run,
     same number/shape of random draws either way), so the two runs are
     identical except for the injected corruption itself.
  2. Controlled wrong-digit injection: instead of borrowing whatever
     digit a random other sample in the batch happens to have (the
     original perturbation_recovery_probe.py mechanism), a SPECIFIC wrong
     digit is chosen per corrupted cell and injected via a real MNIST
     crop — drawn from a small bank built once from the dataset's own
     clean images — noised to the injection timestep via
     scheduler.add_noise. This gives a properly-noised, plausible
     corruption of a KNOWN, chosen class, not an arbitrary borrowed one.
  3. Collateral damage + recovery latency, beyond D4's plain end-of-
     trajectory recovery rate: (a) whether previously-correct, NON-
     corrupted cells get disturbed by the correction process, and (b) at
     which subsequent denoising step each corrupted cell first flips back
     to correct and STAYS correct (a last-flip-to-correct point, not just
     whether it happened to be correct at the very end).

A "silent" (non-constraint-violating) corruption variant was considered
and dropped: in a complete, valid sudoku solution every row/column/box
already contains each digit exactly once, so ANY single-cell substitution
necessarily creates a duplicate somewhere (the row loses its only
occurrence of the true digit and gains a second occurrence of the wrong
one). There is no way to corrupt an already-correct cell without
violating at least the row constraint.

Per-cell outcome (recover/adapt/collapse) vs. per-puzzle outcome:
"adapt" here means a corrupted CELL keeps the exact injected wrong digit
(vs. reverting to truth, "recover", or drifting to some third value,
"collapse") -- a per-cell, per-digit measurement. This does NOT check
whether the model repaired the REST of the grid around that kept digit
into a globally coherent puzzle. That's a separate, per-PUZZLE question --
"puzzle_valid_different_rate" below -- checking whether the full 81-cell
output satisfies every sudoku constraint (via the same
_check_sudoku_constraints used for constraint_puzzle_acc elsewhere in this
codebase) while still NOT matching the ground-truth solution exactly. A
puzzle can score high on cell-level "adapt" while still failing this (kept
the wrong digit, but never fixed up the rest into a valid grid), or the
other way around for a single corrupted cell (kept the digit, and the rest
of the grid was already/still valid around it).

Usage:
    python experiments/targeted_corruption_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +probe.num_samples=64 +probe.n_cells_values=[1,2,4] \\
      +probe.corrupt_fractions=[0.1,0.3,0.5,0.7,0.9]

    # Options (all under +probe.*):
    #   num_samples             — default 64
    #   seed                    — default 0 (also seeds x_init + scheduler RNG per batch)
    #   n_cells_values          — number of cells corrupted per trial (default [1,2,4])
    #   corrupt_fractions       — positions along the trajectory to inject at
    #                             (default [0.1,0.3,0.5,0.7,0.9])
    #   crop_bank_max_per_class — real digit crops cached per class 0-8 (default 32)
    #   cfg_scale / num_inference_steps — default from model.sampling_pipeline
    #   out                     — json path (default: alongside checkpoint)
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
from perturbation_recovery_probe import _decode_cellwise

logger = get_logger(__name__, log_level="INFO")


def _build_digit_crop_bank(dataloader, cell_size: int, num_classes: int = 9, max_per_class: int = 32) -> dict:
    """Scans real dataset batches' clean `images` field for real MNIST
    digit crops per digit class (0-8) — the source for controlled
    wrong-digit corruption (see module docstring, point 2). Returns
    {digit: (N, 1, cell_size, cell_size)} (N up to max_per_class)."""
    bank: dict[int, list] = {d: [] for d in range(num_classes)}
    for batch in dataloader:
        images = batch["images"]
        solutions = batch["solution"]
        B = images.shape[0]
        for b in range(B):
            for cell_idx in range(81):
                d = int(solutions[b, cell_idx].item())
                if 0 <= d < num_classes and len(bank[d]) < max_per_class:
                    row, col = cell_idx // 9, cell_idx % 9
                    r0, r1 = row * cell_size, (row + 1) * cell_size
                    c0, c1 = col * cell_size, (col + 1) * cell_size
                    bank[d].append(images[b, :, r0:r1, c0:c1].clone())
        if all(len(v) >= max_per_class for v in bank.values()):
            break
    missing = [d for d, v in bank.items() if not v]
    if missing:
        raise RuntimeError(f"Could not find example crops for digit classes {missing} in the scanned batches.")
    return {d: torch.stack(v, dim=0) for d, v in bank.items()}


def _select_target_cells(clean_correct: torch.Tensor, given_mask, n_cells: int) -> torch.Tensor:
    """clean_correct: (B, 81) bool — correct in the clean run. given_mask:
    (B, 81) bool or None (True=given). Returns target_mask (B, 81) bool,
    up to n_cells randomly chosen per sample among (correct & blank)
    candidates — corrupting a cell the model already got wrong isn't a
    meaningful "derail". Samples with fewer than n_cells candidates get
    whatever is available (possibly zero)."""
    blank = (~given_mask) if given_mask is not None else torch.ones_like(clean_correct)
    candidates = clean_correct & blank
    target_mask = torch.zeros_like(candidates)
    B = candidates.shape[0]
    for b in range(B):
        idx = candidates[b].nonzero(as_tuple=True)[0].tolist()
        random.shuffle(idx)
        for c in idx[:n_cells]:
            target_mask[b, c] = True
    return target_mask


def _sample_wrong_crops(
    crop_bank: dict, solutions: torch.Tensor, target_mask: torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (clean_wrong_crops, wrong_digits):
      clean_wrong_crops — (B, 81, 1, cs, cs) un-noised real digit crops of a
        specific, randomly-chosen WRONG class (!= the true solution digit)
        for every targeted cell; entries outside target_mask are unused zeros.
      wrong_digits — (B, 81) long, the specific wrong class injected at
        each targeted cell (unused entries left as the true solution digit,
        i.e. never mistakable for "wrong" outside target_mask) — needed to
        tell "adapted" (model keeps this exact wrong digit) apart from
        "collapsed" (model drifts to some THIRD value) later.
    """
    B, N = solutions.shape
    cs = next(iter(crop_bank.values())).shape[-1]
    num_classes = len(crop_bank)
    crops = torch.zeros(B, N, 1, cs, cs, device=device)
    wrong_digits = solutions.clone()
    sol_np = solutions.cpu().numpy()
    idx_b, idx_c = target_mask.nonzero(as_tuple=True)
    for b, c in zip(idx_b.tolist(), idx_c.tolist()):
        true_d = int(sol_np[b, c])
        wrong_d = random.choice([d for d in range(num_classes) if d != true_d])
        bank = crop_bank[wrong_d]
        crops[b, c] = bank[random.randrange(bank.shape[0])].to(device)
        wrong_digits[b, c] = wrong_d
    return crops, wrong_digits


def _corrupt_with_wrong_digits(
    x: torch.Tensor, t_batch: torch.Tensor, target_mask: torch.Tensor, clean_wrong_crops: torch.Tensor,
    cell_size: int, scheduler,
) -> torch.Tensor:
    """Noises each targeted cell's chosen wrong-digit crop to the current
    timestep via scheduler.add_noise (so it's a plausible x_t for THIS
    point in the trajectory, not an out-of-distribution artifact) and
    splices it into x."""
    x = x.clone()
    for row in range(9):
        for col in range(9):
            cell_idx = row * 9 + col
            mask = target_mask[:, cell_idx]
            if not mask.any():
                continue
            r0, r1 = row * cell_size, (row + 1) * cell_size
            c0, c1 = col * cell_size, (col + 1) * cell_size
            clean_crop = clean_wrong_crops[mask, cell_idx]
            noise = torch.randn_like(clean_crop)
            noisy_crop = scheduler.add_noise(clean_crop, noise, t_batch[mask])
            x[mask, :, r0:r1, c0:c1] = noisy_crop.to(x.dtype)
    return x


@torch.no_grad()
def _run_full_trajectory_targeted(
    model, conditions, x_init: torch.Tensor, num_inference_steps: int, cfg_scale: float, classifier, cell_size: int,
    corrupt_at: int | None = None, target_mask: torch.Tensor | None = None, clean_wrong_crops: torch.Tensor | None = None,
) -> list:
    """Runs one real CFG trajectory via model(sample)/null_condition_sample
    (model-agnostic, same as perturbation_recovery_probe.py), decoding +
    classifying at EVERY denoising step. If corrupt_at is given, injects
    the targeted wrong-digit corruption immediately after that step's
    scheduler.step() and continues normally — condition is never touched,
    only the running image. Returns per_step_preds (length
    num_inference_steps list of (B, 81) long tensors)."""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    per_step_preds = []

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
            x = _corrupt_with_wrong_digits(x, t_batch, target_mask, clean_wrong_crops, cell_size, model.scheduler)

        per_step_preds.append(_decode_cellwise(x.clamp(0.0, 1.0), classifier, cell_size))

    return per_step_preds


def _run_config(
    model, classifier, cell_size: int, cached_batches: list, crop_bank: dict,
    num_inference_steps: int, cfg_scale: float, n_cells: int, corrupt_idx: int, base_seed: int,
) -> dict:
    total_corrupted = total_recovered = total_adapted = total_collapsed = 0
    total_recovered_and_valid = total_adapted_and_valid = 0
    total_collateral_candidates = total_collateral_broken = 0
    total_puzzles = total_puzzle_exact = total_puzzle_valid_different = 0
    total_puzzle_same_as_clean = 0
    latencies: list[float] = []

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

        target_mask = _select_target_cells(clean_correct, given_mask, n_cells)
        if not target_mask.any():
            continue

        clean_wrong_crops, wrong_digits = _sample_wrong_crops(crop_bank, solutions, target_mask, device)

        # Same seed as the clean run — replays identical RNG consumption
        # (scheduler.step()'s internal per-step noise draws) so the two
        # trajectories diverge only where the corruption itself causes it.
        torch.manual_seed(seed + 1)
        corrupted_preds = _run_full_trajectory_targeted(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size,
            corrupt_at=corrupt_idx, target_mask=target_mask, clean_wrong_crops=clean_wrong_crops,
        )
        corrupted_final = corrupted_preds[-1]

        # 1. Recovery vs. adapt vs. collapse — mutually exclusive, covers
        # every corrupted cell. "Adapted" means the model kept the EXACT
        # injected wrong digit (accepted it as new information and, if it
        # touched anything, resolved around it) rather than drifting to
        # some other, third value ("collapsed" — neither the truth nor the
        # injected digit stuck).
        recovered = (corrupted_final == solutions) & target_mask
        adapted = (~recovered) & (corrupted_final == wrong_digits) & target_mask
        collapsed = target_mask & (~recovered) & (~adapted)
        total_corrupted += int(target_mask.sum().item())
        total_recovered += int(recovered.sum().item())
        total_adapted += int(adapted.sum().item())
        total_collapsed += int(collapsed.sum().item())

        # 1b. Same recover/adapt split, but jointly with whether the FULL
        # grid the cell sits in is a valid sudoku (broadcasting the
        # per-sample valid_grid flag across that sample's target cells).
        # Subsets of recovered/adapted above (recovered_and_valid_rate <=
        # recovery_rate, etc) -- "of the cells that came back to the truth
        # (or kept the wrong digit), how many left a globally consistent
        # grid" -- avoids puzzle_exact_rate's all-81-cells-must-match-the-
        # one-stored-solution requirement, which collapses to ~0 whenever a
        # puzzle has multiple valid completions (see hard checkpoint).
        valid_grid_per_sample = _check_sudoku_constraints(corrupted_final)  # (B,)
        valid_broadcast = valid_grid_per_sample.unsqueeze(1).expand_as(target_mask)  # (B, 81)
        total_recovered_and_valid += int((recovered & valid_broadcast).sum().item())
        total_adapted_and_valid += int((adapted & valid_broadcast).sum().item())

        # 3a. Collateral damage: previously-correct, non-targeted cells
        # that get disturbed by the correction process.
        collateral_candidates = clean_correct & (~target_mask)
        collateral_broken = collateral_candidates & (corrupted_final != solutions)
        total_collateral_candidates += int(collateral_candidates.sum().item())
        total_collateral_broken += int(collateral_broken.sum().item())

        # 2. Puzzle-level: did the FULL grid end up exactly matching the
        # true solution, or -- distinctly -- did it settle into SOME OTHER
        # valid, constraint-satisfying sudoku (kept at least one injected
        # wrong digit and made the rest of the grid coherent around it,
        # rather than reverting or producing an inconsistent mess)? Only
        # counted over puzzles that actually had >=1 corrupted cell.
        blank_all = (~given_mask) if given_mask is not None else torch.ones_like(solutions, dtype=torch.bool)
        puzzle_exact = ((corrupted_final == solutions) | ~blank_all).all(dim=1)
        puzzle_valid = _check_sudoku_constraints(corrupted_final)
        puzzle_valid_different = puzzle_valid & (~puzzle_exact)
        puzzle_has_target = target_mask.any(dim=1)

        total_puzzles += int(puzzle_has_target.sum().item())
        total_puzzle_exact += int((puzzle_exact & puzzle_has_target).sum().item())
        total_puzzle_valid_different += int((puzzle_valid_different & puzzle_has_target).sum().item())

        # 2b. Did the corrupted-then-denoised grid come back to EXACTLY what
        # the model itself would have produced with no injected mistake at
        # all (clean_final), rather than the dataset's one stored solution?
        # Sidesteps the multi-solution-ambiguity problem behind puzzle_exact
        # entirely, since it's model-vs-itself, not model-vs-external-answer
        # -- "did this mistake leave any lasting trace on the final output."
        puzzle_same_as_clean = (corrupted_final == clean_final).all(dim=1)
        total_puzzle_same_as_clean += int((puzzle_same_as_clean & puzzle_has_target).sum().item())

        # 3b/4. Recovery latency: first denoising step AFTER corrupt_idx at
        # which a corrupted cell is correct AND stays correct through the
        # end (last-flip-to-correct point, same suffix-reduction trick as
        # perturbation_recovery_probe.py's _commitment_steps).
        post_steps = corrupted_preds[corrupt_idx:]
        correct_seq = torch.stack([(p == solutions) for p in post_steps], dim=0)  # (T', B, 81)
        suffix_all = torch.flip(torch.cummin(torch.flip(correct_seq.int(), dims=[0]), dim=0).values, dims=[0]).bool()
        has_commit = suffix_all.any(dim=0)
        full_default = torch.full_like(suffix_all[0], correct_seq.shape[0] - 1, dtype=torch.int64)
        commit_step = torch.where(has_commit, suffix_all.int().argmax(dim=0), full_default)
        recovered_mask = target_mask & has_commit
        if recovered_mask.any():
            latencies.extend(commit_step[recovered_mask].tolist())

    return {
        "n_cells": n_cells,
        "corrupt_idx": corrupt_idx,
        "n_corrupted": total_corrupted,
        "recovery_rate": (total_recovered / total_corrupted) if total_corrupted else None,
        "adapt_rate": (total_adapted / total_corrupted) if total_corrupted else None,
        "collapse_rate": (total_collapsed / total_corrupted) if total_corrupted else None,
        "recovered_and_valid_rate": (total_recovered_and_valid / total_corrupted) if total_corrupted else None,
        "adapted_and_valid_rate": (total_adapted_and_valid / total_corrupted) if total_corrupted else None,
        "n_collateral_candidates": total_collateral_candidates,
        "collateral_break_rate": (total_collateral_broken / total_collateral_candidates) if total_collateral_candidates else None,
        "n_puzzles": total_puzzles,
        "puzzle_exact_rate": (total_puzzle_exact / total_puzzles) if total_puzzles else None,
        "puzzle_valid_different_rate": (total_puzzle_valid_different / total_puzzles) if total_puzzles else None,
        "puzzle_same_as_clean_rate": (total_puzzle_same_as_clean / total_puzzles) if total_puzzles else None,
        "recovery_latency_mean_steps": float(np.mean(latencies)) if latencies else None,
        "recovery_latency_median_steps": float(np.median(latencies)) if latencies else None,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/targeted_corruption_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 64)
    seed: int = pb.get("seed", 0)
    n_cells_values: list[int] = list(pb.get("n_cells_values", [1, 2, 4]))
    corrupt_fractions: list[float] = list(pb.get("corrupt_fractions", [0.1, 0.3, 0.5, 0.7, 0.9]))
    crop_bank_max_per_class: int = pb.get("crop_bank_max_per_class", 32)
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "targeted_corruption.json"))

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
    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  corrupt_idx={corrupt_idx}  n_cells_values={n_cells_values}")

    logger.info("Building digit crop bank from real dataset images...")
    crop_bank = _build_digit_crop_bank(eval_dl, cell_size, max_per_class=crop_bank_max_per_class)
    crop_bank = {d: v.to(device) for d, v in crop_bank.items()}
    logger.info(f"Crop bank sizes: {[(d, v.shape[0]) for d, v in crop_bank.items()]}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict = {}
    for n_cells in n_cells_values:
        for c_idx in corrupt_idx:
            key = f"n_cells={n_cells}/corrupt_idx={c_idx}"
            logger.info(f"Running {key} ...")
            r = _run_config(
                model, classifier, cell_size, cached_batches, crop_bank,
                num_inference_steps, cfg_scale, n_cells, c_idx, seed,
            )
            results[key] = r
            logger.info(f"  → {r}")

    if accelerator.is_main_process:
        print("\n" + "=" * 130)
        print(
            f"{'config':<28}{'n_corrupt':>10}{'recovery':>10}{'adapt':>9}{'collapse':>10}{'collat_brk':>11}"
            f"{'lat_mean':>10}{'lat_median':>11}"
        )
        print("=" * 130)
        for key, r in results.items():
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "n/a"
            print(
                f"{key:<28}{r['n_corrupted']:>10}{_fmt(r['recovery_rate']):>10}{_fmt(r['adapt_rate']):>9}"
                f"{_fmt(r['collapse_rate']):>10}{_fmt(r['collateral_break_rate']):>11}"
                f"{_fmt(r['recovery_latency_mean_steps']):>10}{_fmt(r['recovery_latency_median_steps']):>11}"
            )
        print("=" * 130)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
