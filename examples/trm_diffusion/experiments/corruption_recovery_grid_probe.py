"""
experiments/corruption_recovery_grid_probe.py — the OTHER half of
painter_only_denoise_probe.py's experiment 2, generalized: start from the
real solved-grid image noised to t_start (NOT pure noise — see
violation_sensitivity_probe.py's clean/n_cells=0 baseline for that
different, much harder from-scratch task), inject n_cells wrong digits
under a violating/non_violating condition (same digit-selection rule as
violation_sensitivity_probe.py._sample_condition_wrong_crops), then denoise
the remaining schedule steps and report the same
recovered(+valid)/adapted(kept)(+valid)/collapsed and
puzzle_valid_different/puzzle_same_as_clean metrics.

Unlike violation_sensitivity_probe.py (which recomputes an unconditioned
full trajectory from pure noise on every single job, most of it thrown
away), this script computes the shared per-(t_start, batch) pieces --
noised x_t and the CLEAN (uncorrupted) denoise -- exactly ONCE per t_start
and reuses them across every (condition, n_cells) combination at that
t_start, since neither depends on the corruption. The clean denoise result
also directly gives the n_cells=0 column (clean_puzzle_valid_rate /
clean_puzzle_exact_rate) -- and this time, since the starting point IS the
real solution just noised to t_start (not pure noise), that column should
land near painter_only_denoise_probe.py's original clean puzzle_acc numbers,
not violation_sensitivity_probe.py's near-zero from-scratch numbers.

Usage:
    python experiments/corruption_recovery_grid_probe.py \\
      experiment=mnist_unet_concat_painter_srm_1000t \\
      +checkpoint=runs/sudoku_unet_concat_t1000-30k.pt \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +probe.num_samples=128 +probe.t_starts=[100,300,500,700,900] \\
      +probe.n_cells_values=[0,1,2,4,8,16,32,64] \\
      +probe.conditions=[violating,non_violating]

    # Options (all under +probe.*): same as painter_only_denoise_probe.py,
    # plus conditions (default [violating, non_violating]).
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
from painter_only_denoise_probe import _denoise_from, _get_painter
from targeted_corruption_probe import _build_digit_crop_bank, _corrupt_with_wrong_digits, _select_target_cells
from violation_sensitivity_probe import _sample_condition_wrong_crops

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _run_t_start(
    model, classifier, cell_size: int, cached_batches: list, crop_bank: dict,
    run_timesteps: torch.Tensor, t_start: int, n_cells_values: list[int], conditions: list[str],
    base_seed: int, cfg_scale: float,
) -> dict:
    from eval.mnist_eval import evaluate_grids

    total_clean = total_clean_valid = total_clean_exact = 0
    # per (condition, n_cells): accumulators
    acc = {
        (cond, n): dict(
            n_corrupted=0, recovered=0, adapted=0, collapsed=0,
            recovered_and_valid=0, adapted_and_valid=0,
            n_puzzles=0, puzzle_valid_different=0, puzzle_same_as_clean=0,
        )
        for cond in conditions for n in n_cells_values if n > 0
    }

    for bi, cb in enumerate(cached_batches):
        device = cb["conditions"].images.device
        images = cb["conditions"].images
        conditions_sample = cb["conditions"]
        solutions = cb["solutions"].to(device)
        given_masks = cb["given_masks"].to(device) if cb["given_masks"] is not None else None
        B = images.shape[0]
        seed = base_seed + bi

        torch.manual_seed(seed)
        noise = torch.randn_like(images)
        t_start_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t_base = model.scheduler.add_noise(images, noise, t_start_batch)

        clean_final = _denoise_from(model, conditions_sample, x_t_base, run_timesteps, cfg_scale=cfg_scale)
        clean_final = clean_final.clamp(0.0, 1.0)
        clean_result = evaluate_grids(clean_final, solutions, classifier, cell_size, given_masks=given_masks)
        clean_preds = clean_result["preds"].to(device)
        clean_correct = clean_preds == solutions

        total_clean += B
        total_clean_valid += int(_check_sudoku_constraints(clean_preds).sum().item())
        total_clean_exact += int(clean_correct.all(dim=1).sum().item())

        blank_ok = torch.ones_like(given_masks) if given_masks is not None else torch.ones(B, 81, dtype=torch.bool, device=device)

        for n_cells in n_cells_values:
            if n_cells == 0:
                continue
            for condition in conditions:
                target_mask = _select_target_cells(blank_ok, given_masks, n_cells)
                if not target_mask.any():
                    continue
                crops, wrong_digits, success = _sample_condition_wrong_crops(
                    crop_bank, solutions, given_masks, target_mask, device, condition
                )
                target_mask = target_mask & success
                if not target_mask.any():
                    continue

                x_t = _corrupt_with_wrong_digits(x_t_base, t_start_batch, target_mask, crops, cell_size, model.scheduler)
                corrupted_final = _denoise_from(model, conditions_sample, x_t, run_timesteps, cfg_scale=cfg_scale)
                corrupted_final = corrupted_final.clamp(0.0, 1.0)
                result = evaluate_grids(corrupted_final, solutions, classifier, cell_size, given_masks=given_masks)
                preds = result["preds"].to(device)

                recovered = (preds == solutions) & target_mask
                adapted = (~recovered) & (preds == wrong_digits) & target_mask
                collapsed = target_mask & (~recovered) & (~adapted)
                valid_grid = _check_sudoku_constraints(preds)
                valid_broadcast = valid_grid.unsqueeze(1).expand_as(target_mask)

                blank_all = (~given_masks) if given_masks is not None else torch.ones_like(solutions, dtype=torch.bool)
                puzzle_exact = ((preds == solutions) | ~blank_all).all(dim=1)
                puzzle_valid_different = valid_grid & (~puzzle_exact)
                puzzle_same_as_clean = (preds == clean_preds).all(dim=1)
                puzzle_has_target = target_mask.any(dim=1)

                a = acc[(condition, n_cells)]
                a["n_corrupted"] += int(target_mask.sum().item())
                a["recovered"] += int(recovered.sum().item())
                a["adapted"] += int(adapted.sum().item())
                a["collapsed"] += int(collapsed.sum().item())
                a["recovered_and_valid"] += int((recovered & valid_broadcast).sum().item())
                a["adapted_and_valid"] += int((adapted & valid_broadcast).sum().item())
                a["n_puzzles"] += int(puzzle_has_target.sum().item())
                a["puzzle_valid_different"] += int((puzzle_valid_different & puzzle_has_target).sum().item())
                a["puzzle_same_as_clean"] += int((puzzle_same_as_clean & puzzle_has_target).sum().item())

    results = {
        "clean/n_cells=0": {
            "n_clean": total_clean,
            "clean_puzzle_valid_rate": (total_clean_valid / total_clean) if total_clean else None,
            "clean_puzzle_exact_rate": (total_clean_exact / total_clean) if total_clean else None,
        }
    }
    for (condition, n_cells), a in acc.items():
        nc = a["n_corrupted"]
        npz = a["n_puzzles"]
        results[f"{condition}/n_cells={n_cells}"] = {
            "n_corrupted": nc,
            "recovery_rate": (a["recovered"] / nc) if nc else None,
            "adapt_rate": (a["adapted"] / nc) if nc else None,
            "collapse_rate": (a["collapsed"] / nc) if nc else None,
            "recovered_and_valid_rate": (a["recovered_and_valid"] / nc) if nc else None,
            "adapted_and_valid_rate": (a["adapted_and_valid"] / nc) if nc else None,
            "n_puzzles": npz,
            "puzzle_valid_different_rate": (a["puzzle_valid_different"] / npz) if npz else None,
            "puzzle_same_as_clean_rate": (a["puzzle_same_as_clean"] / npz) if npz else None,
        }
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 128)
    seed: int = pb.get("seed", 0)
    t_starts: list[int] = list(pb.get("t_starts", [50, 300, 500, 700, 950]))
    n_cells_values: list[int] = list(pb.get("n_cells_values", [0, 1, 2, 4, 8, 16, 32, 64]))
    conditions: list[str] = list(pb.get("conditions", ["violating", "non_violating"]))
    crop_bank_max_per_class: int = pb.get("crop_bank_max_per_class", 32)
    out_path: str = pb.get("out", "runs/corruption_recovery_grid.json")

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
            f"t_starts {bad} are not members of the {num_inference_steps}-step schedule {sorted(valid_t, reverse=True)}."
        )
    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  condition_keys={_get_painter(model).condition_keys}  "
        f"t_starts={t_starts}  n_cells_values={n_cells_values}  conditions={conditions}"
    )

    logger.info("Building digit crop bank from real dataset images...")
    crop_bank = _build_digit_crop_bank(eval_dl, cell_size, max_per_class=crop_bank_max_per_class)
    crop_bank = {d: v.to(device) for d, v in crop_bank.items()}

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict = {}
    for t_start in t_starts:
        run_timesteps = full_timesteps[full_timesteps <= t_start]
        logger.info(f"Running t_start={t_start} ({len(run_timesteps)} steps) ...")
        r = _run_t_start(
            model, classifier, cell_size, cached_batches, crop_bank,
            run_timesteps, t_start, n_cells_values, conditions, seed, cfg_scale,
        )
        for key, v in r.items():
            results[f"t_start={t_start}/{key}"] = v
            logger.info(f"  {key} → {v}")

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
