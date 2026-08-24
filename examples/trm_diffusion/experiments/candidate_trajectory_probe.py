"""
experiments/candidate_trajectory_probe.py — Ablation D2: candidate
trajectory scoring across the TRM's n_sup reasoning iterations.

Decodes the model's intermediate prediction y_k at EVERY reasoning
iteration k = 1..n_sup within a denoising step (via forward_with_carry's
logits_out hook, folded to an image via run_painter + the same CFG combine
the real trajectory uses) and scores it against ground truth with the same
classifier-based accuracy used everywhere else in this codebase. This is
the existing convergence-plot idea (prediction distance vs. k) but with
task accuracy on the y-axis instead — directly testable evidence for
whether the model's answer keeps improving across recursion iterations,
not just across denoising steps.

Full instrumentation (decode + classify at every k) at every one of
num_inference_steps denoising steps would be expensive for no real benefit
— D2 only needs it broken out by early/mid/late position in the
trajectory. So only the denoising steps closest to probe.denoise_fractions
(default [0.1, 0.5, 0.9], labeled early/mid/late when there are exactly 3
in ascending order) are instrumented; every other denoising step runs the
normal fast (single decode) path. The generated FINAL image is completely
unaffected by which steps are instrumented — recording logits_out doesn't
change what forward_with_carry computes or returns.

Two outputs are reported:
  1. curves: per bucket (early/mid/late), per reasoning iteration k, the
     mean (cell_acc, puzzle_acc, constraint_puzzle_acc,
     given_consistent_puzzle_acc) of the candidate y_k across all cached
     samples — the primary deliverable, directly plottable as accuracy vs.
     recursion iteration k, one curve per denoising-trajectory position.
  2. pass_at_n: puzzle-level pass@1 / pass@N on the FINAL generated image
     only (not the intermediate y_k's — those are diagnostic mid-
     trajectory decodes, not submitted answers), computed from
     probe.num_repeats independent full trajectories (different initial
     noise, same puzzle/condition) per cached sample. pass@1 is just the
     mean per-attempt puzzle accuracy; pass@N is the fraction of puzzles
     solved by AT LEAST ONE of the N attempts.

Usage:
    python experiments/candidate_trajectory_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +probe.num_samples=256

    # TRM-alone (TRMDiffusionBackbone) works identically:
    python experiments/candidate_trajectory_probe.py \\
      experiment=mnist_trm_diffusion_backbone \\
      checkpoint=runs/mnist_trm_diffusion_backbone/checkpoint_final.pt \\
      +probe.num_samples=256

    # Custom trajectory positions / repeat count:
    python experiments/candidate_trajectory_probe.py ... \\
      +probe.denoise_fractions=[0.0,0.25,0.5,0.75,1.0] +probe.num_repeats=10

Config overrides work exactly like train_trm.py / eval.py.
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

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from eval.mnist_eval import evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


def _bucket_labels(fractions: list[float]) -> list[str]:
    """early/mid/late for exactly 3 ascending fractions; otherwise labeled
    by the fraction value itself (still ordered, just not assumed to mean
    "thirds")."""
    if len(fractions) == 3 and fractions == sorted(fractions):
        return ["early", "mid", "late"]
    return [f"frac={f}" for f in fractions]


@torch.no_grad()
def _per_sample_puzzle_correct(
    images: torch.Tensor, solutions: torch.Tensor, classifier, cell_size: int
) -> torch.Tensor:
    """(B,) bool — whether every cell of the decoded image matches the
    ground-truth solution. Duplicates evaluate_grids' per-cell
    classification (rather than changing evaluate_grids' return contract,
    which other scripts already depend on) since pass@N needs the
    per-sample flags, not just the batch-mean puzzle_acc it returns."""
    device = next(classifier.parameters()).device
    images = images.to(device)
    solutions = solutions.to(device)
    B = images.shape[0]
    cells = images.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)
    preds = classifier(cells).argmax(dim=1).reshape(B, 81)
    return (preds == solutions.reshape(B, 81)).all(dim=1)


@torch.no_grad()
def _run_and_score_trajectory(
    model,
    classifier,
    cell_size: int,
    conditions,
    solutions: torch.Tensor,
    given_masks,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    reset_fn,
    instrumented_idx: set,
) -> tuple[torch.Tensor, dict]:
    """Runs one real generation trajectory — x always advances using the
    FINAL reasoning iteration's CFG-combined prediction, exactly like real
    sampling, so instrumentation never changes the generated image. At
    every denoising step in instrumented_idx, additionally decodes +
    classifies every reasoning iteration's own candidate.

    Returns (final_image, curves) where curves[step_idx] is a list (length
    n_sup) of evaluate_grids() dicts, one per reasoning iteration.
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None
    curves: dict = {}

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        instrument = step_idx in instrumented_idx
        cond_logits_k = [] if instrument else None
        pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c, logits_out=cond_logits_k)
        noise_pred = pred_c.pred

        null_sample = None
        uncond_logits_k = None
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            uncond_logits_k = [] if instrument else None
            pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u, logits_out=uncond_logits_k)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        if instrument:
            per_k = []
            for k in range(len(cond_logits_k)):
                cand = model.run_painter(step_sample, cond_logits_k[k])
                if cfg_scale != 1.0:
                    cand_u = model.run_painter(null_sample, uncond_logits_k[k])
                    cand = cand_u + cfg_scale * (cand - cand_u)
                cand = model.decode_for_eval(cand)
                per_k.append(evaluate_grids(cand, solutions, classifier, cell_size, given_masks=given_masks))
            curves[step_idx] = per_k

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return x, curves


def _average_curves(all_curves: list[dict], sample_counts: list[int], instrumented_idx: list) -> dict:
    """all_curves[i][step_idx] is a length-n_sup list of evaluate_grids()
    dicts for cached batch i; averages across batches (sample-count
    weighted) into curves[step_idx] = length-n_sup list of averaged dicts."""
    out: dict = {}
    for step_idx in instrumented_idx:
        n_sup = len(all_curves[0][step_idx])
        per_k: list[dict] = []
        for k in range(n_sup):
            metrics: dict[str, list] = {}
            for curves, n in zip(all_curves, sample_counts):
                for key, val in curves[step_idx][k].items():
                    if key == "preds" or val is None:
                        continue
                    metrics.setdefault(key, []).append((val, n))
            per_k.append({
                key: float(np.average([v for v, _ in pairs], weights=[n for _, n in pairs]))
                for key, pairs in metrics.items()
            })
        out[step_idx] = per_k
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/candidate_trajectory_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    reset_every = pb.get("reset_every", 1)
    denoise_fractions: list[float] = list(pb.get("denoise_fractions", [0.1, 0.5, 0.9]))
    num_repeats: int = pb.get("num_repeats", 5)
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "candidate_trajectory.json"))

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

    instrumented_idx = sorted({round(f * (num_inference_steps - 1)) for f in denoise_fractions})
    labels = _bucket_labels(denoise_fractions)
    # If rounding collapsed distinct fractions onto the same step index,
    # de-dup while keeping the label list aligned to the surviving indices.
    idx_to_label = {}
    for f, label in zip(denoise_fractions, labels):
        idx_to_label[round(f * (num_inference_steps - 1))] = label
    instrumented_idx = sorted(idx_to_label.keys())

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={model.n_sup}  "
        f"instrumented denoise steps={instrumented_idx} ({[idx_to_label[i] for i in instrumented_idx]})  "
        f"reset_every={reset_every}  num_repeats={num_repeats}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)
    reset_fn = _make_reset_fn(reset_every)

    # ── Pass 1: curves (one trajectory per cached batch, no repeats needed —
    #    the curve is about within-trajectory refinement, not sampling variance) ──
    all_curves = []
    sample_counts = []
    all_final_correct = []  # for this same single trajectory's pass@1 contribution
    for cb in cached_batches:
        x, curves = _run_and_score_trajectory(
            model, classifier, cell_size, cb["conditions"], cb["solutions"], cb["given_masks"], cb["x_init"],
            num_inference_steps, cfg_scale, reset_fn, set(instrumented_idx),
        )
        all_curves.append(curves)
        sample_counts.append(cb["solutions"].shape[0])
        generated = model.decode_for_eval(x)
        all_final_correct.append(
            _per_sample_puzzle_correct(generated, cb["solutions"], classifier, cell_size)
        )
    curves = _average_curves(all_curves, sample_counts, instrumented_idx)

    # ── Pass 2: pass@N on the FINAL image, num_repeats independent noise
    #    draws per cached sample (repeat 1 reuses pass 1's trajectory) ──
    per_sample_correct = [[c] for c in all_final_correct]  # list over batches of list over repeats
    t0 = time.time()
    for r in range(1, num_repeats):
        torch.manual_seed(seed * 1000 + r)
        for bi, cb in enumerate(cached_batches):
            x_init_r = torch.randn_like(cb["x_init"])
            x, _ = _run_and_score_trajectory(
                model, classifier, cell_size, cb["conditions"], cb["solutions"], cb["given_masks"], x_init_r,
                num_inference_steps, cfg_scale, reset_fn, set(),
            )
            generated = model.decode_for_eval(x)
            per_sample_correct[bi].append(
                _per_sample_puzzle_correct(generated, cb["solutions"], classifier, cell_size)
            )
    logger.info(f"pass@N repeats done in {time.time() - t0:.1f}s")

    pass_at_1_list, pass_at_n_list = [], []
    for bi, correct_per_repeat in enumerate(per_sample_correct):
        stacked = torch.stack(correct_per_repeat, dim=0)  # (num_repeats, B)
        pass_at_1_list.append(stacked[0].float().mean().item())
        pass_at_n_list.append(stacked.any(dim=0).float().mean().item())
    pass_at_1 = float(np.average(pass_at_1_list, weights=sample_counts))
    pass_at_n = float(np.average(pass_at_n_list, weights=sample_counts))

    results = {
        "curves": {idx_to_label[i]: curves[i] for i in instrumented_idx},
        "pass_at_1": pass_at_1,
        f"pass_at_{num_repeats}": pass_at_n,
    }

    if accelerator.is_main_process:
        for i in instrumented_idx:
            label = idx_to_label[i]
            print("\n" + "=" * 90)
            print(f"denoise_step={i}  bucket={label}")
            print("=" * 90)
            print(f"{'k':>4}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}{'given_cons':>12}")
            for k, m in enumerate(curves[i]):
                print(
                    f"{k+1:>4}{m['cell_acc']:>10.4f}{m['puzzle_acc']:>12.4f}"
                    f"{m['constraint_puzzle_acc']:>12.4f}{m.get('given_consistent_puzzle_acc', float('nan')):>12.4f}"
                )
        print("\n" + "=" * 90)
        print(f"pass@1  = {pass_at_1:.4f}")
        print(f"pass@{num_repeats} = {pass_at_n:.4f}  ({num_repeats} independent noise draws)")
        print("=" * 90)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
