"""
experiments/repair_transition_probe.py — Ablation D3: repair vs
regression transition counting from experiment D2's per-iteration decodes.

experiments/candidate_trajectory_probe.py decodes and classifies every
reasoning iteration's candidate prediction, but only reports aggregate
accuracy per (denoise bucket, k) — its own JSON output doesn't retain the
raw per-cell predictions (large per-sample tensors, not meant to be
serialized). This script re-runs the exact same instrumented trajectories
(reusing _run_and_score_trajectory directly, so the two experiments are
guaranteed consistent) to get those per-cell predictions, and for every
consecutive reasoning-iteration pair (k, k+1) at each instrumented
denoising step, counts, over BLANK cells only (given cells are trivially
"correct" throughout and would dilute the signal):

  repair     — cell was wrong at iteration k, right at k+1 (self-correction)
  regression — cell was right at iteration k, wrong at k+1 (breaking a
               previously-correct guess)

Reports, pooled across all cached samples (counts summed first, rates
computed from the pooled counts — not an average of per-batch rates):
  repair_rate     = repairs / (# blank cells wrong at k)
  regression_rate = regressions / (# blank cells right at k)
  net             = repairs - regressions (raw count and as a fraction of
                    all blank cells)

A nonzero repair rate is direct evidence the model revises committed
guesses mid-trajectory — a feed-forward (non-recursive) denoiser cannot do
this within a single reasoning pass, since it has no earlier guess of its
own to revise.

Usage: identical CLI to candidate_trajectory_probe.py (same probe.* keys).
    python experiments/repair_transition_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +probe.num_samples=256

    # TRM-alone works identically:
    python experiments/repair_transition_probe.py \\
      experiment=mnist_trm_diffusion_backbone \\
      checkpoint=runs/mnist_trm_diffusion_backbone/checkpoint_final.pt \\
      +probe.num_samples=256
"""

from __future__ import annotations

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

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from candidate_trajectory_probe import _bucket_labels, _run_and_score_trajectory
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


def _transition_counts(preds_k: torch.Tensor, preds_k1: torch.Tensor, solutions: torch.Tensor, given_masks) -> dict:
    """preds_k/preds_k1/solutions: (B, 81) — preds_k/preds_k1 already on
    CPU (from evaluate_grids' own .cpu() preds field); solutions moved to
    match. given_masks: (B, 81) bool or None (no given cells → all blank)."""
    sol = solutions.to(preds_k.device)
    blank = (~given_masks.to(preds_k.device)) if given_masks is not None else torch.ones_like(sol, dtype=torch.bool)

    right_k = (preds_k == sol) & blank
    right_k1 = (preds_k1 == sol) & blank
    wrong_k = (~right_k) & blank

    return {
        "repairs": int((wrong_k & right_k1).sum().item()),
        "regressions": int((right_k & ~right_k1).sum().item()),
        "n_wrong_k": int(wrong_k.sum().item()),
        "n_right_k": int(right_k.sum().item()),
        "n_blank": int(blank.sum().item()),
    }


def _accumulate(acc: dict, delta: dict) -> None:
    for key, val in delta.items():
        acc[key] = acc.get(key, 0) + val


def _rates_from_counts(c: dict) -> dict:
    repair_rate = c["repairs"] / c["n_wrong_k"] if c["n_wrong_k"] > 0 else float("nan")
    regression_rate = c["regressions"] / c["n_right_k"] if c["n_right_k"] > 0 else float("nan")
    net = c["repairs"] - c["regressions"]
    net_rate = net / c["n_blank"] if c["n_blank"] > 0 else float("nan")
    return {
        "repair_rate": repair_rate,
        "regression_rate": regression_rate,
        "net": net,
        "net_rate": net_rate,
        **c,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/repair_transition_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    reset_every = pb.get("reset_every", 1)
    denoise_fractions: list[float] = list(pb.get("denoise_fractions", [0.1, 0.5, 0.9]))
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "repair_transitions.json"))

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

    labels = _bucket_labels(denoise_fractions)
    idx_to_label = {round(f * (num_inference_steps - 1)): label for f, label in zip(denoise_fractions, labels)}
    instrumented_idx = sorted(idx_to_label.keys())

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={model.n_sup}  "
        f"instrumented denoise steps={instrumented_idx} ({[idx_to_label[i] for i in instrumented_idx]})  "
        f"reset_every={reset_every}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)
    reset_fn = _make_reset_fn(reset_every)

    # pooled_counts[step_idx][k] = running transition counts (k → k+1), summed
    # across every cached batch — see _transition_counts/_accumulate.
    pooled_counts: dict = {i: {} for i in instrumented_idx}

    for cb in cached_batches:
        _, curves = _run_and_score_trajectory(
            model, classifier, cell_size, cb["conditions"], cb["solutions"], cb["given_masks"], cb["x_init"],
            num_inference_steps, cfg_scale, reset_fn, set(instrumented_idx),
        )
        for step_idx in instrumented_idx:
            per_k = curves[step_idx]
            for k in range(len(per_k) - 1):
                delta = _transition_counts(
                    per_k[k]["preds"], per_k[k + 1]["preds"], cb["solutions"], cb["given_masks"]
                )
                pooled_counts[step_idx].setdefault(k, {})
                _accumulate(pooled_counts[step_idx][k], delta)

    results: dict = {}
    for step_idx in instrumented_idx:
        label = idx_to_label[step_idx]
        results[label] = {
            f"{k+1}->{k+2}": _rates_from_counts(counts) for k, counts in sorted(pooled_counts[step_idx].items())
        }

    if accelerator.is_main_process:
        for label, transitions in results.items():
            print("\n" + "=" * 90)
            print(f"bucket={label}")
            print("=" * 90)
            print(f"{'k->k+1':>10}{'repair_rate':>14}{'regression_rate':>17}{'net':>8}{'net_rate':>10}")
            for trans_key, r in transitions.items():
                print(
                    f"{trans_key:>10}{r['repair_rate']:>14.4f}{r['regression_rate']:>17.4f}"
                    f"{r['net']:>8d}{r['net_rate']:>10.4f}"
                )
        print("=" * 90)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
