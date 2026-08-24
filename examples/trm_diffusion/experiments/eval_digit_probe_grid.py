"""
experiments/eval_digit_probe_grid.py — evaluate a trained digit probe's
accuracy broken out by (denoising step, reasoning iteration) instead of
just the pooled/first/last-iteration summary experiments/train_digit_probe.py
reports.

Reuses the exact same labeling methodology as training (label = CNN
classification of the model's own final, clean image for that trajectory,
never the puzzle's ground truth -- see train_digit_probe.py's module
docstring for why) and the same forward_with_carry zH_out hook. The only
difference from training's own held-out eval is bookkeeping: z_H captures
are kept in their native (denoising_step, reasoning_iteration) grid
instead of being flattened into one pooled list, so a full accuracy
heatmap can be built after the fact. No extra model compute versus
training's own eval -- one forward_with_carry call per denoising step
already runs all n_sup reasoning iterations internally.

Usage:
    python experiments/eval_digit_probe_grid.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +checkpoint=runs/sudoku-painter-thinker.pt \\
      +probe_checkpoint=runs/digit_probe_sudoku_painter_thinker.pt \\
      +use_ema=true +grid_eval.num_batches=16

    # Options (all under +grid_eval.*):
    #   num_batches — held-out batches to average over (default 16)
    #   out         — json path (default: alongside probe_checkpoint,
    #                 named digit_probe_grid.json)
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _load_checkpoint
from factory import build_datasets, build_model
from models.digit_probe import DigitProbe
from perturbation_recovery_probe import _decode_cellwise

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _sample_trajectory_capture_zH_grid(
    model, conditions, x_init: torch.Tensor, num_inference_steps: int, cfg_scale: float, classifier, cell_size: int
) -> tuple[torch.Tensor, list]:
    """Same as train_digit_probe.py's _sample_trajectory_capture_zH, except
    z_H captures are returned as zH_grid[step_idx] = list of n_sup tensors
    (one per reasoning iteration of that denoising step) instead of one
    flat list across the whole trajectory."""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    zH_grid: list = []

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        step_zH: list = []
        pred_c, _, _ = model.forward_with_carry(step_sample, zH_out=step_zH)
        zH_grid.append(step_zH)
        noise_pred = pred_c.pred

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, _, _ = model.forward_with_carry(null_sample)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    final_image = model.decode_for_eval(x)
    final_digits = _decode_cellwise(final_image, classifier, cell_size)
    return final_digits, zH_grid


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    probe_checkpoint = cfg.get("probe_checkpoint", None)
    if checkpoint is None or probe_checkpoint is None:
        raise SystemExit(
            "ERROR: need both checkpoint= and +probe_checkpoint=.\n"
            "  Usage: python experiments/eval_digit_probe_grid.py experiment=<name> "
            "checkpoint=<main.pt> +probe_checkpoint=<digit_probe.pt> [+grid_eval.xxx=...]"
        )

    ge = cfg.get("grid_eval", {})
    num_batches: int = ge.get("num_batches", 16)
    out_path: str = ge.get("out", str(Path(probe_checkpoint).parent / "digit_probe_grid.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}  Probe: {probe_checkpoint}")

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    if not hasattr(model, "forward_with_carry"):
        raise SystemExit("This model has no forward_with_carry — needs a TRM-based model.")

    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    reasoner = getattr(model, "thinker", model)
    puzzle_emb_len = reasoner.inner.puzzle_emb_len

    probe_ckpt = torch.load(str(probe_checkpoint), map_location=device, weights_only=True)
    probe = DigitProbe(probe_ckpt["hidden_size"]).to(device)
    probe.load_state_dict(probe_ckpt["probe_state"])
    probe.eval()
    assert probe_ckpt["puzzle_emb_len"] == puzzle_emb_len, "probe/model puzzle_emb_len mismatch"

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    pipeline = model.sampling_pipeline
    cfg_scale: float = ge.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = ge.get("num_inference_steps", pipeline.num_inference_steps)
    n_sup = model.n_sup

    _, val_ds = build_datasets(cfg)
    collate_fn = getattr(type(val_ds), "collate_fn", None)
    val_dl = DataLoader(
        val_ds, batch_size=cfg.eval.get("batch_size", cfg.train.batch_size), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=True, collate_fn=collate_fn,
    )

    logger.info(
        f"Evaluating digit-probe accuracy grid: num_inference_steps={num_inference_steps}  n_sup={n_sup}  "
        f"num_batches={num_batches}"
    )

    correct_counts = np.zeros((num_inference_steps, n_sup), dtype=np.int64)
    total_counts = np.zeros((num_inference_steps, n_sup), dtype=np.int64)

    val_iter = iter(val_dl)
    for bi in range(num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            batch = next(val_iter)

        conditions = model._batch_to_sample(batch, device)
        bsz = batch["solution"].shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)

        final_digits, zH_grid = _sample_trajectory_capture_zH_grid(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size
        )
        given_mask = conditions.solution_mask
        blank = (~given_mask.to(device)) if given_mask is not None else torch.ones_like(final_digits, dtype=torch.bool)
        n_blank = int(blank.sum().item())

        for step_idx in range(num_inference_steps):
            for iter_idx in range(n_sup):
                z_H = zH_grid[step_idx][iter_idx]
                preds = probe(z_H[:, puzzle_emb_len:]).argmax(-1)
                correct = (preds == final_digits) & blank
                correct_counts[step_idx, iter_idx] += int(correct.sum().item())
                total_counts[step_idx, iter_idx] += n_blank

        if (bi + 1) % 4 == 0:
            logger.info(f"  {bi + 1}/{num_batches} batches done")

    acc_grid = correct_counts / np.maximum(total_counts, 1)
    pooled_acc = correct_counts.sum() / max(total_counts.sum(), 1)

    results = {
        "num_inference_steps": num_inference_steps,
        "n_sup": n_sup,
        "num_batches": num_batches,
        "pooled_acc": float(pooled_acc),
        "acc_grid": acc_grid.tolist(),  # [denoise_step][reasoning_iter]
    }

    if accelerator.is_main_process:
        logger.info(f"pooled_acc={pooled_acc:.4f}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
