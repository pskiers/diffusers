"""
experiments/violation_trajectory_solver_probe.py — violation-count-per-
denoising-step trajectory for a standalone CONDITIONAL painter (e.g.
ConcatConditionedUNetPainter trained via train_trm.py
experiment=mnist_unet_concat_painter_srm[_1000t]): unlike
violation_trajectory_probe.py's DM baseline (steering=None, condition_keys
empty, genuinely zero information about the puzzle), this painter is
trained to channel-concat the given-cells puzzle image as its own real
condition and is meant to be evaluated as an actual solver -- so there's
only one real curve here (with CFG against the true condition), not a
DM-vs-PT pair.

Same decode-before-scheduler.step methodology as violation_trajectory_probe.py
(decode the model's direct clean-image prediction, not the noisy blended
scheduler.step output -- see that script's docstring for why).

Usage:
    python experiments/violation_trajectory_solver_probe.py \\
      experiment=mnist_unet_concat_painter_srm \\
      +checkpoint=runs/sudoku_unet_concat_t100-30k.pt \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +probe.num_samples=512
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
from factory import build_datasets, build_model
from hydra.utils import instantiate
from perturbation_recovery_probe import _decode_cellwise
from violation_trajectory_probe import _count_violating_units

logger = get_logger(__name__, log_level="INFO")


def _get_painter(model):
    return getattr(model, "painter", model)


@torch.no_grad()
def _run_solver_trajectory(model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size) -> list:
    """From pure noise, the painter's real conditional prediction (its own
    channel-concat condition, e.g. the puzzle's given cells) at every step,
    decoding the clean prediction before scheduler.step reintroduces noise."""
    painter = _get_painter(model)
    use_cfg = cfg_scale != 1.0 and len(painter.condition_keys) > 0
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    per_step_preds = []
    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        noise_pred = painter(step_sample, steering=None).pred
        if use_cfg:
            null_sample = painter.null_condition_sample(step_sample)
            noise_pred_u = painter(null_sample, steering=None).pred
            noise_pred = noise_pred_u + cfg_scale * (noise_pred - noise_pred_u)
        per_step_preds.append(_decode_cellwise(noise_pred.float().clamp(0.0, 1.0), classifier, cell_size))
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return per_step_preds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 256)
    seed: int = pb.get("seed", 0)
    out_path: str = pb.get("out", "runs/violation_trajectory_solver.json")

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
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  condition_keys={_get_painter(model).condition_keys}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in model.scheduler.timesteps]

    violations = [[] for _ in range(num_inference_steps)]
    n_done = 0
    for bi, cb in enumerate(cached_batches):
        conditions = cb["conditions"]
        torch.manual_seed(seed + bi)
        x_init = torch.randn_like(cb["x_init"])

        preds = _run_solver_trajectory(model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size)
        for step_idx in range(num_inference_steps):
            violations[step_idx].append(_count_violating_units(preds[step_idx]))
        n_done += x_init.shape[0]
        logger.info(f"  batch {bi+1}/{len(cached_batches)} done ({n_done} samples)")

    results = {"timesteps": timesteps, "mean_violations": []}
    for step_idx in range(num_inference_steps):
        cat = torch.cat(violations[step_idx])
        results["mean_violations"].append(float(cat.float().mean().item()))

    if accelerator.is_main_process:
        print("\n" + "=" * 40)
        print(f"{'timestep':>10}{'violations':>14}")
        print("=" * 40)
        for t, v in zip(timesteps, results["mean_violations"]):
            print(f"{t:>10}{v:>14.3f}")
        print("=" * 40)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": n_done, **results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
