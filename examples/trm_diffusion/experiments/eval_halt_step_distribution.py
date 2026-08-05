"""
experiments/eval_halt_step_distribution.py — Reproduce the classic
adaptive-halting story ("most examples need very few reasoning steps, but a
minority genuinely need the full budget") for this codebase's continuous-
loss halt head.

experiments/eval_halt_step_profile.py reports the *average* number of
reasoning steps used per denoising step — exactly what the actually
deployed halting rule produces, since ThinkerFrozenPainterBase.
forward_with_carry's use_halt_head only ever checks the halt head's
batch-mean prediction (a single decision for the whole batch). That can't
show per-sample spread: once the batch-mean rule stops a batch, no later
per-sample predictions exist to look at.

This script instead runs the FULL, un-truncated n_sup reasoning budget at
every denoising step (a plain static/carry reference trajectory — the
generated images are unaffected by any threshold analyzed here) while
recording every individual sample's own predict_halt_value() at every
reasoning sub-step (via forward_with_carry's halt_preds_out hook). For one
or more halt_threshold values, it then asks, per sample and per denoising
step: "at which reasoning step would THIS sample's own prediction have
first crossed the threshold?" — and reports the distribution of that
(mean, std, min, max, percentiles, fraction pinned at the full budget),
plus a histogram pooled across the whole trajectory, instead of just the
one mean the real batch-mean rule produces.

Usage:
    python experiments/eval_halt_step_distribution.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true \\
      +checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_with_halt_head.pt \\
      +profile.reset_every=5 +profile.thresholds=[0.0,0.0002]

    # Options (all under +profile.*):
    #   reset_every  — single carry-reset setting for the reference
    #                  trajectory (default 5). Only affects the trajectory
    #                  itself, not the post-hoc threshold analysis.
    #   thresholds   — halt_threshold values to analyze post-hoc (default [0.0])
    #   num_samples  — default 256
    #   seed         — default 0
    #   cfg_scale / num_inference_steps — default from model.sampling_pipeline
    #   out          — json path (default: alongside checkpoint, named
    #                  halt_step_distribution.json)

Config overrides work exactly like train_trm.py / eval.py.
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

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from factory import build_datasets, build_model

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _run_reference_sampling(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    reset_fn,
    preds_by_denoise_idx: list[list[torch.Tensor]],
) -> None:
    """Runs the FULL (un-truncated) n_sup reasoning budget at every
    denoising step — a plain static/carry reference trajectory, unaffected
    by any halt_threshold — while recording every individual sample's own
    predict_halt_value() at every reasoning sub-step into
    preds_by_denoise_idx[step_idx] (one (n_sup, B) tensor appended per
    cached batch; conditional and unconditional CFG branches are both
    appended here, pooled together as independent reasoning trajectories)."""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        this_cond_preds: list[torch.Tensor] = []
        pred_c, z_H_c, z_L_c = model.forward_with_carry(
            step_sample, z_H_c, z_L_c, use_halt_head=False, halt_preds_out=this_cond_preds,
        )
        noise_pred = pred_c.pred
        preds_by_denoise_idx[step_idx].append(torch.stack(this_cond_preds, dim=0))  # (n_sup, B)

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            this_uncond_preds: list[torch.Tensor] = []
            pred_u, z_H_u, z_L_u = model.forward_with_carry(
                null_sample, z_H_u, z_L_u, use_halt_head=False, halt_preds_out=this_uncond_preds,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)
            preds_by_denoise_idx[step_idx].append(torch.stack(this_uncond_preds, dim=0))

        x = model.scheduler.step(noise_pred, t, x).prev_sample


def _first_halt_steps(preds: torch.Tensor, threshold: float) -> torch.Tensor:
    """preds: (n_sup, N). Returns (N,) int64 — the 0-based reasoning step at
    which each sample's own prediction first drops to/below threshold, or
    n_sup - 1 (full budget) if it never does. Mirrors
    ThinkerFrozenPainterBase.forward_with_carry's use_halt_head rule, but
    applied per individual sample instead of to the batch mean."""
    n_sup, n = preds.shape
    device = preds.device
    halts = preds <= threshold
    has_halt = halts.any(dim=0)
    full_traj = torch.full((n,), n_sup - 1, device=device, dtype=torch.int64)
    return torch.where(has_halt, halts.float().argmax(dim=0), full_traj)


def _distribution_stats(steps_used: torch.Tensor, n_sup: int) -> dict:
    """steps_used: 1-D tensor of steps-used values in [1, n_sup]."""
    steps_f = steps_used.float()
    return {
        "n": int(steps_used.shape[0]),
        "mean": float(steps_f.mean().item()),
        "std": float(steps_f.std().item()),
        "min": int(steps_used.min().item()),
        "max": int(steps_used.max().item()),
        "p10": float(torch.quantile(steps_f, 0.10).item()),
        "p50": float(torch.quantile(steps_f, 0.50).item()),
        "p90": float(torch.quantile(steps_f, 0.90).item()),
        "frac_full_budget": float((steps_used == n_sup).float().mean().item()),
    }


def _histogram(steps_used: torch.Tensor, n_sup: int) -> dict:
    """Fraction of observations using exactly k steps, for k = 1..n_sup."""
    counts = torch.bincount(steps_used, minlength=n_sup + 1)[1:]  # drop the k=0 bucket (unreachable)
    total = counts.sum().item()
    return {str(k + 1): float(counts[k].item() / total) for k in range(n_sup)}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/eval_halt_step_distribution.py experiment=<name> "
            "checkpoint=<path/to/checkpoint_with_halt_head.pt> thinker.with_halt_head=true "
            "[+profile.xxx=...]"
        )

    pf = cfg.get("profile", {})
    reset_every = pf.get("reset_every", 5)
    thresholds: list[float] = list(pf.get("thresholds", [0.0]))
    num_samples: int = pf.get("num_samples", 256)
    seed: int = pf.get("seed", 0)
    out_path: str = pf.get("out", str(Path(checkpoint).parent / "halt_step_distribution.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    if not getattr(model.thinker, "with_halt_head", False):
        raise SystemExit(
            "Model was built without a halt head — add thinker.with_halt_head=true to the command line."
        )

    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

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

    pipeline = model.sampling_pipeline
    cfg_scale: float = pf.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pf.get("num_inference_steps", pipeline.num_inference_steps)
    n_sup = model.n_sup

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  "
        f"trained n_sup={n_sup}  reset_every={reset_every}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in model.scheduler.timesteps]

    preds_by_denoise_idx: list[list[torch.Tensor]] = [[] for _ in range(num_inference_steps)]
    reset_fn = _make_reset_fn(reset_every)
    for cb in cached_batches:
        _run_reference_sampling(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale, reset_fn, preds_by_denoise_idx,
        )
    # (n_sup, N_total) per denoising step — N_total pools every cached batch's
    # samples together with both CFG branches (each counted as one observation).
    preds_by_denoise_idx = [torch.cat(step_preds, dim=1) for step_preds in preds_by_denoise_idx]

    results: dict[str, dict] = {}
    for threshold in thresholds:
        per_step_stats = []
        all_steps_used = []
        for step_idx in range(num_inference_steps):
            first_halt = _first_halt_steps(preds_by_denoise_idx[step_idx], threshold)
            steps_used = first_halt + 1
            per_step_stats.append(_distribution_stats(steps_used, n_sup))
            all_steps_used.append(steps_used)

        pooled = torch.cat(all_steps_used, dim=0)
        results[str(threshold)] = {
            "per_denoise_step": per_step_stats,
            "pooled": _distribution_stats(pooled, n_sup),
            "pooled_histogram": _histogram(pooled, n_sup),
        }
        logger.info(f"threshold={threshold}  pooled={results[str(threshold)]['pooled']}")

    if accelerator.is_main_process:
        for threshold in thresholds:
            r = results[str(threshold)]
            print("\n" + "=" * 100)
            print(f"threshold={threshold}  (pooled across all denoising steps and samples)")
            print("=" * 100)
            p = r["pooled"]
            print(
                f"  n={p['n']}  mean={p['mean']:.2f}  std={p['std']:.2f}  min={p['min']}  max={p['max']}  "
                f"p10={p['p10']:.1f}  p50={p['p50']:.1f}  p90={p['p90']:.1f}  "
                f"frac_full_budget={p['frac_full_budget']:.3f}"
            )
            print("\n  steps_used  fraction")
            for k, frac in r["pooled_histogram"].items():
                bar = "#" * int(round(frac * 60))
                print(f"  {k:>10}  {frac:>7.3f}  {bar}")

            print("\n" + "-" * 100)
            print(f"{'denoise_idx':>12}{'timestep':>10}{'mean':>8}{'std':>8}{'min':>6}{'max':>6}"
                  f"{'p10':>7}{'p50':>7}{'p90':>7}{'frac_full':>11}")
            print("-" * 100)
            for i, s in enumerate(r["per_denoise_step"]):
                print(
                    f"{i:>12}{timesteps[i]:>10}{s['mean']:>8.2f}{s['std']:>8.2f}{s['min']:>6}{s['max']:>6}"
                    f"{s['p10']:>7.1f}{s['p50']:>7.1f}{s['p90']:>7.1f}{s['frac_full_budget']:>11.3f}"
                )
            print("-" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"checkpoint": str(checkpoint), "timesteps": timesteps, "n_sup": n_sup, "results": results},
                f,
                indent=2,
            )
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()