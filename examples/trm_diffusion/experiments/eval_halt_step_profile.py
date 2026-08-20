"""
experiments/eval_halt_step_profile.py — Show how the halt head allocates its
reasoning-step budget across the denoising trajectory, for a handful of
already-identified (reset_every, halt_threshold) configs.

experiments/ablate_trm_loop_budget.py's "halt" axis only reports one number
per config (total_sup_calls, summed over the whole trajectory) — useful for
picking a good operating point, but it hides *where* in the trajectory those
calls are spent, and collapses every sample in a batch into one number. This
script instead runs the real, per-sample dynamic-re-batching halting rule
(see ThinkerFrozenPainterBase.forward_with_carry: each sample is removed
from the active batch at its own halt step, genuinely shrinking later
iterations' compute, instead of the whole batch halting together on a
batch-mean decision) and records each individual sample's actual halting
step via the halt_steps_out hook — so what's reported here is genuine
per-sample data from the real generation, not an estimate. For each denoising step it
reports mean/median/std/min/max/frac_full_budget, pooled across the cached
validation batches and both CFG branches (conditional + unconditional, if
cfg_scale != 1) — answering both "does the head spend more steps on
high-noise (early) denoising steps than low-noise (late) ones" and "how
much does step count vary across samples at a given step" (median well
below mean + a nonzero max/frac_full_budget is the signature of a
bimodal "most samples need almost nothing, a few need everything" pattern).

Usage:
    python experiments/eval_halt_step_profile.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true \\
      +checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_with_halt_head.pt \\
      +profile.reset_every_values=[20,20,5] +profile.thresholds=[-0.0002,0.0,0.0002]

    # Options (all under +profile.*):
    #   reset_every_values / thresholds — parallel lists, zipped element-wise
    #       into (reset_every, threshold) combos. Defaults to 3 combos picked
    #       to span "near full budget" / "best accuracy, still trimmed" /
    #       "aggressive savings, modest accuracy loss" — adjust based on
    #       whichever rows in your own ablate_trm_loop_budget.py halt-axis
    #       sweep look interesting.
    #   num_samples  — default 256 (same default as ablation.num_samples)
    #   seed         — default 0 (same convention as ablate_trm_loop_budget.py)
    #   cfg_scale / num_inference_steps — default from model.sampling_pipeline
    #   out          — json path (default: alongside checkpoint, named
    #                  halt_step_profile.json)

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
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from eval_halt_step_distribution import _distribution_stats
from factory import build_datasets, build_model

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _run_profile_sampling(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    halt_threshold: float,
    reset_fn,
    total_calls: list[int],
    halt_steps_by_denoise_idx: list[list[torch.Tensor]],
) -> None:
    """Runs the real, per-sample dynamic-re-batching halting rule
    (use_halt_head=True) and records both the actual compute cost
    (total_calls, the average steps-per-sample used per denoising step per
    cached batch — same accounting as ablate_trm_loop_budget.py's halt
    axis) and each individual sample's own halting step
    (halt_steps_by_denoise_idx, one (B,) tensor per denoising step per
    cached batch, pooling the conditional and unconditional CFG branches
    together as independent reasoning calls)."""
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

        this_cond_calls: list[int] = []
        this_cond_halt: list[torch.Tensor] = []
        pred_c, z_H_c, z_L_c = model.forward_with_carry(
            step_sample, z_H_c, z_L_c, use_halt_head=True, halt_threshold=halt_threshold,
            steps_used=this_cond_calls, halt_steps_out=this_cond_halt,
        )
        noise_pred = pred_c.pred
        halt_steps_by_denoise_idx[step_idx].append(this_cond_halt[0])
        step_calls = this_cond_calls[0]

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            this_uncond_calls: list[int] = []
            this_uncond_halt: list[torch.Tensor] = []
            pred_u, z_H_u, z_L_u = model.forward_with_carry(
                null_sample, z_H_u, z_L_u, use_halt_head=True, halt_threshold=halt_threshold,
                steps_used=this_uncond_calls, halt_steps_out=this_uncond_halt,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)
            halt_steps_by_denoise_idx[step_idx].append(this_uncond_halt[0])
            step_calls += this_uncond_calls[0]

        total_calls.append(step_calls)
        x = model.scheduler.step(noise_pred, t, x).prev_sample


def _run_profile_config(
    model,
    cached_batches: list[dict],
    num_inference_steps: int,
    cfg_scale: float,
    halt_threshold: float,
    reset_fn,
    n_sup: int,
) -> dict:
    total_calls: list[int] = []
    halt_steps_by_denoise_idx: list[list[torch.Tensor]] = [[] for _ in range(num_inference_steps)]

    for cb in cached_batches:
        _run_profile_sampling(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale,
            halt_threshold, reset_fn, total_calls, halt_steps_by_denoise_idx,
        )

    per_denoise_stats = []
    all_steps = []
    for step_idx in range(num_inference_steps):
        pooled = torch.cat(halt_steps_by_denoise_idx[step_idx], dim=0)
        per_denoise_stats.append(_distribution_stats(pooled, n_sup))
        all_steps.append(pooled)

    return {
        "avg_total_calls_per_denoise_step": sum(total_calls) / len(cached_batches) / num_inference_steps,
        "avg_total_calls": sum(total_calls) / len(cached_batches),
        "per_denoise_step": per_denoise_stats,
        "pooled": _distribution_stats(torch.cat(all_steps, dim=0), n_sup),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/eval_halt_step_profile.py experiment=<name> "
            "checkpoint=<path/to/checkpoint_with_halt_head.pt> thinker.with_halt_head=true "
            "[+profile.xxx=...]"
        )

    pf = cfg.get("profile", {})
    reset_every_values: list = list(pf.get("reset_every_values", [20, 20, 5]))
    thresholds: list[float] = list(pf.get("thresholds", [-0.0002, 0.0, 0.0002]))
    if len(reset_every_values) != len(thresholds):
        raise SystemExit(
            "profile.reset_every_values and profile.thresholds must be the same length "
            "(they're zipped element-wise into (reset_every, threshold) combos)."
        )
    num_samples: int = pf.get("num_samples", 256)
    seed: int = pf.get("seed", 0)
    out_path: str = pf.get("out", str(Path(checkpoint).parent / "halt_step_profile.json"))

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

    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={n_sup}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in model.scheduler.timesteps]

    results: dict[str, dict] = {}
    for reset_every, threshold in zip(reset_every_values, thresholds):
        key = f"reset_every={reset_every}/threshold={threshold}"
        logger.info(f"Profiling {key} ...")
        results[key] = _run_profile_config(
            model, cached_batches, num_inference_steps, cfg_scale, threshold, _make_reset_fn(reset_every), n_sup,
        )
        logger.info(f"  → pooled: {results[key]['pooled']}  avg_total_calls: {results[key]['avg_total_calls']:.1f}")

    if accelerator.is_main_process:
        for key, r in results.items():
            print("\n" + "=" * 110)
            print(f"{key}  (avg_total_calls={r['avg_total_calls']:.1f}, i.e. real compute cost per trajectory)")
            print("=" * 110)
            p = r["pooled"]
            print(
                f"  pooled: n={p['n']}  mean={p['mean']:.2f}  std={p['std']:.2f}  min={p['min']}  max={p['max']}  "
                f"p10={p['p10']:.1f}  p50(median)={p['p50']:.1f}  p90={p['p90']:.1f}  "
                f"frac_full_budget={p['frac_full_budget']:.3f}"
            )
            print("\n" + "-" * 110)
            print(
                f"{'denoise_idx':>12}{'timestep':>10}{'mean':>8}{'median':>8}{'std':>8}{'min':>6}{'max':>6}"
                f"{'p10':>7}{'p90':>7}{'frac_full':>11}"
            )
            print("-" * 110)
            for i, s in enumerate(r["per_denoise_step"]):
                print(
                    f"{i:>12}{timesteps[i]:>10}{s['mean']:>8.2f}{s['p50']:>8.1f}{s['std']:>8.2f}"
                    f"{s['min']:>6}{s['max']:>6}{s['p10']:>7.1f}{s['p90']:>7.1f}{s['frac_full_budget']:>11.3f}"
                )
            print("-" * 110)

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