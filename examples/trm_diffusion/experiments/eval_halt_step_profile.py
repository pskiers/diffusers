"""
experiments/eval_halt_step_profile.py — Show how the halt head allocates its
reasoning-step budget across the denoising trajectory, for a handful of
already-identified (reset_every, halt_threshold) configs.

experiments/ablate_trm_loop_budget.py's "halt" axis only reports one number
per config (total_sup_calls, summed over the whole trajectory) — useful for
picking a good operating point, but it hides *where* in the trajectory those
calls are spent. This script picks a small number of configs (e.g. ones that
looked good in that sweep) and records the actual number of reasoning steps
used at each individual denoising step — separately for the conditional and
unconditional CFG branches, if cfg_scale != 1 — averaged across the same
cached validation batches the ablation script uses, so results are directly
comparable to it. Answers: does the head spend more steps on high-noise
(early) denoising steps than low-noise (late) ones, or the reverse, or is it
roughly flat?

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
def _run_profile_sampling(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    halt_threshold: float,
    reset_fn,
    cond_steps: list[list[int]],
    uncond_steps: list[list[int]],
) -> None:
    """Like ablate_trm_loop_budget._run_halt_ablation_sampling, but records
    the per-denoising-step reasoning-step count into cond_steps[i]/
    uncond_steps[i] (one appended value per cached batch, per denoising step
    index i) instead of collapsing the whole trajectory into one total."""
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

        this_cond: list[int] = []
        pred_c, z_H_c, z_L_c = model.forward_with_carry(
            step_sample, z_H_c, z_L_c, use_halt_head=True, halt_threshold=halt_threshold, steps_used=this_cond,
        )
        noise_pred = pred_c.pred
        cond_steps[step_idx].append(this_cond[0])

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            this_uncond: list[int] = []
            pred_u, z_H_u, z_L_u = model.forward_with_carry(
                null_sample, z_H_u, z_L_u, use_halt_head=True, halt_threshold=halt_threshold, steps_used=this_uncond,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)
            uncond_steps[step_idx].append(this_uncond[0])

        x = model.scheduler.step(noise_pred, t, x).prev_sample


def _run_profile_config(
    model,
    cached_batches: list[dict],
    num_inference_steps: int,
    cfg_scale: float,
    halt_threshold: float,
    reset_fn,
) -> dict:
    cond_steps: list[list[int]] = [[] for _ in range(num_inference_steps)]
    uncond_steps: list[list[int]] = [[] for _ in range(num_inference_steps)]

    for cb in cached_batches:
        _run_profile_sampling(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale,
            halt_threshold, reset_fn, cond_steps, uncond_steps,
        )

    result = {"cond_steps_by_denoise_idx": [float(np.mean(s)) for s in cond_steps]}
    if cfg_scale != 1.0:
        result["uncond_steps_by_denoise_idx"] = [float(np.mean(s)) for s in uncond_steps]
        result["total_steps_by_denoise_idx"] = [
            c + u for c, u in zip(result["cond_steps_by_denoise_idx"], result["uncond_steps_by_denoise_idx"])
        ]
    else:
        result["total_steps_by_denoise_idx"] = result["cond_steps_by_denoise_idx"]
    return result


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

    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={model.n_sup}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in model.scheduler.timesteps]

    results: dict[str, dict] = {}
    for reset_every, threshold in zip(reset_every_values, thresholds):
        key = f"reset_every={reset_every}/threshold={threshold}"
        logger.info(f"Profiling {key} ...")
        results[key] = _run_profile_config(
            model, cached_batches, num_inference_steps, cfg_scale, threshold, _make_reset_fn(reset_every),
        )
        rounded = [round(v, 1) for v in results[key]["total_steps_by_denoise_idx"]]
        logger.info(f"  → total steps by denoise idx: {rounded}")

    if accelerator.is_main_process:
        col_w = max(20, max(len(k) for k in results) + 2)
        print("\n" + "=" * (24 + col_w * len(results)))
        header = f"{'denoise_idx':>12}{'timestep':>12}" + "".join(k.rjust(col_w) for k in results)
        print(header)
        print("=" * (24 + col_w * len(results)))
        for i in range(num_inference_steps):
            row = f"{i:>12}{timesteps[i]:>12}"
            for k in results:
                row += f"{results[k]['total_steps_by_denoise_idx'][i]:>{col_w}.2f}"
            print(row)
        print("=" * (24 + col_w * len(results)))

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"checkpoint": str(checkpoint), "timesteps": timesteps, "results": results},
                f,
                indent=2,
            )
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
