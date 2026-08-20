"""
experiments/eval_steering_reuse.py — Check whether the trained halt head,
queried on the z_H CARRIED IN from the previous denoising step (i.e. before
running any fresh reasoning at the current step), predicts whether reusing
the previous step's steering — skipping reasoning entirely this step — would
actually be safe.

This is a genuine extrapolation for the head: train_halt_head.py only ever
trains it on (z_H, loss) pairs where z_H evolved via recursion cycles at a
FIXED x_noisy/timestep — it has never seen a z_H paired with a DIFFERENT
(lower-noise) timestep's x_noisy while itself unchanged. This script
measures whether that extrapolation happens to still be useful; it doesn't
argue it should be.

Runs REAL sampling trajectories (mirrors ablate_trm_loop_budget.py's
_run_halt_ablation_sampling: real DDIM rollout, TRM carry across denoising
steps, real per-sample dynamic-re-batching halting WITHIN each step via
use_halt_head=True) — the deployed reference behavior, completely
unmodified; nothing here changes what image gets generated. At every
denoising step after the first (and after every carry reset), before that
step's own reasoning runs, records:

  pred_skip_k   = predict_halt_value(z_H carried in from step k-1)
  actual_cost_k = per-sample MSE between:
                    - noise_pred using step k's ACTUAL fresh steering (what
                      the real trajectory used)
                    - noise_pred using step k-1's steering reused unchanged
                      (one extra painter forward call — cheap relative to
                      the reasoning compute this would save; the
                      thinker/translator are not rerun for it)

This directly measures "if the head says skip here, how much would the
output actually have changed by skipping" — the ground truth the proposed
skip-reasoning-across-denoising-steps mechanism would need to get right, on
the trajectory the model actually walks (not synthetic single-timestep
teacher-forced data — the question is inherently about consecutive real
timesteps). Only the CFG-conditional branch is instrumented; the
unconditional branch (when cfg_scale != 1) runs untouched but isn't checked.

pred_skip and actual_cost are NOT the same quantity in the same units (the
head predicts a within-step loss-improvement estimate; actual_cost is an
output-space MSE from a cross-step substitution) — so this reports
correlation (Pearson r, Spearman rho) as a discriminative-power diagnostic,
not calibration (no R²/MAE claiming the numbers should match). The decision-
relevant output is the skip_threshold sweep: for each threshold, the
fraction of eligible steps it would skip and the actual_cost actually
incurred among just those — the real damage that threshold's decisions
would cause, not a hypothetical.

This only tells you whether pred_skip correlates with actual_cost at a
single step in isolation — it does NOT run the skip-and-reuse trajectory
end-to-end (a skip decision at step k changes step k+1's x_noisy, which
compounds), so it can't show downstream image/accuracy impact by itself. If
the correlation here looks promising, the next step is a new
ablate_trm_loop_budget.py axis that actually skips (reuses previous logits
and z_H unchanged, rather than just halting within a step) for the real
end-to-end check.

Usage:
    python experiments/eval_steering_reuse.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_with_halt_head.pt \\
      +reuse_eval.num_samples=256 +reuse_eval.reset_every=20 +reuse_eval.halt_threshold=0.0002

    # Options (all under +reuse_eval.*):
    #   num_samples          — validation samples to roll out (default 256)
    #   batch_size           — default cfg.eval.batch_size / cfg.train.batch_size
    #   num_inference_steps  — default model.sampling_pipeline.num_inference_steps
    #   cfg_scale            — default model.sampling_pipeline.cfg_scale
    #   reset_every          — TRM carry reset interval across denoising
    #                          steps (default 20; null = never reset after
    #                          step 0) — matches whatever config this head
    #                          was actually found to work well under.
    #   halt_threshold       — WITHIN-step halting threshold for the
    #                          reference trajectory's own reasoning (default
    #                          0.0002) — this script is about skipping whole
    #                          steps, not about within-step halting, so just
    #                          match whatever you'd actually deploy.
    #   skip_thresholds      — pred_skip thresholds to sweep
    #                          (default [-0.002,-0.001,-0.0005,0.0,0.0005])
    #   seed                 — default 0
    #   out                  — json path (default: alongside the checkpoint,
    #                          named steering_reuse_eval.json)
    # use_ema=false to continue from raw (non-EMA) weights (default true).

Config overrides work exactly like train_trm.py / eval.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from factory import build_datasets, build_model
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Duplicated from eval.py / train_halt_head.py / eval_halt_head.py (see
    train_halt_head.py's own copy's docstring for why this isn't imported
    instead). Keep in sync with eval.py's version if the checkpoint format
    changes."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = strip_compiled_prefix(ckpt["model_state"])
    model.load_state_dict(sd, strict=False)
    step = ckpt.get("step")

    if use_ema and ckpt.get("ema_state"):
        ema_sd = strip_compiled_prefix(ckpt["ema_state"])
        model.load_state_dict(ema_sd, strict=False)
        logger.info(f"Loaded EMA weights on top of model_state (step={step})")
    else:
        logger.info(f"Loaded model_state (step={step}, use_ema={use_ema})")
    return step


def _make_reset_fn(reset_every: Optional[int]) -> Callable[[int], bool]:
    """Same convention as ablate_trm_loop_budget.py's _make_reset_fn."""
    if reset_every is None:
        return lambda step_idx: step_idx == 0
    return lambda step_idx: step_idx % reset_every == 0


@torch.no_grad()
def _rollout_trajectory(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    halt_threshold: float,
    reset_fn: Callable[[int], bool],
) -> tuple[list, list, list, list]:
    """Real DDIM rollout, mirroring ablate_trm_loop_budget.py's
    _run_halt_ablation_sampling exactly (real per-sample dynamic-re-batching
    halting within each step) — the deployed reference behavior, unmodified.
    Additionally records pred_skip/actual_cost for the conditional branch at
    every step where a "previous step" genuinely exists (not step 0, not
    right after a carry reset).

    Returns: (pred_skip_list, actual_cost_list, step_idx_list, t_list).
    pred_skip_list/actual_cost_list are lists of (B,) tensors; step_idx_list/
    t_list are parallel lists of plain ints/floats (one denoising step's
    position/actual timestep is shared by the whole batch, unlike the
    per-sample cost) — lets callers pool across trajectories while still
    breaking results down by where in the schedule they occurred (see
    _per_step_breakdown). The generated image itself is discarded: this
    script only cares about the per-step comparison, not the final output
    (see ablate_trm_loop_budget.py for the end-to-end check).
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None
    prev_logits_c = None

    pred_skip_list, actual_cost_list, step_idx_list, t_list = [], [], [], []

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        can_compare = step_idx > 0 and z_H_c is not None and prev_logits_c is not None
        if can_compare:
            pred_skip = model.thinker.predict_halt_value(z_H_c)
            reused_noise_pred = model.run_painter(step_sample, prev_logits_c)

        pred_c, z_H_c, z_L_c = model.forward_with_carry(
            step_sample, z_H_c, z_L_c, use_halt_head=True, halt_threshold=halt_threshold,
        )
        noise_pred = pred_c.pred

        if can_compare:
            actual_cost = (reused_noise_pred.float() - noise_pred.float()).pow(2).flatten(1).mean(1)
            pred_skip_list.append(pred_skip.detach())
            actual_cost_list.append(actual_cost.detach())
            step_idx_list.append(step_idx)
            t_list.append(float(t.item()))

        prev_logits_c = pred_c.logits

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, z_H_u, z_L_u = model.forward_with_carry(
                null_sample, z_H_u, z_L_u, use_halt_head=True, halt_threshold=halt_threshold,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return pred_skip_list, actual_cost_list, step_idx_list, t_list


def _discrimination_metrics(pred_skip_list: list, actual_cost_list: list) -> dict:
    """Pool (pred_skip, actual_cost) pairs across every eligible step and
    trajectory. Pearson r / Spearman rho only — these are two different
    quantities in different units, so this is a discrimination check ("does
    low pred_skip track low actual_cost"), not a calibration claim."""
    preds = torch.cat(pred_skip_list).cpu().numpy()
    costs = torch.cat(actual_cost_list).cpu().numpy()
    r, _ = pearsonr(preds, costs)
    rho, _ = spearmanr(preds, costs)
    return {
        "n_pairs": int(costs.shape[0]),
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "pred_skip_mean": float(preds.mean()),
        "pred_skip_std": float(preds.std()),
        "actual_cost_mean": float(costs.mean()),
        "actual_cost_std": float(costs.std()),
    }


def _threshold_metrics(pred_skip_list: list, actual_cost_list: list, threshold: float) -> dict:
    """For one skip_threshold: mark every (step, sample) pair 'would skip'
    if pred_skip <= threshold, and report the actual_cost incurred
    specifically among those pairs — the real damage this threshold's skip
    decisions would cause."""
    preds = torch.cat(pred_skip_list)
    costs = torch.cat(actual_cost_list)
    would_skip = preds <= threshold
    frac_skipped = float(would_skip.float().mean().item())

    if would_skip.any():
        skipped_cost = costs[would_skip]
        avg_cost = float(skipped_cost.mean().item())
        median_cost = float(skipped_cost.median().item())
        max_cost = float(skipped_cost.max().item())
    else:
        avg_cost = median_cost = max_cost = float("nan")

    return {
        "threshold": threshold,
        "frac_would_skip": frac_skipped,
        "avg_cost_if_skipped": avg_cost,
        "median_cost_if_skipped": median_cost,
        "max_cost_if_skipped": max_cost,
    }


def _per_step_breakdown(
    pred_skip_list: list, actual_cost_list: list, step_idx_list: list, t_list: list
) -> list[dict]:
    """Group every (step, sample) pair by which denoising step it came from
    (pooling across trajectories/batches — the schedule is deterministic, so
    the same step_idx always corresponds to the same t) and report per-step
    stats. Answers "is actual_cost concentrated at a particular point in the
    trajectory, or spread evenly" — pooled discrimination/threshold numbers
    can't show that, since they collapse position in the trajectory away
    entirely."""
    by_step: dict[int, dict] = {}
    for pred_skip, actual_cost, step_idx, t in zip(pred_skip_list, actual_cost_list, step_idx_list, t_list):
        entry = by_step.setdefault(step_idx, {"pred_skip": [], "actual_cost": [], "t": t})
        entry["pred_skip"].append(pred_skip)
        entry["actual_cost"].append(actual_cost)

    rows = []
    for step_idx in sorted(by_step):
        entry = by_step[step_idx]
        preds = torch.cat(entry["pred_skip"])
        costs = torch.cat(entry["actual_cost"])
        rows.append({
            "step_idx": step_idx,
            "t": entry["t"],
            "n": int(costs.shape[0]),
            "mean_pred_skip": float(preds.mean().item()),
            "mean_cost": float(costs.mean().item()),
            "median_cost": float(costs.median().item()),
            "max_cost": float(costs.max().item()),
        })
    return rows


def _build_cached_batches(model, dataloader, device, num_samples: int, seed: int) -> list[dict]:
    """Cache a fixed set of (conditions, x_init) once — same pairing every
    threshold gets evaluated against, matching
    ablate_trm_loop_budget.py's _build_cached_batches (solutions/given_masks
    dropped: this script has no downstream classifier check, unlike that
    one)."""
    torch.manual_seed(seed)
    cached = []
    n_done = 0
    for batch in dataloader:
        if n_done >= num_samples:
            break
        conditions = model._batch_to_sample(batch, device)
        bsz = conditions.images.shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)
        cached.append({"conditions": conditions, "x_init": x_init})
        n_done += bsz
    logger.info(f"Cached {n_done} samples across {len(cached)} batches.")
    return cached


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/eval_steering_reuse.py experiment=<name> "
            "checkpoint=<path/to/checkpoint_with_halt_head.pt> thinker.with_halt_head=true "
            "[+reuse_eval.xxx=...]"
        )

    re_cfg = cfg.get("reuse_eval", {})
    num_samples: int = re_cfg.get("num_samples", 256)
    batch_size: int = re_cfg.get("batch_size", cfg.eval.get("batch_size", cfg.train.batch_size))
    reset_every = re_cfg.get("reset_every", 20)
    halt_threshold: float = re_cfg.get("halt_threshold", 0.0002)
    skip_thresholds: list[float] = list(re_cfg.get("skip_thresholds", [-0.002, -0.001, -0.0005, 0.0, 0.0005]))
    seed: int = re_cfg.get("seed", 0)
    use_ema: bool = cfg.get("use_ema", True)
    out_path: str = re_cfg.get("out", str(Path(checkpoint).parent / "steering_reuse_eval.json"))

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

    _load_checkpoint(model, str(checkpoint), use_ema=use_ema, device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    cfg_scale: float = re_cfg.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = re_cfg.get("num_inference_steps", pipeline.num_inference_steps)
    reset_fn = _make_reset_fn(reset_every)

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  "
        f"reset_every={reset_every}  halt_threshold={halt_threshold}"
    )

    _, eval_ds = build_datasets(cfg)
    collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    all_pred_skip, all_actual_cost, all_step_idx, all_t = [], [], [], []
    for cb in tqdm(cached_batches, desc="Rollout"):
        pred_skip_list, actual_cost_list, step_idx_list, t_list = _rollout_trajectory(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale, halt_threshold, reset_fn,
        )
        all_pred_skip.extend(pred_skip_list)
        all_actual_cost.extend(actual_cost_list)
        all_step_idx.extend(step_idx_list)
        all_t.extend(t_list)

    if not all_pred_skip:
        raise SystemExit(
            "No eligible (pred_skip, actual_cost) pairs were recorded — reset_every may be forcing a "
            "carry reset at every step. Increase reset_every (or set it to null) so at least one "
            "denoising step per reset interval has a real 'previous step' to compare against."
        )

    discrimination = _discrimination_metrics(all_pred_skip, all_actual_cost)
    threshold_sweep = [_threshold_metrics(all_pred_skip, all_actual_cost, t) for t in skip_thresholds]
    per_step = _per_step_breakdown(all_pred_skip, all_actual_cost, all_step_idx, all_t)

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("Steering-reuse discrimination (pred_skip vs. actual_cost, pooled)")
        print("=" * 70)
        for k, v in discrimination.items():
            print(f"  {k:<18} {v}")

        print("\n" + "=" * 100)
        print(f"{'threshold':>12}{'frac_skip':>12}{'avg_cost':>15}{'median_cost':>15}{'max_cost':>15}")
        print("=" * 100)
        for r in threshold_sweep:
            print(
                f"{r['threshold']:>12.6f}{r['frac_would_skip']:>12.3f}"
                f"{r['avg_cost_if_skipped']:>15.6f}{r['median_cost_if_skipped']:>15.6f}"
                f"{r['max_cost_if_skipped']:>15.6f}"
            )
        print("=" * 100)

        print("\n" + "=" * 100)
        print("Per-step breakdown (is actual_cost concentrated at a particular point in the trajectory?)")
        print(f"{'step_idx':>10}{'t':>10}{'n':>8}{'mean_pred_skip':>18}{'mean_cost':>14}{'median_cost':>14}{'max_cost':>14}")
        print("=" * 100)
        for r in per_step:
            print(
                f"{r['step_idx']:>10d}{r['t']:>10.1f}{r['n']:>8d}{r['mean_pred_skip']:>18.6f}"
                f"{r['mean_cost']:>14.6f}{r['median_cost']:>14.6f}{r['max_cost']:>14.6f}"
            )
        print("=" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "num_samples": num_samples,
                    "reset_every": reset_every,
                    "halt_threshold": halt_threshold,
                    "discrimination": discrimination,
                    "threshold_sweep": threshold_sweep,
                    "per_step_breakdown": per_step,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved → {out_path}")

    return {"discrimination": discrimination, "threshold_sweep": threshold_sweep, "per_step_breakdown": per_step}


if __name__ == "__main__":
    main()
