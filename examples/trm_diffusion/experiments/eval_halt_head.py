"""
experiments/eval_halt_head.py — Evaluate a trained adaptive-halting head
(models.trm_wrappers.SpatialTRM.halt_head) in isolation, on held-out data.

Rolls out the frozen thinker+painter checkpoint's full n_sup trajectory on
validation data (model.eval(), no_grad — no optimizer step, unlike
train_halt_head.py) and compares the halt head's predictions against the
actual future loss reduction (loss_t - min(loss_{t+1..})) that
train_halt_head.py trains it to predict. Reports:

  - calibration: Pearson r / R² / MAE between predicted and actual targets,
    pooled over all steps and samples (from held-out data the head was
    never fit on — train_halt_head.py only ever trains on train_ds).
  - per-threshold halting behavior: for a list of halt_threshold values,
    the step at which predict_halt_value(z_H).mean() <= threshold would
    trigger (mirrors ThinkerFrozenPainterBase.forward_with_carry's
    use_halt_head loop) per sample, the resulting average steps used, and
    "regret" — actual loss at the halt step minus the oracle-best loss
    achievable anywhere in that sample's own trajectory — averaged.

This only tells you whether the regression itself is any good on data it
wasn't fit on; it says nothing about whether early-exiting actually helps
downstream image/puzzle quality at matched compute — see
experiments/ablate_trm_loop_budget.py's "halt" axis for that end-to-end
check.

Usage:
    python experiments/eval_halt_head.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_with_halt_head.pt \\
      +halt_eval.num_batches=50

    # Options (all under +halt_eval.*):
    #   num_batches — validation batches to roll out (default 50)
    #   batch_size  — default cfg.eval.batch_size / cfg.train.batch_size
    #   thresholds  — halt_threshold values to sweep
    #                 (default [-0.05, -0.02, 0.0, 0.02, 0.05])
    #   out         — json path for results (default: alongside the checkpoint,
    #                 named halt_head_eval.json)
    # use_ema=false to continue from raw (non-EMA) weights (default true,
    # matching eval.py / ablate_trm_loop_budget.py's own default).

Config overrides work exactly like train_trm.py / eval.py.
"""

from __future__ import annotations

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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from factory import build_datasets, build_model
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Duplicated from eval.py / train_halt_head.py (see the latter's own
    copy's docstring for why this isn't imported instead). Keep in sync with
    eval.py's version if the checkpoint format changes."""
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


@torch.no_grad()
def _rollout_batch(model, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen model's full n_sup trajectory on one batch.

    Returns:
        preds:   (n_sup, B) halt_head predictions after each step
        losses:  (n_sup, B) real per-sample diffusion loss after each step
    """
    [d] = model.prep_mb_data([batch], device)
    preds, losses = [], []
    for _ in range(model.n_sup):
        noise_pred, logits, d["z_H"], d["z_L"] = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
        per_sample_loss = (noise_pred.float() - d["sample"].target.float()).pow(2).flatten(1).mean(1)
        pred = model.thinker.predict_halt_value(d["z_H"])
        preds.append(pred.detach())
        losses.append(per_sample_loss.detach())
    return torch.stack(preds, dim=0), torch.stack(losses, dim=0)


def _calibration_metrics(all_preds: torch.Tensor, all_losses: torch.Tensor) -> dict:
    """Pool (prediction, actual future-loss-reduction) pairs across all
    steps/samples/batches and report standard regression-quality metrics.

    all_preds / all_losses: (n_sup, N) — predictions/losses for every
    rolled-out sample, stacked across batches. The last step is skipped
    (no future to compare against), matching train_halt_head's own target
    construction.
    """
    n_sup = all_losses.shape[0]
    suffix_min = torch.flip(torch.cummin(torch.flip(all_losses, dims=[0]), dim=0).values, dims=[0])
    future_min = suffix_min[1:]
    targets = (all_losses[:-1] - future_min).reshape(-1)
    preds = all_preds[:-1].reshape(-1)

    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()
    r, _ = pearsonr(preds_np, targets_np)
    mae = float(np.abs(preds_np - targets_np).mean())
    ss_res = float(((targets_np - preds_np) ** 2).sum())
    ss_tot = float(((targets_np - targets_np.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_pairs": int(targets_np.shape[0]),
        "pearson_r": float(r),
        "r2": r2,
        "mae": mae,
        "target_mean": float(targets_np.mean()),
        "target_std": float(targets_np.std()),
        "pred_mean": float(preds_np.mean()),
        "pred_std": float(preds_np.std()),
        "n_sup": n_sup,
    }


def _threshold_metrics(all_preds: torch.Tensor, all_losses: torch.Tensor, threshold: float) -> dict:
    """For one halt_threshold, replay the same per-sample halting rule as
    ThinkerFrozenPainterBase.forward_with_carry — except using each
    individual sample's own prediction (batch-mean in the real inference
    loop bundles a whole batch into one decision; here every sample gets
    its own halting step so the regret/steps-used stats reflect the head's
    per-sample behavior, not one batch-mean trigger).

    Regret: loss at the (per-sample) halt step minus the best loss anywhere
    in that sample's own trajectory (the oracle, compute-unaware optimum).
    """
    n_sup, n = all_preds.shape
    device = all_preds.device
    halts = all_preds <= threshold  # (n_sup, N)
    has_halt = halts.any(dim=0)
    # first True index along dim 0, or n_sup - 1 (run to completion) if none
    full_traj = torch.full((n,), n_sup - 1, device=device, dtype=torch.int64)
    first_halt = torch.where(has_halt, halts.float().argmax(dim=0), full_traj)

    halt_loss = all_losses[first_halt, torch.arange(n, device=device)]
    oracle_loss = all_losses.min(dim=0).values
    regret = (halt_loss - oracle_loss)

    return {
        "threshold": threshold,
        "avg_steps_used": float((first_halt + 1).float().mean().item()),
        "frac_full_trajectory": float((first_halt == n_sup - 1).float().mean().item()),
        "avg_regret": float(regret.mean().item()),
        "median_regret": float(regret.median().item()),
        "avg_halt_loss": float(halt_loss.mean().item()),
        "avg_oracle_loss": float(oracle_loss.mean().item()),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/eval_halt_head.py experiment=<name> "
            "checkpoint=<path/to/checkpoint_with_halt_head.pt> thinker.with_halt_head=true "
            "[+halt_eval.xxx=...]"
        )

    he = cfg.get("halt_eval", {})
    num_batches: int = he.get("num_batches", 50)
    batch_size: int = he.get("batch_size", cfg.eval.get("batch_size", cfg.train.batch_size))
    thresholds: list[float] = list(he.get("thresholds", [-0.05, -0.02, 0.0, 0.02, 0.05]))
    use_ema: bool = cfg.get("use_ema", True)
    out_path: str = he.get("out", str(Path(checkpoint).parent / "halt_head_eval.json"))

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

    _, eval_ds = build_datasets(cfg)
    collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Rolling out {num_batches} held-out batches (n_sup={model.n_sup} each) ...")

    all_preds, all_losses = [], []
    for i, batch in enumerate(tqdm(eval_dl, total=num_batches, desc="Rollout")):
        if i >= num_batches:
            break
        preds, losses = _rollout_batch(model, batch, device)
        all_preds.append(preds)
        all_losses.append(losses)

    all_preds = torch.cat(all_preds, dim=1)   # (n_sup, N)
    all_losses = torch.cat(all_losses, dim=1)  # (n_sup, N)

    calibration = _calibration_metrics(all_preds, all_losses)
    threshold_sweep = [_threshold_metrics(all_preds, all_losses, t) for t in thresholds]

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("Halt head calibration (held-out data, pooled over all steps)")
        print("=" * 70)
        for k, v in calibration.items():
            print(f"  {k:<18} {v}")

        print("\n" + "=" * 100)
        print(f"{'threshold':>10}{'avg_steps':>12}{'frac_full':>12}{'avg_regret':>13}{'median_regret':>15}")
        print("=" * 100)
        for r in threshold_sweep:
            print(
                f"{r['threshold']:>10.3f}{r['avg_steps_used']:>12.2f}{r['frac_full_trajectory']:>12.3f}"
                f"{r['avg_regret']:>13.4f}{r['median_regret']:>15.4f}"
            )
        print("=" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "num_batches": num_batches,
                    "batch_size": batch_size,
                    "calibration": calibration,
                    "threshold_sweep": threshold_sweep,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved → {out_path}")

    return {"calibration": calibration, "threshold_sweep": threshold_sweep}


if __name__ == "__main__":
    main()