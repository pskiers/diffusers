"""
experiments/diag_halt_target_by_timestep.py — Does the halt head's training
TARGET (loss_t - min(loss_{t+1..}), see SpatialTRM.train_halt_head) genuinely
carry less signal at high-noise (early-sampling) timesteps, or does the
current halt head just fail to predict a real, similarly-sized signal there?

Read-only diagnostic, no training. For many random batches (real
timesteps ~ Uniform(0, num_train_timesteps), same as normal training), runs
the checkpoint's own full n_sup reasoning trajectory under no_grad and
buckets, by diffusion timestep, both:
  - the TRUE target in both forms: the raw headroom magnitude (useful
    background — "is there real room for more reasoning to help, per
    timestep, in raw MSE units") AND its binarized form, the empirical
    fraction of samples with headroom > label_margin — this is the actual
    quantity train_halt_head's BCE label estimates, call it TRUE P(improves).
  - the CURRENT halt head's own prediction, if the loaded checkpoint has one
    (thinker.with_halt_head=true): both the raw logit and sigmoid(logit) —
    comparable directly against TRUE P(improves) above — this answers "is
    the head's predicted probability anywhere close to the true one, per
    timestep, or is it just flatlined regardless of bucket".

Bucket 0 = t near 0 (cleanest / last sampling steps), bucket N-1 = t near
num_train_timesteps (noisiest / first sampling steps) — the regime the user
reported the head prematurely halts in.

Usage (no-halt-head base checkpoint — true-target-only stats):
    python experiments/diag_halt_target_by_timestep.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      +checkpoint=runs/sudoku-painter-thinker.pt \\
      +diag.num_batches=100

Usage (existing halt-head checkpoint — also reports predicted-vs-true gap):
    python experiments/diag_halt_target_by_timestep.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true +checkpoint=runs/sudoku_painter_thinker_with_halt_head.pt \\
      +diag.num_batches=100

    # Options (all under +diag.*):
    #   num_batches      — how many random batches to pool stats over (default 100)
    #   batch_size       — default cfg.train.batch_size
    #   num_buckets      — timestep buckets (default 10)
    #   use_ema          — default true (matches train_halt_head.py's own default)
    #   label_margin     — single margin (default 0.0), or
    #   label_margins    — a list, e.g. +diag.label_margins=[0.0,1e-4,1e-3] to
    #                      see TRUE P(improves) at several margins in one pass
    #                      (no need to re-run the forward pass per candidate)
    #   out              — json path (default: alongside checkpoint, named
    #                      halt_target_by_timestep.json)
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from factory import build_datasets, build_model
from models.utility_models import strip_compiled_prefix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit("ERROR: No checkpoint specified. Pass +checkpoint=<path>.")

    diag = cfg.get("diag", {})
    num_batches: int = diag.get("num_batches", 100)
    batch_size: int = diag.get("batch_size", cfg.train.batch_size)
    num_buckets: int = diag.get("num_buckets", 10)
    use_ema: bool = diag.get("use_ema", True)
    # Accepts either diag.label_margin (single float) or diag.label_margins
    # (list) — the latter reports frac_positive at every margin in one pass
    # over the data, to calibrate the noise floor without re-running the
    # (expensive) model forward pass per candidate margin.
    label_margins: list = list(diag.get("label_margins", [diag.get("label_margin", 0.0)]))
    out_path: str = diag.get("out", str(Path(checkpoint).parent / "halt_target_by_timestep.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    scheduler = instantiate(cfg.diffusion)
    num_train_timesteps = scheduler.config.num_train_timesteps
    model = build_model(cfg, scheduler)
    has_halt_head = getattr(model.thinker, "with_halt_head", False)

    _load_checkpoint(model, str(checkpoint), use_ema=use_ema, device="cpu")
    model = model.to(device)
    model.eval()

    train_ds, _ = build_datasets(cfg)
    collate_fn = getattr(type(train_ds), "collate_fn", None)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True, collate_fn=collate_fn,
    )
    train_iter = iter(train_dl)

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            return next(train_iter)

    logger.info(
        f"Pooling stats over {num_batches} batches (n_sup={model.n_sup}), "
        f"{num_buckets} timestep buckets over [0, {num_train_timesteps}), "
        f"has_halt_head={has_halt_head}"
    )

    # Per-bucket accumulators.
    target0_sum = torch.zeros(num_buckets)   # loss_0 - min(loss_1..end): headroom right after step 1
    target0_sqsum = torch.zeros(num_buckets)
    target0_count = torch.zeros(num_buckets)
    loss_first_sum = torch.zeros(num_buckets)
    loss_last_sum = torch.zeros(num_buckets)
    pos_label_sum = {m: torch.zeros(num_buckets) for m in label_margins}  # frac(target0 > margin), per margin
    pred0_sum = torch.zeros(num_buckets)      # halt_head's own predict_halt_value at step 1, if available
    pred0_sqsum = torch.zeros(num_buckets)
    pred0_prob_sum = torch.zeros(num_buckets)  # sigmoid(pred0), comparable to pos_label_sum/count

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Collecting"):
            batch = _next_batch()
            [d] = model.prep_mb_data([batch], device)

            timesteps = d["sample"].timesteps.float()
            bucket = (timesteps / num_train_timesteps * num_buckets).long().clamp(0, num_buckets - 1).cpu()

            loss_list, zH0_list = [], []
            for _ in range(model.n_sup):
                noise_pred, logits, d["z_H"], d["z_L"] = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
                per_sample_loss = (noise_pred.float() - d["sample"].target.float()).pow(2).flatten(1).mean(1)
                loss_list.append(per_sample_loss.detach().cpu())
                zH0_list.append(d["z_H"][:, 0].detach())

            losses = torch.stack(loss_list, dim=0)  # (n_sup, B)
            suffix_min = torch.flip(torch.cummin(torch.flip(losses, dims=[0]), dim=0).values, dims=[0])
            target0 = losses[0] - suffix_min[1]  # (B,) headroom available after the very first reasoning step

            target0_sum.scatter_add_(0, bucket, target0)
            target0_sqsum.scatter_add_(0, bucket, target0.pow(2))
            target0_count.scatter_add_(0, bucket, torch.ones_like(target0))
            loss_first_sum.scatter_add_(0, bucket, losses[0])
            loss_last_sum.scatter_add_(0, bucket, losses[-1])
            for m in label_margins:
                pos_label_sum[m].scatter_add_(0, bucket, (target0 > m).float())

            if has_halt_head:
                # zH0_list[0] is already the pooled z_H[:, 0] token (see loop above),
                # so build the head's input via _halt_feat directly instead of
                # predict_halt_value (which expects the full (B, seq_len, hidden)
                # z_H and does that indexing itself) — this exercises the exact
                # same timestep-concatenation path train_halt_head trains against.
                feat = model.thinker._halt_feat(zH0_list[0], timesteps)
                pred0 = model.thinker.halt_head(feat).squeeze(-1).detach().cpu()
                pred0_sum.scatter_add_(0, bucket, pred0)
                pred0_sqsum.scatter_add_(0, bucket, pred0.pow(2))
                pred0_prob_sum.scatter_add_(0, bucket, torch.sigmoid(pred0))

    count = target0_count.clamp(min=1)
    mean_target0 = target0_sum / count
    std_target0 = (target0_sqsum / count - mean_target0.pow(2)).clamp(min=0).sqrt()
    mean_loss_first = loss_first_sum / count
    mean_loss_last = loss_last_sum / count
    frac_positive = {m: pos_label_sum[m] / count for m in label_margins}

    rows = []
    for b in range(num_buckets):
        row = {
            "bucket": b,
            "t_range": [round(b / num_buckets * num_train_timesteps), round((b + 1) / num_buckets * num_train_timesteps)],
            "n": int(target0_count[b].item()),
            "mean_loss_step1": mean_loss_first[b].item(),
            "mean_loss_step_last": mean_loss_last[b].item(),
            # scientific notation here on purpose — a fixed-decimal format
            # previously rounded this to "0.000000" in buckets that turned
            # out to have real (if tiny) sub-threshold floating-point jitter,
            # which silently corrupted label_margin=0 into a near-coin-flip
            # label in that regime (see conversation / commit message).
            "mean_target_headroom_after_step1": mean_target0[b].item(),
            "std_target_headroom_after_step1": std_target0[b].item(),
            "frac_positive_label_after_step1": {m: frac_positive[m][b].item() for m in label_margins},
        }
        if has_halt_head:
            row["mean_predicted_halt_value_step1"] = (pred0_sum[b] / count[b]).item()
            row["mean_predicted_prob_after_step1"] = (pred0_prob_sum[b] / count[b]).item()
        rows.append(row)
        margins_str = "  ".join(
            f"TRUE P(improves)@margin={m:g}={row['frac_positive_label_after_step1'][m]:.3f}" for m in label_margins
        )
        logger.info(
            f"bucket {b} (t~{row['t_range']}, n={row['n']}): "
            f"loss step1={row['mean_loss_step1']:.5f} -> step{model.n_sup}={row['mean_loss_step_last']:.5f}  "
            f"headroom after step1 mean={row['mean_target_headroom_after_step1']:.3e} "
            f"(std={row['std_target_headroom_after_step1']:.3e})  "
            f"{margins_str}"
            + (
                f"  PREDICTED P(improves)={row['mean_predicted_prob_after_step1']:.3f} "
                f"(logit={row['mean_predicted_halt_value_step1']:.3f})"
                if has_halt_head
                else ""
            )
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(checkpoint), "num_batches": num_batches, "rows": rows}, f, indent=2)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
