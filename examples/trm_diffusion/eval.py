"""
eval.py — Generic evaluation script for TRM-diffusion models.

Loads a checkpoint, instantiates the model from the same Hydra config used
during training, and runs all configured eval callbacks on the full val dataset.

Usage:
    python eval.py experiment=mnist_unet_painter checkpoint=runs/mnist_unet_painter/checkpoint_final.pt
    python eval.py experiment=mnist_thinker_controlnet checkpoint=runs/my_run/checkpoint_step-50000.pt
    python eval.py experiment=clevr_dit_painter checkpoint=... eval.batch_size=64

    # Override where metrics are saved (default: <checkpoint_dir>/eval_metrics.json):
    python eval.py ... +metrics_path=results/my_run.json

    # Skip EMA, use raw model weights:
    python eval.py ... +use_ema=false

    # Diagnostic: keep BatchNorm in train-mode stats during sampling:
    python eval.py ... +force_bn_train=true

    # Reset every BatchNorm's running_mean/running_var and re-accumulate
    # them from N real train()-mode forward passes over the train split —
    # no weights touched, only the running-stat buffers. Fixes running
    # stats that drifted to extreme values under training instability
    # (see models/paper_unet.py) without retraining:
    python eval.py ... +recalibrate_bn_batches=100

    # Save every eval callback's wandb.Image panel(s) to disk as PNGs — the
    # callback builds these either way (best-of-N sample vs. condition vs.
    # ground truth), but without an active wandb run they're normally just
    # discarded when _save_metrics drops non-JSON-serializable values:
    python eval.py ... +save_panels_dir=results/panels

    # Override which callbacks to run:
    python eval.py experiment=... checkpoint=... eval_callbacks=sudoku_ddim

Config overrides work exactly like train_trm.py — all Hydra override syntax is supported.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from factory import build_datasets, build_model
from hydra.utils import instantiate
from models.utility_models import load_checkpoint as _load_checkpoint
from models.utility_models import recalibrate_batchnorm as _recalibrate_batchnorm

logger = get_logger(__name__, log_level="INFO")


def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    width = max((len(k) for k in metrics), default=0)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<{width}}  {v:.4f}")
        else:
            print(f"  {k:<{width}}  {v}")
    print("=" * 60)


def _save_metrics(metrics: dict, path: str, extra: dict | None = None) -> None:
    out = {}
    if extra:
        out.update(extra)

    # Some eval_callbacks (e.g. sudoku thinker's wandb sample panels) put
    # non-JSON-serializable objects (wandb.Image, lists of them, ...) into
    # the same dict used for wandb logging — those aren't meaningful in a
    # JSON metrics summary anyway, so drop them here rather than crashing.
    json_metrics = {}
    dropped = []
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            json_metrics[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        else:
            dropped.append(k)
    if dropped:
        logger.info(f"Skipping non-JSON-serializable metric(s) in saved output: {dropped}")
    out["metrics"] = json_metrics

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved → {path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        print(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python eval.py experiment=<name> checkpoint=<path/to/checkpoint.pt>",
            file=sys.stderr,
        )
        sys.exit(1)

    use_ema: bool = cfg.get("use_ema", True)

    default_metrics_path = str(Path(checkpoint).parent / "eval_metrics.json")
    metrics_path: str = cfg.get("metrics_path", default_metrics_path)

    torch.set_float32_matmul_precision("high")

    accelerator = Accelerator(
        mixed_precision=cfg.precision.mixed_precision,
    )
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint : {checkpoint}")
        logger.info(f"Use EMA    : {use_ema}")
        logger.info(f"Metrics out: {metrics_path}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, eval_ds = build_datasets(cfg)
    logger.info(f"Val dataset: {type(eval_ds).__name__}  ({len(eval_ds)} samples)")

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

    # ── Model ─────────────────────────────────────────────────────────────────
    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)

    # See train_trm.py: closed-loop eval callbacks need the dataset's fitted
    # normalizer, which can't be expressed as a static Hydra config value.
    eval_normalizer = getattr(eval_ds, "normalizer", None)
    if eval_normalizer is not None:
        for cb in getattr(model, "eval_callbacks", []) or []:
            if getattr(cb, "normalizer", None) is None:
                cb.normalizer = eval_normalizer

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    step = _load_checkpoint(model, str(checkpoint), use_ema=use_ema, device="cpu")

    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)

    # ── Optional steering ablation/amplification (diagnostic) ──────────────────
    # +steering_scale=0.0 fully ablates the thinker's steering (pure frozen-
    # painter baseline); +steering_scale=2.0/5.0/10.0 amplifies it. See
    # ThinkerSteering.scaled() in models/interfaces.py.
    steering_scale = cfg.get("steering_scale", None)
    if steering_scale is not None:
        translator = getattr(unwrapped, "thinker_painter_translator", None)
        if translator is None:
            logger.warning("steering_scale set but model has no thinker_painter_translator — ignoring.")
        else:
            orig_forward = translator.forward

            def _scaled_forward(*args, _orig=orig_forward, _scale=steering_scale, **kwargs):
                return _orig(*args, **kwargs).scaled(_scale)

            translator.forward = _scaled_forward
            logger.info(f"Steering scale override active: {steering_scale}x")

    # ── Eval callbacks ────────────────────────────────────────────────────────
    unwrapped.eval()

    recalibrate_bn_batches = int(cfg.get("recalibrate_bn_batches", 0))
    if recalibrate_bn_batches > 0:
        recal_collate_fn = getattr(type(train_ds), "collate_fn", None)
        recal_dl = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=recal_collate_fn
        )
        n_bn = _recalibrate_batchnorm(unwrapped, recal_dl, accelerator.device, recalibrate_bn_batches)
        logger.info(f"Recalibrated {n_bn} BatchNorm modules from {recalibrate_bn_batches} train-mode batches")

    if cfg.get("force_bn_train", False):
        n_bn = 0
        for m in unwrapped.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                n_bn += 1
        logger.info(f"force_bn_train: {n_bn} BatchNorm modules set back to train()")

    callbacks = getattr(unwrapped, "eval_callbacks", [])
    if not callbacks:
        logger.warning(
            "No eval_callbacks configured on this model. "
            "Make sure your experiment config includes an eval_callbacks group."
        )

    save_panels_dir = cfg.get("save_panels_dir", None)

    metrics: dict[str, float] = {}
    with torch.no_grad():
        for cb in callbacks:
            logger.info(f"Running {type(cb).__name__} …")
            cb_metrics = cb(unwrapped, eval_dl, accelerator, step=step)

            # Extract this callback's own panels before merging into the
            # aggregate `metrics` dict below — dict.update would otherwise
            # silently overwrite an earlier callback's "samples" key with a
            # later one's if more than one callback is configured.
            panels = cb_metrics.get("samples") if accelerator.is_main_process else None
            if save_panels_dir and panels:
                panel_dir = Path(save_panels_dir) / type(cb).__name__
                panel_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(panels):
                    pil_img = getattr(img, "image", None)
                    if pil_img is not None:
                        pil_img.save(panel_dir / f"panel_{i:03d}.png")
                logger.info(f"Saved {len(panels)} panel(s) -> {panel_dir}")

            metrics.update(cb_metrics)
            if accelerator.is_main_process and cb_metrics:
                logger.info(
                    "  → " + "  ".join(
                        f"{k}={v:.4f}" for k, v in sorted(cb_metrics.items())
                        if isinstance(v, float)
                    )
                )

    # ── Report ────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        _print_metrics(metrics)
        _save_metrics(
            metrics,
            metrics_path,
            extra={"checkpoint": str(checkpoint), "step": step},
        )

    return metrics


if __name__ == "__main__":
    main()
