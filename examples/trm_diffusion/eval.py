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
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Load weights from a checkpoint written by train_trm.py.

    Format: {"step": int, "model_state": ..., "ema_state": {"shadow": ...}, ...}

    Falls back to loading the file directly as a state_dict if no known keys
    are found (plain torch.save(model.state_dict(), path)).

    Returns the training step recorded in the checkpoint, or None.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        step = ckpt.get("step", None)

        # Always load model_state first — covers frozen params and buffers
        # (e.g. H_init, L_init) that EMA doesn't track.
        sd = strip_compiled_prefix(ckpt["model_state"])
        model.load_state_dict(sd, strict=False)

        if use_ema and ckpt.get("ema_state") is not None:
            # EMAHelper.state_dict() returns self.shadow directly:
            # {param_name: tensor} — no extra nesting.
            ema_state = ckpt["ema_state"]
            if isinstance(ema_state, dict) and ema_state:
                ema_sd = strip_compiled_prefix(ema_state)
                missing, unexpected = model.load_state_dict(ema_sd, strict=False)
                logger.info(
                    f"Loaded EMA weights on top of model_state "
                    f"({len(ema_sd)} EMA params, missing={len(missing)})"
                )
                if missing:
                    logger.info(f"  Missing (first 5): {missing[:5]}")
                return step
            logger.warning("EMA state is empty — using raw model_state")
            logger.info(f"Loaded model_state (step={step})")
        else:
            logger.info(f"Loaded model_state (step={step}, use_ema=False)")
        return step

    # Fallback: raw state_dict
    sd = strip_compiled_prefix(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded raw state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
    return None


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
    _, eval_ds = build_datasets(cfg)
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

    metrics: dict[str, float] = {}
    with torch.no_grad():
        for cb in callbacks:
            logger.info(f"Running {type(cb).__name__} …")
            cb_metrics = cb(unwrapped, eval_dl, accelerator, step=step)
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
