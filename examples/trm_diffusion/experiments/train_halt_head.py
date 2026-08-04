"""
experiments/train_halt_head.py — Train SpatialTRM's adaptive-halting head
(models.trm_wrappers.SpatialTRM.with_halt_head) on top of an already-trained
thinker+painter checkpoint, without touching anything else.

Only the halt_head's own parameters (and its own dedicated optimizer, set up
in SpatialTRM.__init__) are ever updated here — the rest of the checkpoint's
weights are loaded and never written to. This intentionally sidesteps
train_trm.py's resume path (global_step/num_steps bookkeeping, EMA shadow
dict, the main thinker optimizer's state) entirely: none of that is needed
just to add this one small head, and reusing it would mean teaching those
code paths to tolerate a checkpoint whose parameter set doesn't match them
(e.g. EMAHelper's shadow dict missing the new halt_head keys).

For every training example, this collects the real per-step diffusion loss
across the checkpoint's own full n_sup reasoning trajectory (model.eval(),
no_grad — cheap, no gradient needed through the frozen model) and trains
halt_head to predict, from z_H at step t, the future loss reduction still
available: loss_t - min(loss_{t+1..}). See SpatialTRM.train_halt_head.

Usage:
    python experiments/train_halt_head.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      thinker.with_halt_head=true \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80_repro/checkpoint_final.pt \\
      +halt_train.num_steps=500

    # Options (all under +halt_train.*):
    #   num_steps   — training iterations (default 500)
    #   batch_size  — default cfg.train.batch_size
    #   out         — output checkpoint path (default: alongside the input
    #                 checkpoint, named checkpoint_with_halt_head.pt)
    # use_ema=false to continue from raw (non-EMA) weights instead of the
    # checkpoint's EMA weights (default true, matching eval.py/
    # ablate_trm_loop_budget.py's own default).

Config overrides work exactly like train_trm.py / eval.py.
"""

from __future__ import annotations

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
from tqdm.auto import tqdm

from factory import build_datasets, build_model
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Duplicated from eval.py / ablate_trm_loop_budget.py (see the latter's
    own copy's docstring for why this isn't imported instead). Keep in sync
    with eval.py's version if the checkpoint format changes."""
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
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/train_halt_head.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> thinker.with_halt_head=true [+halt_train.xxx=...]"
        )

    ht = cfg.get("halt_train", {})
    num_steps: int = ht.get("num_steps", 500)
    batch_size: int = ht.get("batch_size", cfg.train.batch_size)
    log_every: int = ht.get("log_every", 50)
    use_ema: bool = cfg.get("use_ema", True)
    out_path: str = ht.get("out", str(Path(checkpoint).parent / "checkpoint_with_halt_head.pt"))

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
    model.eval()  # matches how the checkpoint is actually used at real inference/eval

    train_ds, _ = build_datasets(cfg)
    collate_fn = getattr(type(train_ds), "collate_fn", None)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    logger.info(
        f"Training halt head for {num_steps} steps (n_sup={model.n_sup} per step) — "
        "everything else in the checkpoint is loaded and never updated."
    )

    train_iter = iter(train_dl)

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            return next(train_iter)

    recent_losses: list[float] = []
    for step in tqdm(range(num_steps), desc="Training halt head"):
        batch = _next_batch()
        [d] = model.prep_mb_data([batch], device)

        z_H0_list, loss_list = [], []
        with torch.no_grad():
            for _ in range(model.n_sup):
                noise_pred, logits, d["z_H"], d["z_L"] = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
                per_sample_loss = (noise_pred.float() - d["sample"].target.float()).pow(2).flatten(1).mean(1)
                z_H0_list.append(d["z_H"][:, 0].detach())
                loss_list.append(per_sample_loss.detach())

        halt_loss = model.thinker.train_halt_head(z_H0_list, loss_list)
        recent_losses.append(halt_loss)
        if step % log_every == 0:
            window = recent_losses[-log_every:]
            logger.info(f"step={step}  halt_head_loss={halt_loss:.4f}  (last {len(window)} avg={sum(window)/len(window):.4f})")

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": None, "model_state": accelerator.unwrap_model(model).state_dict()}, out_path)
        tail = recent_losses[-log_every:]
        logger.info(f"Saved → {out_path} (last {len(tail)}-step avg halt_head_loss={sum(tail)/len(tail):.4f})")


if __name__ == "__main__":
    main()
