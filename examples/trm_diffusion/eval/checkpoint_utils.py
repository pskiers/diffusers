"""
Shared checkpoint loading for eval scripts. Convenient for imports
"""

from __future__ import annotations

import logging

import torch

from models.utility_models import strip_compiled_prefix


logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
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
