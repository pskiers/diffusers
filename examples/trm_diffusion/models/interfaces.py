"""
models/interfaces.py — Typed dataclasses for inter-component communication.

These are the typed "wires" between the major pipeline stages:

    DataSample
        └─► ConditionEncoder  ──► TRMInput
                                      └─► Thinker ──► TRMOutput
                                                          └─► Translator ──► ThinkerSteering
                                                                                  └─► Painter ──► DiffusionPrediction

Each dataclass is intentionally minimal now; fields are added here when a
real use-case demands them, not speculatively.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Literal, Optional

import torch
from torch import Tensor

# ── Thinker I/O ──────────────────────────────────────────────────────────────


@dataclass
class TRMInput:
    """Output of a ConditionEncoder; input to the Thinker.

    Contains only the encoded embedding produced by the condition encoder.
    Any additional inputs the thinker needs (puzzle_ids, timesteps,
    attention_mask, …) are declared via the thinker's own ``condition_keys``
    and routed from the DataSample by the model — they are not part of this
    dataclass.
    """

    enc_emb: Tensor
    """(B, S, H) — condition embedding tokens."""

    def to_painter_kwargs(self, enc_mask: Optional[Tensor] = None) -> dict:
        """Convert to DiT/UNet cross-attention kwargs (mirrors ThinkerSteering)."""
        result: dict = {"encoder_hidden_states": self.enc_emb}
        if enc_mask is not None:
            result["encoder_attention_mask"] = enc_mask
        return result


@dataclass
class TRMOutput:
    """Output of the Thinker; input to the Translator."""

    logits: Tensor
    """(B, N, vocab_size) — per-position class logits over the discrete
    output vocabulary."""


# ── Translator output (ThinkerSteering) ──────────────────────────────────────


@dataclass
class ThinkerSteering:
    """Abstract base for translator outputs.

    A subclass is produced by each Translator variant and consumed by the
    matching Painter variant.  Subclasses implement ``to_painter_kwargs``
    to unpack themselves into the raw keyword arguments the underlying UNet
    forward expects.

    This is a temporary bridge until Painter becomes a custom class that
    accepts ThinkerSteering directly.
    """

    @abstractmethod
    def to_painter_kwargs(self) -> dict:
        """Return a dict of keyword arguments to unpack into the painter's
        forward call."""

    def scaled(self, scale: float) -> "ThinkerSteering":
        """Return a copy with every tensor field scaled by `scale`.

        scale=0.0 fully ablates the steering (pure frozen-painter baseline);
        scale>1.0 amplifies it. Generic across every subclass (ControlNetSteering,
        IPAdapterSteering, CrossAttnSteering, ...) — scales whatever
        tensor/list-of-tensor fields it finds rather than hardcoding
        per-subclass field names, so new subclasses get this for free.
        Used for the steering-ablation/amplification diagnostics in eval.py.
        """
        if scale == 1.0:
            return self
        updates = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Tensor):
                updates[f.name] = val * scale
            elif isinstance(val, list) and val and isinstance(val[0], Tensor):
                updates[f.name] = [v * scale for v in val]
        return replace(self, **updates) if updates else self


@dataclass
class CrossAttnSteering(ThinkerSteering):
    """Steering produced by a translator for CrossAttnSteeredDiTPainter.

    Replaces (or supplements) the condition encoder's encoder_hidden_states
    with translator-produced tokens fed into the DiT's cross-attention.
    """

    encoder_hidden_states: Tensor
    """(B, N, D) — steering tokens injected into DiT cross-attention."""

    encoder_attention_mask: Optional[Tensor] = None
    """(B, N) bool mask — True for real tokens, False for padding."""

    def to_painter_kwargs(self) -> dict:
        result: dict = {"encoder_hidden_states": self.encoder_hidden_states}
        if self.encoder_attention_mask is not None:
            result["encoder_attention_mask"] = self.encoder_attention_mask
        return result


@dataclass
class IPAdapterSteering(ThinkerSteering):
    """Steering produced by a translator for IPAdapterSteeredDiTPainter.

    IP tokens are injected into each DiT transformer block via a separate
    trainable cross-attention layer that sits on top of the frozen block.
    The condition encoder still runs normally; this adds extra conditioning
    rather than replacing it.
    """

    ip_hidden_states: Tensor
    """(B, N, D) — IP tokens for per-block cross-attention injection.
    D must match the DiT hidden dim (num_attention_heads * attention_head_dim)."""

    def to_painter_kwargs(self) -> dict:
        return {"cross_attention_kwargs": {"ip_hidden_states": self.ip_hidden_states}}


@dataclass
class ControlNetSteering(ThinkerSteering):
    """Steering produced by ControlNetTranslator for a ControlNet UNet."""

    down_block_additional_residuals: list
    """Per-layer residuals for the UNet's down blocks."""

    mid_block_additional_residual: Tensor
    """Residual for the UNet's mid block."""

    def to_painter_kwargs(self) -> dict:
        return {
            "down_block_additional_residuals": self.down_block_additional_residuals,
            "mid_block_additional_residual": self.mid_block_additional_residual,
        }


# ── Painter output ────────────────────────────────────────────────────────────

PredictionType = Literal["epsilon", "sample", "v_prediction"]


@dataclass
class DiffusionPrediction:
    """Output of the Painter (and optionally the Thinker)."""

    pred: Tensor
    """(B, C, H, W) — model prediction."""

    pred_type: PredictionType
    """Scheduler prediction type: 'epsilon' (noise), 'sample' (x0), or
    'v_prediction'."""

    logits: Optional[Tensor] = None
    """(B, N, vocab_size) — thinker logits if available; None for pure
    painter models.  Used by eval_step for CE loss and accuracy metrics."""
