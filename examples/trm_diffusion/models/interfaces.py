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
from dataclasses import dataclass
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
