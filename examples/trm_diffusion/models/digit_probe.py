"""models/digit_probe.py — small standalone post-hoc probe for reading a
digit prediction directly off a TRM's z_H state.

Trained by experiments/train_digit_probe.py against the model's OWN final
generated (and classified) output as the label, never the puzzle's ground-
truth solution — see that script's module docstring for why. Deliberately
kept as a tiny, separate module (not attached to SpatialTRM /
TRMDiffusionBackbone / ThinkerFrozenPainterBase) so training and using it
needs no changes to those classes or to train_trm.py — it lives entirely
in its own small checkpoint file.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DigitProbe(nn.Module):
    """Linear map from a TRM's per-cell z_H to a 9-way digit distribution
    (raw digit values 0-8, matching DataSample.solution's convention).

    Deliberately linear, not an MLP: a probe with real hidden-layer
    capacity can learn to guess plausible digits from structural leakage
    in z_H (e.g. attention smearing nearby given-digit info into a cell's
    token) independent of whatever the TRM has actually reasoned out for
    that specific cell — see experiments/train_digit_probe.py's
    control-task floor check, which this constraint is meant to support.
    """

    def __init__(self, hidden_size: int, num_classes: int = 9):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """z_H: (..., hidden_size) -> (..., num_classes) logits."""
        return self.linear(z_H.float())
