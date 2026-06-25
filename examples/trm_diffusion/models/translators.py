"""
models/translators.py — Thinker-to-painter translator modules.

A ThinkerPainterTranslatorBase subclass:
  - takes TRMOutput plus any DataSample fields it declares in ``condition_keys``
  - produces a ThinkerSteering subclass instance

The translator owns the logits-to-spatial conversion (mode: logits/onehot/softmax)
and any trainable conditioning modules (e.g. ConditioningPyramid for ControlNet).

Calling convention (model's run_painter):
    extra = {k: getattr(sample, k) for k in translator.condition_keys}
    steering = self.thinker_painter_translator(trm_output, **extra)
    self.painter(x_noisy, timesteps, **steering.to_painter_kwargs(), **painter_cond).sample
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.interfaces import ControlNetSteering, ThinkerSteering, TRMOutput
from models.utility_models import ConditioningPyramid


class ThinkerPainterTranslatorBase(nn.Module):
    """
    Abstract base for thinker-to-painter translators.

    Args:
        grid:        thinker output grid size (seq_len = grid*grid)
        bridge_mode: how to convert logits to spatial map —
                     "logits" (raw floats), "softmax", or "onehot"
    """

    condition_keys: list[str] = []
    """DataSample fields this translator needs beyond TRMOutput.
    The model passes ``{k: getattr(sample, k) for k in condition_keys}``
    as keyword arguments to forward."""

    def __init__(self, grid: int, bridge_mode: str = "logits"):
        super().__init__()
        self._grid = grid
        self._bridge_mode = bridge_mode

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, C, grid, grid) using self._bridge_mode."""
        B, N, C = logits.shape
        if self._bridge_mode == "onehot":
            soft = logits.float().softmax(dim=-1)
            hard = F.one_hot(logits.argmax(dim=-1), num_classes=C).float()
            onehot = hard - soft.detach() + soft  # straight-through
            return onehot.transpose(1, 2).reshape(B, C, self._grid, self._grid)
        elif self._bridge_mode == "softmax":
            return logits.float().softmax(dim=-1).transpose(1, 2).reshape(B, C, self._grid, self._grid)
        else:
            return logits.float().transpose(1, 2).reshape(B, C, self._grid, self._grid)

    @abstractmethod
    def forward(self, trm_output: TRMOutput, **kwargs) -> ThinkerSteering:
        """
        Args:
            trm_output: output of the thinker
            **kwargs:   DataSample fields listed in ``condition_keys``,
                        passed by the model as keyword arguments.

        Returns:
            ThinkerSteering subclass instance
        """
        pass


class ControlNetTranslator(ThinkerPainterTranslatorBase):
    """
    Translates thinker logits into ControlNet residuals via a ConditioningPyramid.

    Steps:
      1. Optional logit_expand (Linear) if thinker_out_channels != vocab_size
      2. _logits_to_spatial: (B, N, C) → (B, C, grid, grid)
      3. Bilinear upsample to painter_size
      4. ConditioningPyramid → ControlNetSteering
    """

    def __init__(
        self,
        in_channels: int,
        painter_channels: tuple[int, ...],
        layers_per_block: int,
        painter_size: int,
        grid: int,
        bridge_mode: str = "logits",
        thinker_out_channels: Optional[int] = None,
    ):
        super().__init__(grid=grid, bridge_mode=bridge_mode)
        self.painter_size = painter_size

        if thinker_out_channels is not None and thinker_out_channels != in_channels:
            self.logit_expand = nn.Linear(in_channels, thinker_out_channels, bias=False)
            ctrl_in = thinker_out_channels
        else:
            self.logit_expand = None
            ctrl_in = in_channels

        self.control_pyramid = ConditioningPyramid(
            in_channels=ctrl_in,
            block_out_channels=painter_channels,
            layers_per_block=layers_per_block,
        )

    def forward(self, trm_output: TRMOutput, **_) -> ControlNetSteering:
        logits = trm_output.logits
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        spatial = self._logits_to_spatial(logits)
        spatial = F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)
        down_res, mid_res = self.control_pyramid(spatial)
        return ControlNetSteering(
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )
