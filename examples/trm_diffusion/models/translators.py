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

from models.interfaces import ControlNetSteering, CrossAttnSteering, IPAdapterSteering, ThinkerSteering, TRMOutput
from models.utility_models import ConditioningPyramid, ConditioningPyramid1D, TimestepMLP


class ThinkerPainterTranslatorBase(nn.Module):
    """
    Abstract base for thinker-to-painter translators.

    Args:
        grid:        thinker output grid size (seq_len = grid*grid)
        bridge_mode: how to convert logits to spatial map —
                     "logits" (raw floats), "softmax", "onehot", or "normalized"
                     (per-channel BatchNorm1d; for continuous/latent thinker
                     outputs that aren't actually class logits)
        bridge_channels: number of channels the spatial map will have; required
                     when bridge_mode="normalized" (to size the BatchNorm1d).
    """

    condition_keys: list[str] = []
    """DataSample fields this translator needs beyond TRMOutput.
    The model passes ``{k: getattr(sample, k) for k in condition_keys}``
    as keyword arguments to forward."""

    def __init__(self, grid: int, bridge_mode: str = "logits", bridge_channels: Optional[int] = None):
        super().__init__()
        self._grid = grid
        self._bridge_mode = bridge_mode
        if bridge_mode == "normalized":
            if bridge_channels is None:
                raise ValueError("bridge_mode='normalized' requires bridge_channels")
            self._norm = nn.BatchNorm1d(bridge_channels)

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
        elif self._bridge_mode == "normalized":
            normed = self._norm(logits.float().transpose(1, 2))  # (B, C, N)
            return normed.reshape(B, C, self._grid, self._grid)
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


class CrossAttnTranslator(ThinkerPainterTranslatorBase):
    """Projects thinker logits to encoder_hidden_states for DiT cross-attention.

    in_dim: should match the TRM's output dim (vocab_size or hidden_size).
    out_dim: must match the painter's cross_attention_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, with_timestep_emb: bool = False):
        super().__init__(grid=1, bridge_mode="logits")
        self.proj = nn.Linear(in_dim, out_dim)
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> CrossAttnSteering:
        states = self.proj(trm_output.logits.float())  # (B, N, out_dim)
        if self.with_timestep_emb and timesteps is not None:
            states = states + self.timestep_mlp(timesteps.float()).unsqueeze(1)
        return CrossAttnSteering(encoder_hidden_states=states)


class IPAdapterTranslator(ThinkerPainterTranslatorBase):
    """Projects thinker logits to IP-adapter tokens for per-block DiT injection.

    in_dim: should match the TRM's output dim (vocab_size or hidden_size).
    out_dim: must match the DiT hidden dim (num_attention_heads * attention_head_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, with_timestep_emb: bool = False):
        super().__init__(grid=1, bridge_mode="logits")
        self.proj = nn.Linear(in_dim, out_dim)
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> IPAdapterSteering:
        ip_states = self.proj(trm_output.logits.float())  # (B, N, out_dim)
        if self.with_timestep_emb and timesteps is not None:
            ip_states = ip_states + self.timestep_mlp(timesteps.float()).unsqueeze(1)
        return IPAdapterSteering(ip_hidden_states=ip_states)


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
        seq_len: Optional[int] = None,
        with_timestep_emb: bool = False,
    ):
        if thinker_out_channels is not None and thinker_out_channels != in_channels:
            ctrl_in = thinker_out_channels
        else:
            ctrl_in = in_channels

        super().__init__(
            grid=grid,
            bridge_mode=bridge_mode,
            bridge_channels=ctrl_in if bridge_mode == "normalized" else None,
        )
        self.painter_size = painter_size

        # Project sequence length to grid² when they don't match (e.g. CLEVR
        # max_objects=10 → 4×4, or V1 seq_len=74 → 8×8). None = identity (MNIST).
        self.seq_proj = (
            nn.Linear(seq_len, grid * grid, bias=False)
            if seq_len is not None and seq_len != grid * grid
            else None
        )

        self.logit_expand = (
            nn.Linear(in_channels, thinker_out_channels, bias=False)
            if thinker_out_channels is not None and thinker_out_channels != in_channels
            else None
        )

        self.control_pyramid = ConditioningPyramid(
            in_channels=ctrl_in,
            block_out_channels=painter_channels,
            layers_per_block=layers_per_block,
        )
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=painter_channels[-1])

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> ControlNetSteering:
        logits = trm_output.logits
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        if self.seq_proj is not None:
            # (B, N, C) → (B, C, N) → Linear(N→grid²) → (B, C, grid²) → (B, grid², C)
            logits = self.seq_proj(logits.float().transpose(1, 2)).transpose(1, 2)
        spatial = self._logits_to_spatial(logits)
        spatial = F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)
        down_res, mid_res = self.control_pyramid(spatial)
        if self.with_timestep_emb and timesteps is not None:
            mid_res = mid_res + self.timestep_mlp(timesteps).unsqueeze(-1).unsqueeze(-1)
        return ControlNetSteering(
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )


class ControlNetTranslator1D(ThinkerPainterTranslatorBase):
    """
    Translates thinker logits into 1D ControlNet residuals for
    ControlPainterUNet1D (models/action_backbones.py) via
    ConditioningPyramid1D — the 1D analog of ControlNetTranslator, for any
    sequence-diffusion backbone conditioned via ConditionalUnet1D's skip
    connections.

    Steps:
      1. Optional logit_expand (Linear) if thinker_out_channels != vocab_size
      2. Optional seq_proj if thinker seq_len != painter_length
      3. (B, N, C) -> (B, C, N): no spatial grid reshape needed, already 1D
      4. ConditioningPyramid1D -> ControlNetSteering
    """

    def __init__(
        self,
        in_channels: int,
        painter_channels: tuple[int, ...],
        painter_length: int,
        bridge_mode: str = "logits",
        thinker_out_channels: Optional[int] = None,
        seq_len: Optional[int] = None,
        with_timestep_emb: bool = False,
    ):
        if thinker_out_channels is not None and thinker_out_channels != in_channels:
            ctrl_in = thinker_out_channels
        else:
            ctrl_in = in_channels

        super().__init__(
            grid=1,  # unused: this class reshapes to (B, C, painter_length), not a spatial grid
            bridge_mode=bridge_mode,
            bridge_channels=ctrl_in if bridge_mode == "normalized" else None,
        )
        self.painter_length = painter_length

        # Project sequence length to painter_length when they don't match. None = identity.
        self.seq_proj = (
            nn.Linear(seq_len, painter_length, bias=False)
            if seq_len is not None and seq_len != painter_length
            else None
        )

        self.logit_expand = (
            nn.Linear(in_channels, thinker_out_channels, bias=False)
            if thinker_out_channels is not None and thinker_out_channels != in_channels
            else None
        )

        self.control_pyramid = ConditioningPyramid1D(
            in_channels=ctrl_in,
            block_out_channels=painter_channels,
        )
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=painter_channels[-1])

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def _logits_to_1d(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, N) using self._bridge_mode (no spatial grid)."""
        if self._bridge_mode == "onehot":
            B, N, C = logits.shape
            soft = logits.float().softmax(dim=-1)
            hard = F.one_hot(logits.argmax(dim=-1), num_classes=C).float()
            onehot = hard - soft.detach() + soft  # straight-through
            return onehot.transpose(1, 2)
        elif self._bridge_mode == "softmax":
            return logits.float().softmax(dim=-1).transpose(1, 2)
        elif self._bridge_mode == "normalized":
            return self._norm(logits.float().transpose(1, 2))  # (B, C, N)
        else:
            return logits.float().transpose(1, 2)

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> ControlNetSteering:
        logits = trm_output.logits
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        if self.seq_proj is not None:
            # (B, N, C) → (B, C, N) → Linear(N→painter_length) → (B, C, painter_length) → (B, painter_length, C)
            logits = self.seq_proj(logits.float().transpose(1, 2)).transpose(1, 2)
        spatial = self._logits_to_1d(logits)  # (B, C, painter_length)
        down_res, mid_res = self.control_pyramid(spatial)
        if self.with_timestep_emb and timesteps is not None:
            mid_res = mid_res + self.timestep_mlp(timesteps).unsqueeze(-1)
        return ControlNetSteering(
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )
