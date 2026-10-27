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

from models.interfaces import (
    AttentionKVSteering,
    BlockResidualSteering,
    ControlNetSteering,
    CrossAttnSteering,
    IPAdapterSteering,
    ThinkerSteering,
    TRMOutput,
)
from models.utility_models import ConditioningPyramid, ConditioningPyramid1D, ConditioningPyramidPaperUNet, TimestepMLP
from third_party.minit2i import mmdit


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

    def _bridge_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, N, C) using self._bridge_mode."""
        if self._bridge_mode == "onehot":
            soft = logits.float().softmax(dim=-1)
            hard = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()
            return hard - soft.detach() + soft  # straight-through
        elif self._bridge_mode == "softmax":
            return logits.float().softmax(dim=-1)
        elif self._bridge_mode == "normalized":
            return self._norm(logits.float().transpose(1, 2)).transpose(1, 2)
        else:
            return logits.float()

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, C, grid, grid) using self._bridge_mode."""
        B, N, C = logits.shape
        return self._bridge_tokens(logits).transpose(1, 2).reshape(B, C, self._grid, self._grid)

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


class ControlNetTranslatorPaperUNet(ControlNetTranslator):
    """ControlNetTranslator variant for ControlPainterPaperUNet (models/paper_unet.py)
    — same steps as ControlNetTranslator, just backed by a
    ConditioningPyramidPaperUNet (one residual per level, no layers_per_block
    concept) instead of ConditioningPyramid. Only __init__ differs; forward()
    is inherited unchanged since it only calls self.control_pyramid(spatial)
    generically.
    """

    def __init__(
        self,
        in_channels: int,
        painter_channels: tuple[int, ...],
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

        ThinkerPainterTranslatorBase.__init__(
            self,
            grid=grid,
            bridge_mode=bridge_mode,
            bridge_channels=ctrl_in if bridge_mode == "normalized" else None,
        )
        self.painter_size = painter_size

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

        self.control_pyramid = ConditioningPyramidPaperUNet(
            in_channels=ctrl_in,
            block_out_channels=painter_channels,
        )
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=painter_channels[-1])


class SlicedControlNetTranslator(ControlNetTranslator):
    """
    ControlNetTranslator variant that reads only a contiguous slice of the
    TRM's output sequence — e.g. the token range produced by
    NoisySpatialConditionEncoderV2 (models/condition_encoders.py) that
    corresponds to the x_noisy-derived tokens, positionally registered with
    the real output image, rather than the whole sequence (which also
    includes the puzzle/reasoning tokens the TRM used to think but that
    carry no output-space registration of their own).

    slice_start/slice_len select the token range; everything else
    (bridge_mode, seq_proj, ConditioningPyramid) is inherited unchanged from
    ControlNetTranslator and operates on the sliced logits only.
    """

    def __init__(self, *args, slice_start: int, slice_len: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_start = slice_start
        self.slice_len = slice_len

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> ControlNetSteering:
        sliced = trm_output.logits[:, self.slice_start : self.slice_start + self.slice_len]
        return super().forward(TRMOutput(logits=sliced), timesteps=timesteps)


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


class PainterResidualTranslator(ThinkerPainterTranslatorBase):
    """
    Thinker output → one residual per steered transformer-painter block
    (BlockResidualSteering, for MMDiTPainter / MiniT2IPainter).

    Reads the first grid*grid output tokens — the image grid, see
    PainterHiddenEncoder — upsamples them bilinearly to the painter's own token
    grid, and projects them to the painter width with one bias-free,
    zero-initialized Linear per steered block. So training starts exactly from
    the unsteered painter, and (with bridge_mode "logits") an all-zero thinker
    output (null steering, CFG dropout of the steering) gives exactly zero
    residuals.

    bridge_mode: applied to every output token first (see
    ThinkerPainterTranslatorBase). "normalized" (BatchNorm over channels, as the
    ControlNet translators use) makes the steering independent of the thinker
    output's scale — e.g. of weight decay shrinking the thinker's output layer
    while the zero-initialized projections are still growing. All-zero thinker
    output then no longer maps to zero residuals.

    steer_cond: also steer the painter's condition stream from the output tokens
    after the image grid (one per condition token, as PainterHiddenEncoder lays
    them out), with num_blocks - 1 more such projections — a condition residual
    after the last steered block could not reach the output.

    hidden_dim: instead of one Linear per block, the IPAdapterTranslator +
    _DirectIPAdapterDiTBlock construction (models/painter_base.py): a shared
    Linear(in_dim, hidden_dim) with ordinary init, then per block a bias-free
    Linear(hidden_dim, hidden_dim) ("to_v", ordinary init) and a bias-free,
    zero-initialized Linear(hidden_dim, painter_dim) ("to_out").

    with_timestep_emb: add a TimestepMLP embedding of timesteps * timestep_scale
    to every token before the per-block projections (after the shared Linear
    with hidden_dim, as IPAdapterTranslator), so the steering can depend on the
    noise level. timestep_scale 1000 maps flow-matching t in [0, 1] to the
    DDPM-step range the sinusoidal embedding was built for.
    """

    def __init__(
        self,
        in_dim: int,
        painter_dim: int,
        grid: int,
        painter_grid: int,
        num_blocks: int,
        steer_cond: bool = False,
        bridge_mode: str = "logits",
        hidden_dim: Optional[int] = None,
        with_timestep_emb: bool = False,
        timestep_scale: float = 1000.0,
    ):
        super().__init__(grid=grid, bridge_mode=bridge_mode, bridge_channels=in_dim)
        self.painter_grid = painter_grid
        self.in_proj = nn.Linear(in_dim, hidden_dim) if hidden_dim is not None else None
        feat = hidden_dim if hidden_dim is not None else in_dim

        def head() -> nn.Module:
            if hidden_dim is None:
                out = nn.Linear(feat, painter_dim, bias=False)
                nn.init.zeros_(out.weight)
                return out
            to_v, to_out = nn.Linear(feat, feat, bias=False), nn.Linear(feat, painter_dim, bias=False)
            nn.init.zeros_(to_out.weight)
            return nn.Sequential(to_v, to_out)

        self.projs = nn.ModuleList(head() for _ in range(num_blocks))
        self.cond_projs = nn.ModuleList(head() for _ in range(num_blocks - 1)) if steer_cond else None
        self.with_timestep_emb = with_timestep_emb
        self.timestep_scale = timestep_scale
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=feat)

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> BlockResidualSteering:
        g = self._grid
        tokens = self._bridge_tokens(trm_output.logits)
        if self.in_proj is not None:
            tokens = self.in_proj(tokens)
        if self.with_timestep_emb:
            tokens = tokens + self.timestep_mlp(timesteps.float() * self.timestep_scale).unsqueeze(1)
        x = tokens[:, : g * g]
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, g, g)
        if self.painter_grid != g:
            x = F.interpolate(x, size=self.painter_grid, mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2)  # (B, painter_grid², C)
        cond_residuals = None
        if self.cond_projs is not None:
            cond = tokens[:, g * g :]
            cond_residuals = [proj(cond) for proj in self.cond_projs]
        return BlockResidualSteering(
            block_residuals=[proj(x) for proj in self.projs], cond_block_residuals=cond_residuals
        )


class PainterAttentionTranslator(ThinkerPainterTranslatorBase):
    """
    Thinker output → per steered MMDiTPainter block, key/value tokens its image
    tokens attend to (AttentionKVSteering): steering through attention instead
    of residuals added to the token stream, and outside the painter's rotary
    embeddings — its image queries are taken QK-normed but not rotated, and the
    keys carry no rotation (see _FusedDoubleStreamDiTBlock._img_attn).

    Reads the first grid*grid output tokens (the image grid), applies the
    bridge, a shared Linear(in_dim, hidden_dim), optionally a timestep
    embedding (as PainterResidualTranslator), upsamples them bilinearly to the
    painter's token grid, then adds fixed 2D sin/cos positions of that grid, so
    image queries — which carry the painter's own sin/cos positions — can find
    their patch's tokens. Per block: a bias-free
    Linear to the keys (ordinary init) and one to the values (zero init), both
    hidden_dim → num_heads * head_dim; the painter's own frozen attention output
    projection then maps the attended values to its width. Zero values make the
    steering an exact no-op at initialization.

    rope: instead, the steering attention uses the painter's rotated image
    queries and rotates the keys with the same 2D RoPE at their patch positions,
    so it attends by relative position, as the painter's own attention does.
    """

    def __init__(
        self,
        in_dim: int,
        grid: int,
        painter_grid: int,
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int = 384,
        bridge_mode: str = "logits",
        with_timestep_emb: bool = False,
        timestep_scale: float = 1000.0,
        rope: bool = False,
    ):
        super().__init__(grid=grid, bridge_mode=bridge_mode, bridge_channels=in_dim)
        self.rope = rope
        if hidden_dim % 4:
            raise ValueError(f"hidden_dim must be a multiple of 4 (2D sin/cos positions), got {hidden_dim}")
        self.painter_grid = painter_grid
        self.num_heads, self.head_dim = num_heads, head_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        pos = mmdit.get_2d_sincos_pos_embed(hidden_dim, painter_grid, torch.device("cpu"), torch.float32)
        self.register_buffer("pos", pos, persistent=False)
        self.to_k = nn.ModuleList(nn.Linear(hidden_dim, num_heads * head_dim, bias=False) for _ in range(num_blocks))
        self.to_v = nn.ModuleList(nn.Linear(hidden_dim, num_heads * head_dim, bias=False) for _ in range(num_blocks))
        for v in self.to_v:
            nn.init.zeros_(v.weight)
        self.with_timestep_emb = with_timestep_emb
        self.timestep_scale = timestep_scale
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=hidden_dim)

    @property
    def condition_keys(self) -> list[str]:
        return ["timesteps"] if self.with_timestep_emb else []

    def forward(self, trm_output: TRMOutput, timesteps=None, **_) -> AttentionKVSteering:
        g = self._grid
        x = self.in_proj(self._bridge_tokens(trm_output.logits)[:, : g * g])
        if self.with_timestep_emb:
            x = x + self.timestep_mlp(timesteps.float() * self.timestep_scale).unsqueeze(1)
        b, _, c = x.shape
        if self.painter_grid != g:
            x = x.transpose(1, 2).reshape(b, c, g, g)
            x = F.interpolate(x, size=self.painter_grid, mode="bilinear", align_corners=False)
            x = x.flatten(2).transpose(1, 2)
        x = x + self.pos[None]
        m = x.shape[1]
        heads = lambda t: t.view(b, m, self.num_heads, self.head_dim)  # noqa: E731
        return AttentionKVSteering(
            block_keys=[heads(k(x)) for k in self.to_k], block_values=[heads(v(x)) for v in self.to_v], rope=self.rope
        )
