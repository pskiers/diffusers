"""
Ratatouille TRM-DiT model variants for MNIST Sudoku.

Architecture overview:
  Painter  – UNet2DModel in pixel space (1-channel grayscale).
             Input: cat([noisy_image, bridge_features], dim=1).
             Conditioning reaches the painter ONLY via spatial bridge features.

  Thinker  – SpatialTRM: small transformer TRM that reasons about the puzzle.
             Operates on a compressed spatial grid (seq_len = grid_h × grid_w tokens).
             States: z_H (slow), z_L (fast) as (B, seq_len, d_model).
             Output: spatial feature map → bridge → painter conditioning.

  Bridge   – upsamples thinker output to painter resolution.
             V0–V3: bilinear upsample + 2-conv (SpatialBridge).
             V4:    cross-attention (AttentiveBridge from models.py).

  Encoder  – SpatialEncoder from models.py compresses input image(s) to thinker grid.

Training interface (n_sup backward passes per batch, identical to SudokuTRM):
    z_H, z_L = model.get_initial_states(B)
    for _ in range(model.n_sup):
        noise_pred, logits, z_H, z_L = model.reasoning_step(cond, noisy, z_H, z_L, ts)
        loss = mse(noise_pred, noise) + sudoku_w * ce(logits, solution)
        backward(loss); optimizer.step(); zero_grad(); global_step += 1

Inference interface:
    noise_pred, logits = model(noisy_images, timesteps, condition)
    Runs full n_sup × H_cycles × L_cycles loops (no training-specific grad split).

Five variants — each removes one training wheel from the previous:

  V0: Encoder sees CONDITION ONLY; 9×9 TRM; TRM output = digit logits.
      Losses: diffusion + weighted sudoku CE.
      Most constrained: clean input, explicit digit supervision, 9-channel output,
      9×9 grid. Closest to vanilla SudokuTRM + DiT painter.

  V1: Encoder sees cat(condition, noisy); 9×9 TRM; digit logits.
      Training wheel removed: TRM must reason from noisy input too.
      Losses: diffusion + weighted sudoku CE.

  V2: Same as V1 but no sudoku CE loss.
      Training wheel removed: no explicit digit-level supervision.

  V3: Same as V2 but thinker_out_channels >> 9.
      Training wheel removed: output not forced into digit-logit space.

  V4: Same as V3 but compression_factor ≠ cell_size; AttentiveBridge decoder.
      Training wheel removed: grid uncoupled from puzzle cell structure.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from models_pt import SpatialEncoder, AttentiveBridge, ConditioningPyramid   # reuse from trm pipeline
from sudoku_models import SudokuTRM


# ── Spatial TRM ──────────────────────────────────────────────────────────────────

class SpatialTRM(SudokuTRM):
    """
    Spatial variant of SudokuTRM — identical internals, spatial I/O.

    Inherits unchanged from SudokuTRM:
      blocks (SwiGLU + post-norm RMSNorm), rotary_emb (RoPE), out_norm,
      lm_head (reused as out_proj), z_H_init / z_L_init (trunc_normal std=1),
      reasoning_step, get_initial_states, count_parameters.

    vocab_size=out_channels is passed to the parent so that lm_head has the
    right output dimension.  The token embedding (nn.Embedding) is replaced
    in-place by a Conv2d spatial projection — same attribute name, different
    type.

    embed() mirrors SudokuTRM's: scale + optional puzzle prefix token (RoPE
    handles positions, no separate 2D learned table needed).
    decode() reshapes the lm_head output back to (B, out_channels, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_h: int,
        grid_w: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        dropout: float = 0.0,
        num_puzzle_ids: int | None = None,
    ):
        seq_len = grid_h * grid_w
        # Parent builds the full TRM stack. vocab_size=out_channels so that
        # lm_head is Linear(d_model, out_channels) — the desired output shape.
        super().__init__(
            vocab_size=out_channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Replace the token embedding with a Conv2d spatial projection.
        # Everything else (rotary_emb, blocks, lm_head, z_H/z_L init) is unchanged.
        #
        # IMPORTANT: embed_scale = sqrt(d_model) is applied in embed() just like
        # SudokuTRM.  For that scaling to produce transformer inputs with std ≈ 1.0,
        # the Conv2d output (for a unit-std input) must have std = 1/embed_scale.
        #   std_out = std_w * sqrt(in_channels) * std_in
        #   => std_w = 1 / (embed_scale * sqrt(in_channels))
        # Kaiming-relu would give std_w = sqrt(2/in_channels), making the final
        # token magnitude ~embed_scale * sqrt(2) ≈ 32×, which swamps z_H + z_L and
        # kills the recurrent mechanism.
        embed_std = 1.0 / self.embed_scale          # = 1/sqrt(d_model)
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=1)
        nn.init.normal_(self.embedding.weight, std=embed_std / math.sqrt(in_channels))
        nn.init.zeros_(self.embedding.bias)

    def embed(self, x: torch.Tensor, puzzle_ids=None) -> torch.Tensor:
        """
        (B, in_channels, grid_h, grid_w) → (B, [1+]seq_len, d_model)

        Mirrors SudokuTRM.embed(): embed_scale * projection + optional puzzle
        prefix token prepended to the sequence (same as SudokuTRM item 6).
        RoPE in _run_blocks handles positions — no separate 2D table needed.
        """
        tokens = self.embedding(x).flatten(2).transpose(1, 2)   # (B, seq, d_model)
        tokens = self.embed_scale * tokens

        if self.puzzle_id_embedding is not None:
            if puzzle_ids is None:
                puzzle_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            puzzle_token = self.puzzle_id_embedding(puzzle_ids).unsqueeze(1)
            tokens = torch.cat([puzzle_token, tokens], dim=1)

        return tokens

    # reasoning_step() is inherited from SudokuTRM unchanged.
    # It returns (lm_head_output, z_H_det, z_L_det) where lm_head_output is
    # (B, seq_len, out_channels) — exactly what decode() below expects.

    def decode(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        (B, seq_len, out_channels) → (B, out_channels, grid_h, grid_w)

        spatial_tokens is reasoning_step()[0]: lm_head has already been applied,
        so this is a pure reshape back to a spatial feature map.
        """
        B = spatial_tokens.shape[0]
        return spatial_tokens.transpose(1, 2).reshape(B, -1, self.grid_h, self.grid_w)


# ── Bilinear bridge (V0–V3) ──────────────────────────────────────────────────────

class SpatialBridge(nn.Module):
    """
    Bilinear upsample + 2 conv layers.
    (B, in_c, H_t, W_t) → (B, bridge_c, painter_size, painter_size)
    Used for V0–V3.  V4 uses AttentiveBridge from models.py instead.
    """

    def __init__(self, in_channels: int, out_channels: int, painter_size: int):
        super().__init__()
        self.painter_size = painter_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.painter_size, mode="bilinear", align_corners=False)
        return self.conv(x)


# ── Painter factory ──────────────────────────────────────────────────────────────

def _make_painter(
    painter_size: int,
    bridge_channels: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
    image_channels: int = 1,
) -> UNet2DModel:
    """
    Build the denoising UNet.  Uses plain conv blocks throughout (no attention)
    to keep compute O(pixels) rather than O(pixels²).  For a 9×9 grid of
    independent digits, self-attention across the full image adds little value
    and dominates wall-clock time.
    """
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return UNet2DModel(
        sample_size=painter_size,
        in_channels=image_channels + bridge_channels,
        out_channels=image_channels,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


# ── ControlNet-capable painter ───────────────────────────────────────────────────

class _ControlPainterUNet(UNet2DModel):
    """UNet2DModel extended with ControlNet-style residual injection.

    Identical to UNet2DModel in every way except forward() accepts
    down_block_additional_residuals and mid_block_additional_residual, which are
    added to the skip connections before the up-blocks (standard ControlNet math).
    """

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        class_labels=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
    ):
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        t_emb = self.time_proj(timestep).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels required for class conditioning")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            emb = emb + self.class_embedding(class_labels).to(dtype=self.dtype)

        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down = ()
            for orig, add in zip(down_block_res_samples, down_block_additional_residuals):
                new_down += (orig + add,)
            down_block_res_samples = new_down

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timestep = timestep.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timestep

        if not return_dict:
            return (sample,)
        return UNet2DOutput(sample=sample)


def _make_painter_control(
    painter_size: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
) -> _ControlPainterUNet:
    """Build a ControlNet-capable painter UNet (in_channels=1, no bridge concat)."""
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return _ControlPainterUNet(
        sample_size=painter_size,
        in_channels=1,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


# ── SPADE (Spatially Adaptive Normalization) painter ─────────────────────────────

class SPADEGroupNorm(nn.Module):
    """GroupNorm + spatially adaptive scale/bias from a semantic map.

    h_out = gamma(s) * GroupNorm(h) + beta(s)
    where gamma, beta are predicted by a small CNN applied to s resized to h's size.
    """

    def __init__(self, num_groups: int, num_channels: int, sem_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        mid = max(num_channels, sem_channels)
        self.shared     = nn.Sequential(nn.Conv2d(sem_channels, mid, 3, padding=1), nn.SiLU())
        self.gamma_proj = nn.Conv2d(mid, num_channels, 3, padding=1)
        self.beta_proj  = nn.Conv2d(mid, num_channels, 3, padding=1)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        s_r    = F.interpolate(s, size=h.shape[-2:], mode="bilinear", align_corners=False)
        feat   = self.shared(s_r)
        return self.gamma_proj(feat) * h_norm + self.beta_proj(feat)


class SPADEResBlock(nn.Module):
    """ResNet block with both GroupNorms replaced by SPADE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sem_channels: int,
        temb_channels: int,
        norm_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = SPADEGroupNorm(norm_groups, in_channels, sem_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = SPADEGroupNorm(norm_groups, out_channels, sem_channels)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.act            = nn.SiLU()
        self.time_emb_proj  = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels))
        self.conv_shortcut  = (nn.Conv2d(in_channels, out_channels, 1)
                               if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, s))
        h = self.conv1(h)
        h = h + self.time_emb_proj(temb)[:, :, None, None]
        h = self.act(self.norm2(h, s))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _SPADEDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, sem_ch, num_layers, add_downsample, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList([
            SPADEResBlock(in_ch if i == 0 else out_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout)
            for i in range(num_layers)
        ])
        self.downsamplers = (
            nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)])
            if add_downsample else None
        )

    def forward(self, hidden, temb, s):
        outputs = ()
        for r in self.resnets:
            hidden = r(hidden, temb, s);  outputs += (hidden,)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                hidden = d(hidden)
            outputs += (hidden,)
        return hidden, outputs


class _SPADEMidBlock(nn.Module):
    def __init__(self, channels, temb_ch, sem_ch, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList([
            SPADEResBlock(channels, channels, sem_ch, temb_ch, norm_groups, dropout)
            for _ in range(2)   # UNetMidBlock2D default: num_layers=1 → 2 resnets
        ])

    def forward(self, hidden, temb, s):
        for r in self.resnets:
            hidden = r(hidden, temb, s)
        return hidden


class _SPADEUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, prev_out_ch, temb_ch, sem_ch, num_layers, add_upsample, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            # Mirrors diffusers UpBlock2D channel formula exactly
            res_skip_ch = in_ch if (i == num_layers - 1) else out_ch
            res_in_ch   = prev_out_ch if i == 0 else out_ch
            self.resnets.append(
                SPADEResBlock(res_in_ch + res_skip_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout)
            )
        self.upsamplers = (
            nn.ModuleList([nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )])
            if add_upsample else None
        )

    def forward(self, hidden, res_tuple, temb, s):
        for r in self.resnets:
            skip       = res_tuple[-1]
            res_tuple  = res_tuple[:-1]
            hidden     = r(torch.cat([hidden, skip], dim=1), temb, s)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                hidden = up(hidden)
        return hidden


class SPADEUNet2D(nn.Module):
    """UNet2DModel with SPADE normalization throughout.

    Takes an extra semantic map `s` (B, sem_channels, H, W) in forward().
    Each SPADEGroupNorm bilinearly resizes `s` to the current feature map
    resolution — no pyramid required.

    Architecturally matches _make_painter_control (in_channels=1, no concat).
    """

    def __init__(
        self,
        painter_size: int,
        sem_channels: int,
        block_out_channels: tuple[int, ...] = (32, 64, 128, 256),
        layers_per_block: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        norm_groups = 32
        while norm_groups > 1 and any(c % norm_groups != 0 for c in block_out_channels):
            norm_groups //= 2

        ch0    = block_out_channels[0]
        temb_ch = ch0 * 4

        self.time_proj      = Timesteps(ch0, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(ch0, temb_ch)
        self.conv_in        = nn.Conv2d(1, ch0, 3, padding=1)

        # Down blocks
        n = len(block_out_channels)
        self.down_blocks = nn.ModuleList()
        cur_ch = ch0
        for i, out_ch in enumerate(block_out_channels):
            self.down_blocks.append(_SPADEDownBlock(
                in_ch=cur_ch, out_ch=out_ch, temb_ch=temb_ch, sem_ch=sem_channels,
                num_layers=layers_per_block, add_downsample=(i < n - 1),
                norm_groups=norm_groups, dropout=dropout,
            ))
            cur_ch = out_ch

        # Mid block
        self.mid_block = _SPADEMidBlock(
            channels=cur_ch, temb_ch=temb_ch, sem_ch=sem_channels,
            norm_groups=norm_groups, dropout=dropout,
        )

        # Up blocks (mirrors UNet2DModel channel formula)
        rev = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        prev_out_ch = rev[0]
        for i, out_ch in enumerate(rev):
            in_skip_ch = rev[min(i + 1, n - 1)]
            self.up_blocks.append(_SPADEUpBlock(
                in_ch=in_skip_ch, out_ch=out_ch, prev_out_ch=prev_out_ch,
                temb_ch=temb_ch, sem_ch=sem_channels,
                num_layers=layers_per_block + 1, add_upsample=(i < n - 1),
                norm_groups=norm_groups, dropout=dropout,
            ))
            prev_out_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(norm_groups, ch0)
        self.conv_act      = nn.SiLU()
        self.conv_out      = nn.Conv2d(ch0, 1, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep, s: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        emb = self.time_embedding(self.time_proj(timestep).to(sample.dtype))

        x = self.conv_in(sample)
        skips = (x,)
        for block in self.down_blocks:
            x, res = block(x, emb, s);  skips += res

        x = self.mid_block(x, emb, s)

        for block in self.up_blocks:
            n_skip   = len(block.resnets)
            res_tup  = skips[-n_skip:]
            skips    = skips[:-n_skip]
            x = block(x, res_tup, emb, s)

        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


# ── Gradient scaling hook ────────────────────────────────────────────────────────

class _GradScale(torch.autograd.Function):
    """Pass-through in forward; multiply gradient by `scale` in backward."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale (or zero) the gradient that flows backward through x."""
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return _GradScale.apply(x, scale)


# ── TRM Ratatouille base ─────────────────────────────────────────────────────────

class _TRMRatatouilleBase(nn.Module):
    """
    Base class for TRM-based Ratatouille models.

    Sub-classes set:
      _encoder_sees_noisy (bool) – encoder gets condition only (V0) or
                                   cat(condition, noisy) (V1–V4).

    Sub-classes override:
      _compute_sudoku_logits(spatial_cond) → tensor | None

    Training: call reasoning_step in a loop (n_sup times per batch).
    Inference: call forward() — runs the full n_sup × H_cycles × L_cycles loop.

    diff_thinker_weight controls how much diffusion-loss gradient flows back
    into the thinker through the bridge path:
      0.0 → thinker trained only on sudoku_loss (bridge sees detached spatial_cond)
      1.0 → full diffusion gradient reaches thinker (default)
    The sudoku_loss gradient is always unscaled regardless of this value.
    """

    _encoder_sees_noisy: bool = True

    def __init__(
        self,
        encoder: SpatialEncoder,
        thinker: SpatialTRM,
        bridge: Optional[nn.Module],   # None for ControlNet variants
        painter: nn.Module,
        diff_thinker_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.thinker = thinker
        self.bridge  = bridge          # not registered as submodule when None
        self.painter = painter
        self.diff_thinker_weight = diff_thinker_weight

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond_scaled: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Inject thinker features into painter. Override for ControlNet."""
        bridge_feat = self.bridge(spatial_cond_scaled)
        return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample

    @property
    def n_sup(self) -> int:
        return self.thinker.n_sup

    def get_initial_states(self, bsz: int):
        return self.thinker.get_initial_states(bsz)

    def _get_encoder_input(
        self, condition: torch.Tensor, noisy: torch.Tensor
    ) -> torch.Tensor:
        if self._encoder_sees_noisy:
            return torch.cat([condition, noisy], dim=1)
        return condition

    def _compute_sudoku_logits(
        self, spatial_cond: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def reasoning_step(
        self,
        condition: torch.Tensor,   # (B, 1, H, W)
        noisy: torch.Tensor,       # (B, 1, H, W)
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        One n_sup supervision step.

        Encoder and thinker.embed are re-run on every call so each backward()
        gets a fresh computation graph (their parameters are nn.Parameters —
        saved tensors are freed after backward()).

        Returns: (noise_pred, sudoku_logits, z_H_next, z_L_next)
        """
        enc       = self.encoder(self._get_encoder_input(condition, noisy))
        input_emb = self.thinker.embed(enc, puzzle_ids=puzzle_ids)
        spatial_tokens, z_H_next, z_L_next = self.thinker.reasoning_step(input_emb, z_H, z_L)

        spatial_cond  = self.thinker.decode(spatial_tokens)
        # Scale (or zero) the diffusion-loss gradient flowing back into the thinker.
        # sudoku_logits uses the unscaled spatial_cond so its gradient is unaffected.
        noise_pred    = self._run_painter(
            noisy, _grad_scale(spatial_cond, self.diff_thinker_weight), timesteps
        )
        sudoku_logits = self._compute_sudoku_logits(spatial_cond)
        return noise_pred, sudoku_logits, z_H_next, z_L_next

    def forward(
        self,
        noisy_images: torch.Tensor,   # (B, 1, H, W)
        timesteps: torch.Tensor,      # (B,)
        condition: torch.Tensor,      # (B, 1, H, W)
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        Full inference: n_sup × H_cycles × L_cycles, single painter pass at end.
        No training-specific grad split — all cycles run uniformly (like predict()
        in SudokuTRM).  Encoder run once (condition+noisy fixed during inference).

        Returns: (noise_pred, sudoku_logits)
        """
        B    = noisy_images.shape[0]
        z_H, z_L = self.get_initial_states(B)
        z_H  = z_H.to(noisy_images.device)
        z_L  = z_L.to(noisy_images.device)

        enc  = self.encoder(self._get_encoder_input(condition, noisy_images))

        for _ in range(self.thinker.n_sup):
            input_emb = self.thinker.embed(enc, puzzle_ids=puzzle_ids)
            T         = input_emb.shape[1]
            freqs_cis = self.thinker.rotary_emb(T)
            for _ in range(self.thinker.H_cycles):
                for _ in range(self.thinker.L_cycles):
                    z_L = self.thinker._run_blocks(input_emb + z_H + z_L, freqs_cis)
                z_H = self.thinker._run_blocks(z_H + z_L, freqs_cis)

        # Produce spatial tokens the same way reasoning_step() does.
        spatial_tokens = self.thinker.lm_head(
            self.thinker.out_norm(z_H[:, self.thinker.puzzle_emb_len:])
        )
        spatial_cond  = self.thinker.decode(spatial_tokens)
        noise_pred    = self._run_painter(noisy_images, spatial_cond, timesteps)
        sudoku_logits = self._compute_sudoku_logits(spatial_cond)
        return noise_pred, sudoku_logits


# ── ControlNet base ──────────────────────────────────────────────────────────────

class _TRMRatatouilleControlBase(_TRMRatatouilleBase):
    """
    Same as _TRMRatatouilleBase but painter conditioning uses ControlNet-style
    multi-scale residual injection instead of channel concatenation.

    Thinker spatial output is bilinearly upsampled to painter resolution, then
    processed by ConditioningPyramid which produces per-layer residuals that are
    added to the UNet's skip connections and mid block.

    The painter is a _ControlPainterUNet with in_channels=1 (noisy only —
    no bridge channels concatenated).
    """

    def __init__(
        self,
        encoder: SpatialEncoder,
        thinker: SpatialTRM,
        painter: _ControlPainterUNet,
        control_pyramid: ConditioningPyramid,
        diff_thinker_weight: float = 1.0,
    ):
        # bridge=None: this class overrides _run_painter, self.bridge is never called
        super().__init__(
            encoder=encoder, thinker=thinker,
            bridge=None, painter=painter,
            diff_thinker_weight=diff_thinker_weight,
        )
        self.control_pyramid = control_pyramid

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond_scaled: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        painter_size = noisy.shape[-1]
        upsampled = F.interpolate(
            spatial_cond_scaled, size=painter_size, mode="bilinear", align_corners=False
        )
        down_res, mid_res = self.control_pyramid(upsampled)
        return self.painter(
            noisy, timesteps,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample


# ── SPADE base ───────────────────────────────────────────────────────────────────

class _TRMRatatouilleSPADEBase(_TRMRatatouilleBase):
    """SPADE-conditioned variant: thinker output spatially normalizes every UNet ResBlock.

    Unlike ControlNet (additive residuals), SPADE multiplies and shifts feature
    map activations at every norm layer — stronger, spatially precise conditioning.
    The painter is a SPADEUNet2D; no bridge module is used.
    """

    def __init__(
        self,
        encoder: SpatialEncoder,
        thinker: SpatialTRM,
        painter: SPADEUNet2D,
        diff_thinker_weight: float = 1.0,
    ):
        super().__init__(
            encoder=encoder, thinker=thinker,
            bridge=None, painter=painter,
            diff_thinker_weight=diff_thinker_weight,
        )

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond_scaled: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        painter_size = noisy.shape[-1]
        s = F.interpolate(
            spatial_cond_scaled, size=painter_size, mode="bilinear", align_corners=False
        )
        return self.painter(noisy, timesteps, s)


# ── V0 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV0(_TRMRatatouilleBase):
    """
    Closest to vanilla SudokuTRM: encoder sees CONDITION ONLY, 9×9 grid,
    TRM output channels = num_classes → direct digit logits per cell.

    The TRM that explicitly solves Sudoku also provides spatial conditioning
    for the DiT painter that renders the MNIST digits — joint training.

    Losses: diffusion MSE + weighted sudoku CE.
    Training wheels present (removed in V1→V4):
      clean input | explicit CE supervision | 9-channel output | 9×9 grid
    """

    _encoder_sees_noisy = False

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size   # e.g. 144//16 = 9

        encoder = SpatialEncoder(1, enc_channels, factor=cell_size)   # 1-ch: condition only
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        bridge  = SpatialBridge(num_classes, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels, layers_per_block=painter_layers_per_block)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V1 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV1(_TRMRatatouilleBase):
    """
    Same as V0 but encoder sees cat(condition, noisy_image).
    Training wheel removed: TRM must reason from a noisy/corrupted signal.
    Grid: 9×9.  Losses: diffusion + weighted sudoku CE.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)   # 2-ch: cond + noisy
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        bridge  = SpatialBridge(num_classes, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels, layers_per_block=painter_layers_per_block)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V2 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV2(_TRMRatatouilleBase):
    """
    Same as V1 but no sudoku CE loss.
    Training wheel removed: TRM gets no explicit digit-level supervision.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        bridge  = SpatialBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels, layers_per_block=painter_layers_per_block)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── V3 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV3(_TRMRatatouilleBase):
    """
    Same as V2 but thinker_out_channels >> 9 (not constrained to digit-logit space).
    Training wheel removed: output dimensionality is unconstrained.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        bridge  = SpatialBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels, layers_per_block=painter_layers_per_block)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── V4 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV4(_TRMRatatouilleBase):
    """
    Same as V3 but compression_factor ≠ cell_size → thinker grid ≠ 9×9,
    and uses AttentiveBridge (Perceiver-IO cross-attention) for upsampling.
    Training wheel removed: grid topology uncoupled from puzzle cell structure.
    Default: compression_factor=16 → 144//16=9×9 thinker grid.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 144,
        compression_factor: int = 16,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        bridge_num_heads: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // compression_factor

        encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        # AttentiveBridge: Perceiver-IO cross-attention upsamples low-res thinker
        # output to painter_size × painter_size via learned positional queries.
        bridge  = AttentiveBridge(
            in_channels=thinker_out_channels,
            out_channels=bridge_channels,
            out_resolution=painter_size,
            factor=compression_factor,
            num_heads=bridge_num_heads,
        )
        painter = _make_painter(painter_size, bridge_channels, painter_channels, layers_per_block=painter_layers_per_block)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── V0Control ─────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV0Control(_TRMRatatouilleControlBase):
    """V0 with ControlNet conditioning: encoder sees CONDITION ONLY, digit logits.

    Thinker output (9×9 × num_classes) is bilinearly upsampled to painter
    resolution and injected via ConditioningPyramid residuals instead of
    channel concatenation.  Painter has in_channels=1 (noisy only).
    """

    _encoder_sees_noisy = False

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(1, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter         = _make_painter_control(painter_size, painter_channels, layers_per_block=painter_layers_per_block)
        control_pyramid = ConditioningPyramid(num_classes, block_out_channels=painter_channels, layers_per_block=2)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         control_pyramid=control_pyramid, diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V1Control ─────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV1Control(_TRMRatatouilleControlBase):
    """V1 with ControlNet: encoder sees cat(condition, noisy), digit logits."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter         = _make_painter_control(painter_size, painter_channels, layers_per_block=painter_layers_per_block)
        control_pyramid = ConditioningPyramid(num_classes, block_out_channels=painter_channels, layers_per_block=2)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         control_pyramid=control_pyramid, diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V2Control ─────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV2Control(_TRMRatatouilleControlBase):
    """V2 with ControlNet: encoder sees cat(condition, noisy), no sudoku CE loss."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter         = _make_painter_control(painter_size, painter_channels, layers_per_block=painter_layers_per_block)
        control_pyramid = ConditioningPyramid(thinker_out_channels, block_out_channels=painter_channels, layers_per_block=2)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         control_pyramid=control_pyramid, diff_thinker_weight=diff_thinker_weight)


# ── V3Control ─────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV3Control(_TRMRatatouilleControlBase):
    """V3 with ControlNet: thinker_out_channels unconstrained (default 64)."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter         = _make_painter_control(painter_size, painter_channels, layers_per_block=painter_layers_per_block)
        control_pyramid = ConditioningPyramid(thinker_out_channels, block_out_channels=painter_channels, layers_per_block=2)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         control_pyramid=control_pyramid, diff_thinker_weight=diff_thinker_weight)


# ── V4Control ─────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV4Control(_TRMRatatouilleControlBase):
    """V4 with ControlNet: compression_factor decoupled from cell_size.

    Grid topology is uncoupled from puzzle cell structure; ControlNet pyramid
    handles the upsampling to painter resolution instead of AttentiveBridge.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 144,
        compression_factor: int = 16,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // compression_factor
        encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter         = _make_painter_control(painter_size, painter_channels, layers_per_block=painter_layers_per_block)
        control_pyramid = ConditioningPyramid(thinker_out_channels, block_out_channels=painter_channels, layers_per_block=2)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         control_pyramid=control_pyramid, diff_thinker_weight=diff_thinker_weight)


# ── V0SPADE ───────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV0SPADE(_TRMRatatouilleSPADEBase):
    """V0 with SPADE conditioning: encoder sees CONDITION ONLY, digit logits.

    Each ResBlock in the UNet is normalized by gamma/beta predicted from the
    thinker's spatial output, providing stronger per-activation conditioning.
    """

    _encoder_sees_noisy = False

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(1, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter = SPADEUNet2D(painter_size, sem_channels=num_classes,
                              block_out_channels=painter_channels, dropout=dropout,
                              layers_per_block=painter_layers_per_block)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V1SPADE ───────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV1SPADE(_TRMRatatouilleSPADEBase):
    """V1 with SPADE: encoder sees cat(condition, noisy), digit logits."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter = SPADEUNet2D(painter_size, sem_channels=num_classes,
                              block_out_channels=painter_channels, dropout=dropout,
                              layers_per_block=painter_layers_per_block)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V2SPADE ───────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV2SPADE(_TRMRatatouilleSPADEBase):
    """V2 with SPADE: encoder sees cat(condition, noisy), no sudoku CE loss."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter = SPADEUNet2D(painter_size, sem_channels=thinker_out_channels,
                              block_out_channels=painter_channels, dropout=dropout,
                              layers_per_block=painter_layers_per_block)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── V3SPADE ───────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV3SPADE(_TRMRatatouilleSPADEBase):
    """V3 with SPADE: thinker_out_channels unconstrained (default 64)."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter = SPADEUNet2D(painter_size, sem_channels=thinker_out_channels,
                              block_out_channels=painter_channels, dropout=dropout,
                              layers_per_block=painter_layers_per_block)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── V4SPADE ───────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV4SPADE(_TRMRatatouilleSPADEBase):
    """V4 with SPADE: compression_factor decoupled from cell_size."""

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 144,
        compression_factor: int = 16,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // compression_factor
        encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        painter = SPADEUNet2D(painter_size, sem_channels=thinker_out_channels,
                              block_out_channels=painter_channels, dropout=dropout,
                              layers_per_block=painter_layers_per_block)
        super().__init__(encoder=encoder, thinker=thinker, painter=painter,
                         diff_thinker_weight=diff_thinker_weight)


# ── Token encoder (integer tokens → spatial feature map) ─────────────────────────

class _TokenEncoder(nn.Module):
    """Converts integer puzzle tokens (B, 81) to a spatial feature map (B, out_channels, 9, 9).

    Acts as a drop-in replacement for SpatialEncoder when the input is discrete
    tokens rather than pixel images.  The output is compatible with SpatialTRM.embed().
    """

    def __init__(self, vocab_size: int, out_channels: int, grid_h: int = 9, grid_w: int = 9):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.embedding = nn.Embedding(vocab_size, out_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, grid_h*grid_w) long → (B, out_channels, grid_h, grid_w)"""
        B = tokens.shape[0]
        x = self.embedding(tokens)                             # (B, 81, out_channels)
        return x.transpose(1, 2).reshape(B, -1, self.grid_h, self.grid_w)


# ── V0Tok — integer-token input (no CNN encoder) ─────────────────────────────────

class MNISTRatatouilleV0Tok(MNISTRatatouilleV0):
    """V0 with direct integer puzzle-token input — no CNN encoder.

    Replaces SpatialEncoder with _TokenEncoder (nn.Embedding-based), exactly
    matching train_sudoku.py's discrete input representation.  Everything else
    (SpatialTRM thinker, bridge, painter, sudoku logits, EMA, etc.) is unchanged.

    The dataset must provide "puzzle_tokens" (B, 81) long with values
    0=PAD, 1=blank, 2-10=given digit.

    token_input=True tells the training loop to pass puzzle_tokens as the
    condition argument instead of the MNIST image.
    _encoder_sees_noisy=False ensures _get_encoder_input returns the tokens
    unchanged (no concatenation with noisy).
    """

    token_input: bool = True
    _encoder_sees_noisy = False

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        vocab_size: int = 11,      # 0=PAD, 1=blank, 2-10=given digit
        enc_channels: int = 64,    # embedding dim fed to SpatialTRM
        num_classes: int = 9,      # thinker output channels = digit classes
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        bridge_channels: int = 16,
        painter_channels: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.0,
        painter_layers_per_block: int = 2,
        num_puzzle_ids: int | None = None,
        diff_thinker_weight: float = 1.0,
    ):
        grid_size = painter_size // cell_size
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            num_classes=num_classes,
            enc_channels=enc_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            dropout=dropout,
            painter_layers_per_block=painter_layers_per_block,
            num_puzzle_ids=num_puzzle_ids,
            diff_thinker_weight=diff_thinker_weight,
        )
        # Replace CNN encoder with token embedding encoder.
        self.encoder = _TokenEncoder(vocab_size, enc_channels, grid_h=grid_size, grid_w=grid_size)
