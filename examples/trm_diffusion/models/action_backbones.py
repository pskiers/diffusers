"""
models/action_backbones.py — Diffusion backbones for action-sequence data,
ported from real-stanford/diffusion_policy (Chi et al. 2023) so this repo can
reproduce their base-DM results before layering TRM/ControlNet/IP-Adapter on top.

Both backbones operate on (B, T, action_dim) sequences and take a single flat
`global_cond` conditioning vector per sample (concat of the diffusion timestep
embedding with the observation embedding produced by a condition encoder) —
this matches upstream's conditioning scheme, which is FiLM/prefix-token based
rather than the cross-attention conditioning used by UNet2DConditionModel /
DiTPainter elsewhere in this repo.

  ConditionalUnet1D    — FiLM-conditioned conv1d U-Net (conditional_unet1d.py).
  TransformerForDiffusion — GPT-style causal decoder (transformer_for_diffusion.py).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


# ---------------------------------------------------------------------------
# ConditionalUnet1D
# ---------------------------------------------------------------------------


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish, matching upstream's Conv1dBlock."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning injected after the first block.

    cond_predict_scale=True (upstream default): the conditioning MLP predicts
    both a scale and a bias per channel (out = scale * x + bias). Otherwise
    only a bias is predicted (out = x + bias).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])

        self.cond_predict_scale = cond_predict_scale
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)

        embed = self.cond_encoder(cond)  # (B, cond_channels)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            bias = embed.reshape(embed.shape[0], self.out_channels, 1)
            out = out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """FiLM-conditioned conv1d U-Net over action sequences, matching upstream's
    conditional_unet1d.py::ConditionalUnet1D.

    Args:
        input_dim: action dimensionality (channels of the sequence).
        global_cond_dim: dimensionality of the flat observation conditioning
            vector (before it's concatenated with the diffusion timestep
            embedding). Pass 0 for an unconditional model.
        down_dims: channel width at each U-Net resolution level.
        diffusion_step_embed_dim: width of the sinusoidal timestep embedding.
        kernel_size, n_groups, cond_predict_scale: as in ConditionalResidualBlock1D.

    forward(sample, timestep, global_cond=None) -> (B, T, input_dim), matching
    the (sample, timestep, **kwargs) -> tensor convention the painter classes
    in action_painters.py call directly (no diffusers ModelOutput wrapper).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int = 0,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 256,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim

        dsed = diffusion_step_embed_dim
        self.time_proj = Timesteps(num_channels=dsed, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_mlp = nn.Sequential(
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        all_dims = (input_dim,) + tuple(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def res_block(dim_in, dim_out):
            return ConditionalResidualBlock1D(
                dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                n_groups=n_groups, cond_predict_scale=cond_predict_scale,
            )

        self.down_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                res_block(dim_in, dim_out),
                res_block(dim_out, dim_out),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            res_block(mid_dim, mid_dim),
            res_block(mid_dim, mid_dim),
        ])

        # Every down level except the last one downsamples (see the loop above),
        # so every up level here must upsample to mirror it symmetrically.
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(nn.ModuleList([
                res_block(dim_out * 2, dim_in),
                res_block(dim_in, dim_in),
                Upsample1d(dim_in),
            ]))

        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (B, T, input_dim) -> (B, input_dim, T)
        x = sample.transpose(1, 2)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        t_emb = self.time_mlp(self.time_proj(timestep).to(dtype=x.dtype))
        if global_cond is not None and global_cond.ndim == 3:
            # (B, n_obs_steps, D) -> (B, n_obs_steps*D): this backbone's FiLM
            # conditioning takes one flat vector per sample, matching upstream's
            # obs_as_global_cond=True path (condition encoders may otherwise
            # produce a per-step token sequence for TransformerForDiffusion).
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        cond = t_emb if global_cond is None else torch.cat([t_emb, global_cond], dim=-1)

        skip_connections = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            skip_connections.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, T, input_dim)


# ---------------------------------------------------------------------------
# TransformerForDiffusion
# ---------------------------------------------------------------------------


class TransformerForDiffusion(nn.Module):
    """GPT-style causal decoder over action tokens, matching upstream's
    transformer_for_diffusion.py::TransformerForDiffusion (single-decoder,
    n_cond_layers=0 configuration: cond/time tokens are prepended to the
    action-token sequence and attended to causally, rather than routed
    through a separate cross-attention encoder-decoder stack).

    forward(sample, timestep, global_cond=None) -> (B, T, input_dim).
    global_cond is a per-obs-step token sequence, (B, n_obs_steps, cond_dim) —
    matching upstream, which projects each observation step to its own cond
    token via a single shared Linear(cond_dim, n_emb) rather than flattening
    the obs history into one vector (that flattening is ConditionalUnet1D's
    convention, not this one). Pass cond_dim=0 for an unconditional model.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cond_dim: int = 0,
        n_obs_steps: int = 0,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.01,
        causal_attn: bool = True,
    ):
        super().__init__()
        self.n_emb = n_emb
        self.causal_attn = causal_attn
        self.has_cond = cond_dim > 0 and n_obs_steps > 0

        # 1 time token + one cond token per obs step, prepended to the action sequence.
        self.n_cond_tokens = (1 + n_obs_steps) if self.has_cond else 1

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.n_cond_tokens, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        time_dim = n_emb
        self.time_proj = Timesteps(num_channels=time_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_emb = TimestepEmbedding(time_dim, n_emb)

        if self.has_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, input_dim)

        self._horizon = horizon
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def _causal_mask(self, n_cond: int, n_act: int, device) -> torch.Tensor:
        """Cond/time tokens attend only to themselves; action tokens attend
        causally to all cond tokens plus earlier (and same-index) action tokens."""
        total = n_cond + n_act
        mask = torch.full((total, total), float("-inf"), device=device)
        mask[:, :n_cond] = 0.0  # everyone can see the cond/time tokens
        act_causal = torch.triu(torch.ones(n_act, n_act, device=device), diagonal=1).bool()
        mask[n_cond:, n_cond:] = act_causal.float().masked_fill(act_causal, float("-inf"))
        return mask

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = sample.shape
        device = sample.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.expand(B)
        time_token = self.time_emb(self.time_proj(timestep).to(dtype=sample.dtype)).unsqueeze(1)  # (B, 1, n_emb)

        cond_tokens = [time_token]
        if self.has_cond:
            if global_cond is None:
                raise ValueError("This TransformerForDiffusion was built with cond_dim > 0.")
            cond_tokens.append(self.cond_obs_emb(global_cond))  # (B, n_obs_steps, n_emb)
        cond = torch.cat(cond_tokens, dim=1) + self.cond_pos_emb

        act = self.input_emb(sample) + self.pos_emb[:, :T]
        x = self.drop(torch.cat([cond, act], dim=1))

        mask = self._causal_mask(cond.shape[1], T, device) if self.causal_attn else None
        x = self.decoder(x, mask=mask)

        x = self.ln_f(x)
        out = self.head(x[:, cond.shape[1]:])  # drop cond/time positions, keep action tokens
        return out
