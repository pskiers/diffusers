"""
models/condition_encoders.py — Pluggable condition encoders.

A ConditionEncoderBase subclass:
  - declares which batch keys it reads via condition_keys
  - encodes the extracted condition tensor (+ optional noisy latent) into
    TRM-compatible token sequences

Calling convention (model's _get_enc_emb):
    condition = batch[condition_encoder.condition_keys[0]].to(device)
    enc_emb = condition_encoder(condition, noisy, timesteps)
    -> (B, n_tokens, hidden_size)

The noisy and timesteps arguments are optional; V0 encoders ignore them, V1
encoders use noisy to produce additional latent tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utility_models import SpatialEncoder


class ConditionEncoderBase(nn.Module):
    """
    Abstract base.  Subclasses declare which batch keys they read.

    Attributes:
        condition_keys: list of batch keys this encoder reads.  The model uses
                        condition_keys[0] as the primary condition key.  Future
                        encoders can read multiple keys.
    """

    condition_keys: list[str] = ["conditions"]

    def forward(
        self,
        condition: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ── Spatial image encoder (pixel-space MNIST-style models) ────────────────────


class SpatialImageEncoder(ConditionEncoderBase):
    """
    CNN condition encoder for pixel-space MNIST-style models.

    Wraps SpatialEncoder (CNN) + 1x1 Conv2d projection into the ConditionEncoderBase
    interface. V0: condition image only. V1: cat(condition, noisy) as input.

    Input:  condition  (B, 1, H, W) — puzzle image in [-1,1]
            noisy      (B, 1, H, W) — noisy image (V1 only)
    Output: (B, grid², output_dim)  where grid = image_size // factor
    """

    condition_keys: list[str] = ["conditions"]

    def __init__(
        self,
        in_channels: int,  # 1 (V0) or 2 (V1)
        enc_channels: int,  # SpatialEncoder output channels
        hidden_channels: list[int],  # intermediate CNN channels
        output_dim: int,  # = thinker.hidden_size
        factor: int,  # downsampling factor = cell_size
        use_noisy_input: bool = False,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.use_noisy_input = use_noisy_input
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.enc = SpatialEncoder(in_channels, enc_channels, factor=factor, hidden_channels=tuple(hidden_channels))
        std = 1.0 / (math.sqrt(output_dim) * math.sqrt(enc_channels))
        self.proj = nn.Conv2d(enc_channels, output_dim, 1)
        nn.init.normal_(self.proj.weight, std=std)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        condition: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_noisy_input and noisy is not None:
            noisy_in = noisy
            if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
                t_norm = timesteps.float() / self.num_train_timesteps
                p = self.noisy_dropout_p_max * (1.0 - t_norm)
                keep = (torch.rand(p.shape, device=p.device) > p).float()
                noisy_in = noisy * keep[:, None, None, None]
            x = torch.cat([condition, noisy_in], dim=1)
        else:
            x = condition
        feat = self.enc(x)
        proj = self.proj(feat)
        return proj.flatten(2).transpose(1, 2)  # (B, grid², output_dim)


# ── Object feature encoders ───────────────────────────────────────────────────


class ObjectFeatureEncoder(ConditionEncoderBase):
    """
    V0 encoder: MLP that maps per-object feature vectors to TRM tokens.

    Wraps a small MLP with the ConditionEncoderBase interface.
    noisy and timesteps are ignored.

    Input:  condition (B, max_objects, object_feat_dim)
    Output: (B, max_objects, out_dim)
    """

    condition_keys: list[str] = ["conditions"]

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        condition: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.net(condition)  # (B, max_objects, out_dim)


class ClevrLatentEncoder(nn.Module):
    """
    Two-block CNN that maps a noisy VAE latent to G×G spatial tokens.

    Input:  (B, latent_channels, latent_size, latent_size)
    Output: (B, G², hidden_size)

    Zero-init last conv so the encoder contributes nothing at the start of training.
    """

    def __init__(self, in_channels: int, hidden_size: int, grid_size: int):
        super().__init__()
        mid = hidden_size // 2
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        nn.init.zeros_(self.conv[2].weight)
        nn.init.zeros_(self.conv[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        if feat.shape[-1] != self.grid_size:
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        return feat.flatten(2).transpose(1, 2)  # (B, G², hidden)


class ObjectFeatureEncoderV1(ConditionEncoderBase):
    """
    V1 encoder: per-object tokens concatenated with noisy-latent tokens.

    Encodes both the object feature condition and a noisy VAE latent, then
    concatenates them into a single token sequence for the TRM.

    Input:  condition (B, max_objects, object_feat_dim)
            noisy     (B, latent_channels, H, W)  — required at call time
    Output: (B, max_objects + grid_size², hidden_size)

    Args:
        noisy_dropout_p_max: maximum per-sample dropout probability for the
            noisy-latent tokens during training (linearly scaled by 1 - t/T).
            0.0 disables dropout.
        num_train_timesteps: total diffusion steps (for computing dropout prob).
    """

    condition_keys: list[str] = ["conditions"]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        latent_channels: int,
        hidden_size: int,
        grid_size: int,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim)
        self.latent_encoder = ClevrLatentEncoder(latent_channels, hidden_size, grid_size)
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps

    def forward(
        self,
        condition: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obj_tokens = self.object_encoder(condition)  # (B, max_objects, hidden)

        noisy_for_enc = noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None and noisy is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = noisy * keep[:, None, None, None]

        img_tokens = self.latent_encoder(noisy_for_enc)  # (B, G², hidden)
        return torch.cat([obj_tokens, img_tokens], dim=1)  # (B, max_objects+G², hidden)


# ── Backward-compatible aliases ───────────────────────────────────────────────

ClevrObjectFeatureEncoder = ObjectFeatureEncoder
ClevrNoisyObjectFeatureEncoder = ObjectFeatureEncoderV1
