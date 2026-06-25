"""
models/condition_encoders.py — Pluggable condition encoders.

A ConditionEncoderBase subclass:
  - declares which DataSample fields it reads via ``condition_keys``
  - encodes the extracted condition tensor (+ optional noisy latent) into
    a TRMInput

Calling convention (model):
    field = getattr(sample, condition_encoder.condition_keys[0])
    trm_input = condition_encoder(field, noisy, timesteps)
    # trm_input.enc_emb: (B, n_tokens, hidden_size)

The noisy and timesteps arguments are optional; V0 encoders ignore them, V1
encoders use noisy to produce additional latent tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.interfaces import TRMInput
from models.utility_models import SpatialEncoder


class ConditionEncoderBase(nn.Module):
    """
    Abstract base.  Subclasses declare which DataSample fields they read.

    Attributes:
        condition_keys: list of DataSample field names this encoder reads.
                        The model passes ``getattr(sample, condition_keys[0])``
                        as the primary condition argument.
    """

    condition_keys: list[str] = ["spatial_conditions"]

    def forward(
        self,
        condition: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        raise NotImplementedError


# ── Spatial image encoders (pixel-space MNIST-style models) ──────────────────


def _build_spatial_enc(in_channels, enc_channels, hidden_channels, output_dim, factor):
    enc = SpatialEncoder(in_channels, enc_channels, factor=factor, hidden_channels=tuple(hidden_channels))
    std = 1.0 / (math.sqrt(output_dim) * math.sqrt(enc_channels))
    proj = nn.Conv2d(enc_channels, output_dim, 1)
    nn.init.normal_(proj.weight, std=std)
    nn.init.zeros_(proj.bias)
    return enc, proj


class SpatialConditionEncoder(ConditionEncoderBase):
    """
    CNN encoder for a spatial condition image only (V0).

    Reads ``spatial_conditions`` from the DataSample.

    Input:  condition  (B, C, H, W) — spatial condition image
    Output: TRMInput with enc_emb (B, grid², output_dim)
            where grid = image_size // factor
    """

    condition_keys: list[str] = ["spatial_conditions"]

    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        hidden_channels: list[int],
        output_dim: int,
        factor: int,
    ):
        super().__init__()
        self.enc, self.proj = _build_spatial_enc(in_channels, enc_channels, hidden_channels, output_dim, factor)

    def forward(self, condition: torch.Tensor, **_) -> TRMInput:
        feat = self.enc(condition)
        emb = self.proj(feat).flatten(2).transpose(1, 2)  # (B, grid², output_dim)
        return TRMInput(enc_emb=emb)


class NoisySpatialConditionEncoder(ConditionEncoderBase):
    """
    CNN encoder for a spatial condition concatenated with the noisy image (V1).

    Reads ``spatial_conditions`` and ``x_noisy`` from the DataSample.
    ``x_noisy`` is populated by the model from the diffusion schedule
    before the encoder is called.

    Input:  condition (B, C_cond,  H, W) — spatial condition image
            x_noisy   (B, C_noisy, H, W) — noisy diffusion image (x_t)
    Output: TRMInput with enc_emb (B, grid², output_dim)
    """

    condition_keys: list[str] = ["spatial_conditions", "x_noisy"]

    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        hidden_channels: list[int],
        output_dim: int,
        factor: int,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.enc, self.proj = _build_spatial_enc(in_channels, enc_channels, hidden_channels, output_dim, factor)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        noisy_in = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_in = x_noisy * keep[:, None, None, None]
        feat = self.enc(torch.cat([condition, noisy_in], dim=1))
        emb = self.proj(feat).flatten(2).transpose(1, 2)  # (B, grid², output_dim)
        return TRMInput(enc_emb=emb)


# Backward-compatible alias — existing Hydra configs reference SpatialImageEncoder.
SpatialImageEncoder = SpatialConditionEncoder


# ── Object feature encoders ───────────────────────────────────────────────────


class ObjectFeatureEncoder(ConditionEncoderBase):
    """
    V0 encoder: MLP that maps per-object feature vectors to TRM tokens.

    Reads ``embedding_conditions`` from the DataSample.

    Input:  condition (B, max_objects, object_feat_dim)
    Output: TRMInput with enc_emb (B, max_objects, out_dim)
    """

    condition_keys: list[str] = ["embedding_conditions"]

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

    def forward(self, condition: torch.Tensor, **_) -> TRMInput:
        return TRMInput(enc_emb=self.net(condition))  # (B, max_objects, out_dim)


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

    condition_keys: list[str] = ["embedding_conditions", "x_noisy"]

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
        x_noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, max_objects, hidden)

        noisy_for_enc = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = x_noisy * keep[:, None, None, None]

        img_tokens = self.latent_encoder(noisy_for_enc)  # (B, G², hidden)
        return TRMInput(enc_emb=torch.cat([obj_tokens, img_tokens], dim=1))


# ── Backward-compatible aliases ───────────────────────────────────────────────

ClevrObjectFeatureEncoder = ObjectFeatureEncoder
ClevrNoisyObjectFeatureEncoder = ObjectFeatureEncoderV1
