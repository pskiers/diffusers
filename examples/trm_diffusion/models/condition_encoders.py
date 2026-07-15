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
from models.utility_models import SpatialEncoder, TimestepMLP


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
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.enc, self.proj = _build_spatial_enc(in_channels, enc_channels, hidden_channels, output_dim, factor)
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=output_dim)

    def forward(self, condition: torch.Tensor, timesteps=None, **_) -> TRMInput:
        feat = self.enc(condition)
        emb = self.proj(feat).flatten(2).transpose(1, 2)  # (B, grid², output_dim)
        if self.with_timestep_emb and timesteps is not None:
            emb = emb + self.timestep_mlp(timesteps).unsqueeze(1)
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
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.enc, self.proj = _build_spatial_enc(in_channels, enc_channels, hidden_channels, output_dim, factor)
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=output_dim)

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
        if self.with_timestep_emb and timesteps is not None:
            emb = emb + self.timestep_mlp(timesteps).unsqueeze(1)
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

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, with_timestep_emb: bool = False):
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
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    def forward(self, condition: torch.Tensor, timesteps=None, **_) -> TRMInput:
        emb = self.net(condition)  # (B, max_objects, out_dim)
        if self.with_timestep_emb and timesteps is not None:
            emb = emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=emb)


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
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels, hidden_size, grid_size)
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

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
        enc_emb = torch.cat([obj_tokens, img_tokens], dim=1)
        if self.with_timestep_emb and timesteps is not None:
            enc_emb = enc_emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=enc_emb)


# ── Backward-compatible aliases ───────────────────────────────────────────────

ClevrObjectFeatureEncoder = ObjectFeatureEncoder
ClevrNoisyObjectFeatureEncoder = ObjectFeatureEncoderV1


# ── Action-sequence (global_cond) encoders ────────────────────────────────────
#
# The encoders above all produce a TRMInput (a token sequence consumed via
# cross-attention, per interfaces.py's TRMInput.to_painter_kwargs()). The
# action_backbones.py painters (ConditionalUnet1D / TransformerForDiffusion)
# instead take conditioning per observation step — FiLM (flattened into one
# vector by ConditionalUnet1D itself) or per-step prefix tokens (consumed
# as-is by TransformerForDiffusion) — matching real-stanford/diffusion_policy.
# These encoders implement that different (simpler) contract: condition_keys
# still declares which DataSample fields to read, but forward() returns a
# plain (B, n_obs_steps, D) tensor instead of a TRMInput.


class GlobalCondEncoderBase(nn.Module):
    """Base for condition encoders that produce a (B, n_obs_steps, D) tensor.

    Used by models/action_painters.py, not by the thinker/TRM pipeline.
    """

    condition_keys: list[str] = []

    def forward(self, *args, timesteps: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LowdimObsConditionEncoder(GlobalCondEncoderBase):
    """Passes the observation-history tensor through unchanged.

    Matches upstream diffusion_policy's lowdim policies: no learned encoder,
    the raw (normalized) observation history is used directly as conditioning
    (ConditionalUnet1D flattens it into one FiLM vector itself;
    TransformerForDiffusion consumes it as n_obs_steps separate cond tokens).

    Input/Output: embedding_conditions (B, n_obs_steps, obs_dim)
    """

    condition_keys: list[str] = ["embedding_conditions"]

    def forward(self, embedding_conditions: torch.Tensor, timesteps=None, **_) -> torch.Tensor:
        return embedding_conditions


def _replace_batchnorm_with_groupnorm(module: nn.Module) -> nn.Module:
    """Recursively replace every BatchNorm2d with GroupNorm(num_features // 16),
    matching upstream's MultiImageObsEncoder(use_group_norm=True)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=max(1, child.num_features // 16), num_channels=child.num_features))
        else:
            _replace_batchnorm_with_groupnorm(child)
    return module


class ImageObsConditionEncoder(GlobalCondEncoderBase):
    """ResNet18 vision encoder + raw low-dim obs, matching upstream's
    MultiImageObsEncoder (share_rgb_model=True path): a single shared ResNet18
    backbone (fc replaced with Identity, BatchNorm optionally replaced with
    GroupNorm) encodes every camera view via its own global average pool (no
    spatial softmax — upstream doesn't use one either); per-view features are
    concatenated, then concatenated with the raw low-dim observation.

    crop_shape (e.g. [76, 76]) matches upstream's train-time random-crop /
    eval-time center-crop augmentation ("eval_fixed_crop: True"). The random
    crop offset is shared across all obs-steps/views of a given sample (a
    fixed camera crop for the whole rollout) but drawn independently per
    sample in the batch.

    Input:
      spatial_conditions   (B, T, C, H, W)     — single camera view, or
                            (B, T, V, C, H, W)  — V camera views (kept
                            separate, e.g. ToolHangImageDataset)
      embedding_conditions (B, T, obs_dim)     — optional raw low-dim obs
                            (e.g. agent_pos / robot proprioception)
    Output: (B, T, V * resnet_feature_dim + obs_dim)
    """

    condition_keys: list[str] = ["spatial_conditions", "embedding_conditions"]

    def __init__(self, use_group_norm: bool = True, pretrained: bool = False, crop_shape=None):
        super().__init__()
        import torchvision

        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        resnet.fc = nn.Identity()
        if use_group_norm:
            resnet = _replace_batchnorm_with_groupnorm(resnet)
        self.resnet = resnet
        self.feature_dim = 512  # resnet18's pre-fc feature width
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None

    def _crop(self, imgs: torch.Tensor, per_sample: int) -> torch.Tensor:
        """imgs: (B*per_sample, C, H, W), grouped in contiguous per-sample blocks."""
        if self.crop_shape is None:
            return imgs
        n, c, h, w = imgs.shape
        ch, cw = self.crop_shape
        out = imgs.new_empty(n, c, ch, cw)
        for b in range(n // per_sample):
            sl = slice(b * per_sample, (b + 1) * per_sample)
            if self.training:
                top = int(torch.randint(0, h - ch + 1, (1,)))
                left = int(torch.randint(0, w - cw + 1, (1,)))
            else:
                top, left = (h - ch) // 2, (w - cw) // 2
            out[sl] = imgs[sl][:, :, top:top + ch, left:left + cw]
        return out

    def forward(self, spatial_conditions: torch.Tensor, embedding_conditions=None, timesteps=None, **_) -> torch.Tensor:
        x = spatial_conditions
        multiview = x.ndim == 6
        if multiview:
            B, T, V, C, H, W = x.shape
            x = self._crop(x.reshape(B * T * V, C, H, W), per_sample=T * V)
            feat = self.resnet(x)
            feat = feat.reshape(B, T, V * self.feature_dim)
        else:
            B, T, C, H, W = x.shape
            x = self._crop(x.reshape(B * T, C, H, W), per_sample=T)
            feat = self.resnet(x)
            feat = feat.reshape(B, T, self.feature_dim)

        if embedding_conditions is not None:
            feat = torch.cat([feat, embedding_conditions], dim=-1)

        return feat
