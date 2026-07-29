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

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.data_sample import DataSample
from models.diffusion_utils import x0_from_noise_pred
from models.interfaces import TRMInput
from models.utility_models import SpatialEncoder, TimestepMLP


class ConditionEncoderBase(nn.Module):
    """
    Abstract base.  Subclasses declare which DataSample fields they read.

    Attributes:
        condition_keys: list of DataSample field names this encoder reads.
                        The model passes ``getattr(sample, condition_keys[0])``
                        as the primary condition argument.
        needs_sample:   if True, ThinkerFrozenPainterBase._encode_condition
                        also passes the full DataSample as `sample=` — for
                        encoders that need more than just their declared
                        condition_keys (e.g. X0PredHintConditionEncoder,
                        which runs an internal frozen-painter forward pass
                        that may need the painter's own conditioning fields
                        too, not just what the thinker's encoder reads).
    """

    condition_keys: list[str] = ["spatial_conditions"]
    needs_sample: bool = False

    def bind_painter(self, painter) -> None:
        """Optional hook: encoders that need access to the frozen painter
        (e.g. to run an extra denoising pass for an x0-prediction hint)
        override this. No-op by default. Called once by the owning model
        (ThinkerFrozenPainterBase) right after both painter and
        condition_encoder are instantiated."""

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


class NoisySpatialConditionEncoderV2(ConditionEncoderBase):
    """
    CNN encoder for a spatial condition and the noisy image, encoded through
    separate CNN branches and concatenated token-wise (sequence-length axis)
    rather than channel-wise like ``NoisySpatialConditionEncoder`` (V1).

    V1's channel-wise concat assumes condition and noisy image are pixel-
    aligned (same cell grid, same position/scale) — true for plain MNIST
    Sudoku, false once the target is randomly scaled/offset (mnist_sudoku_scaled)
    or when the "condition" isn't even spatial to begin with (CLEVR relations).
    Concatenating as two separate token blocks instead lets the thinker's
    self-attention learn whatever relation actually holds between them.

    Reads ``spatial_conditions`` and ``x_noisy`` from the DataSample.

    Output: TRMInput with enc_emb (B, 2*grid², output_dim) — condition
    tokens first, followed by noisy tokens. The noisy-token block is
    positionally registered with the real output image (it's encoded
    straight from x_noisy pixels), which is what SlicedControlNetTranslator
    reads back out for steering — see models/translators.py.
    """

    condition_keys: list[str] = ["spatial_conditions", "x_noisy"]

    def __init__(
        self,
        cond_in_channels: int,
        noisy_in_channels: int,
        enc_channels: int,
        hidden_channels: list[int],
        output_dim: int,
        factor: int,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.cond_enc, self.cond_proj = _build_spatial_enc(cond_in_channels, enc_channels, hidden_channels, output_dim, factor)
        self.noisy_enc, self.noisy_proj = _build_spatial_enc(noisy_in_channels, enc_channels, hidden_channels, output_dim, factor)
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
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

        cond_tokens = self.cond_proj(self.cond_enc(condition)).flatten(2).transpose(1, 2)  # (B, grid², output_dim)
        noisy_tokens = self.noisy_proj(self.noisy_enc(noisy_in)).flatten(2).transpose(1, 2)  # (B, grid², output_dim)

        emb = torch.cat([cond_tokens, noisy_tokens], dim=1)  # (B, 2*grid², output_dim)
        if self.with_timestep_emb and timesteps is not None:
            emb = emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=emb)


class X0PredHintConditionEncoder(ConditionEncoderBase):
    """
    Wraps another (noisy-aware) condition encoder and replaces its raw
    ``x_noisy`` input with a blurred x0 estimate before delegating — a
    coarse-to-fine anti-exposure-bias hint instead of the raw diffusion
    state.

    Motivation: feeding the TRM raw x_t lets it exploit whatever ground-truth
    signal survives at low noise levels ("exposure bias") instead of
    reasoning over the puzzle. This computes x0_pred via one frozen,
    unsteered forward pass of the bound painter (see ``bind_painter``) at
    ``t_hint = max(timesteps, threshold)`` — i.e. never at a noise level
    below ``threshold`` — first re-noising x_noisy up to ``threshold`` when
    the real timestep is already cleaner than that. The result carries the
    low-frequency spatial layout (scale/offset) the TRM needs while hiding
    exact digit identity behind noise that never drops below the floor.

    Requires ``bind_painter()`` to have been called with a frozen painter
    exposing ``.scheduler`` and callable as ``painter(sample, steering=None)``
    — the owning model (ThinkerFrozenPainterBase) does this automatically.

    Generic across conditioned and unconditioned frozen painters: the internal
    hint pass carries through the real sample's own fields (via
    ``needs_sample``, see ConditionEncoderBase) rather than assuming an
    unconditional pixel-space painter with no condition_encoder of its own
    (true for MNIST, false for e.g. CLEVR's object-feature-conditioned UNet).
    If the bound painter's ``train_cfg.force_unconditional_painter`` is set,
    the hint pass also nulls the painter's own conditioning (via
    ``painter.null_condition_sample``), matching what
    ThinkerFrozenPainterBase.run_painter does for the real forward pass —
    so the hint reflects what the painter will actually be run as. Also
    handles VAE-latent-space painters correctly (skips x0_from_noise_pred's
    [0,1] pixel clamp, which would otherwise corrupt unbounded latents).

    Args:
        inner:     Hydra config for the wrapped ConditionEncoderBase (e.g.
                    NoisySpatialConditionEncoder, ...V2, or
                    ObjectFeatureEncoderV1) — instantiated recursively by
                    Hydra before reaching this constructor. Must accept an
                    ``x_noisy`` kwarg.
        threshold: minimum diffusion timestep the hint is ever computed at.
    """

    needs_sample = True

    def __init__(self, inner: ConditionEncoderBase, threshold: int):
        super().__init__()
        self.inner = inner
        self.threshold = threshold
        self._painter = None

    @property
    def condition_keys(self) -> list[str]:
        return self.inner.condition_keys

    def bind_painter(self, painter) -> None:
        self._painter = painter

    @torch.no_grad()
    def _x0_pred_hint(self, sample: DataSample, x_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self._painter is None:
            raise RuntimeError("X0PredHintConditionEncoder requires bind_painter() before use.")
        scheduler = self._painter.scheduler
        threshold = torch.full_like(timesteps, self.threshold)
        t_hint = torch.maximum(timesteps, threshold)

        x_for_hint = x_noisy
        needs_renoise = timesteps < threshold
        if needs_renoise.any():
            renoised = scheduler.add_noise(x_noisy, torch.randn_like(x_noisy), t_hint)
            mask = needs_renoise.view(-1, *([1] * (x_noisy.ndim - 1)))
            x_for_hint = torch.where(mask, renoised, x_noisy)

        # Carry the real sample's own fields through — the painter's own
        # conditioning (e.g. CLEVR's embedding_conditions) may be required
        # for it to run at all, not just x_noisy/timesteps.
        hint_sample = dataclasses.replace(sample, x_noisy=x_for_hint, timesteps=t_hint)
        train_cfg = getattr(self._painter, "train_cfg", None)
        if train_cfg is not None and getattr(train_cfg, "force_unconditional_painter", False):
            hint_sample = self._painter.null_condition_sample(hint_sample)

        eps_pred = self._painter(hint_sample, steering=None).pred
        has_vae = getattr(self._painter, "vae", None) is not None
        return x0_from_noise_pred(eps_pred, x_for_hint, t_hint, scheduler, clamp=not has_vae)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        sample: Optional[DataSample] = None,
        **extra,
    ) -> TRMInput:
        """**extra passes through untouched to self.inner — needed for inner
        encoders that declare condition_keys beyond embedding_conditions/
        x_noisy/timesteps (e.g. ObjectFeatureEncoderV1Reveal/CentroidMask's
        spatial_conditions). Only x_noisy is intercepted and replaced with
        the hint; everything else is the inner encoder's business."""
        hint = self._x0_pred_hint(sample, x_noisy, timesteps) if timesteps is not None else x_noisy
        return self.inner(condition, x_noisy=hint, timesteps=timesteps, **extra)


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


# ── CLEVR diagnostics: swatch / reveal / centroid-mask encoders ──────────────
#
# Three variants testing what MNIST-Sudoku's same_digit_images and
# center_condition/aligned-dataset findings translate to for CLEVR (object-
# relational conditioning, not a rendered puzzle image). Each augments
# ObjectFeatureEncoderV1 (object tokens + noisy-latent tokens) with one extra
# signal:
#   - Swatch:        a real photorealistic anchor per object, built purely
#                     from its attributes — deployable (available at real
#                     inference, unlike the other two).
#   - Reveal:        an exact crop of the REAL target image around a few
#                     objects' true positions — diagnostic only (needs the
#                     real image, which doesn't exist yet at generation time).
#   - Centroid mask: a spatial map of every object's TRUE position/attributes
#                     — also diagnostic only, for the same reason.
#
# All three assume embedding_conditions' first 15 dims are always
# [rot_sin, rot_cos, size_id, material_id, shape_onehot(3), color_onehot(8)]
# — true for every mode (absolute/relative/reduced); see
# datasets.clevr_dataset.make_tensor_from_scene, which always builds this
# `base` block before appending mode-specific dims.

from datasets.clevr_dataset import COLORS as _CLEVR_COLORS_
from datasets.clevr_dataset import MATERIALS as _CLEVR_MATERIALS_
from datasets.clevr_dataset import SHAPES as _CLEVR_SHAPES_
from datasets.clevr_dataset import SIZES as _CLEVR_SIZES_


def _clevr_swatch_indices(condition: torch.Tensor) -> torch.Tensor:
    """Decode each object's (color, shape, material, size) combo index
    straight out of embedding_conditions' first 15 dims — no dataset changes
    needed, since this block is identical across every mode. Must match the
    (color, shape, material, size) nesting order
    datasets.clevr_dataset.extract_clevr_swatch_table builds its table in."""
    size_id = condition[..., 2].round().long().clamp(0, 1)
    mat_id = condition[..., 3].round().long().clamp(0, 1)
    shape_id = condition[..., 4:7].argmax(dim=-1)
    color_id = condition[..., 7:15].argmax(dim=-1)
    return ((color_id * len(_CLEVR_SHAPES_) + shape_id) * len(_CLEVR_MATERIALS_) + mat_id) * len(
        _CLEVR_SIZES_
    ) + size_id


class ObjectFeatureEncoderV1Swatch(ConditionEncoderBase):
    """ObjectFeatureEncoderV1 (object tokens + noisy-latent tokens) plus a
    per-object visual anchor: a small CNN feature extracted from a real
    photorealistic crop of that object's (shape, color, material, size)
    combination, pulled once from real training images (see
    datasets.clevr_dataset.extract_clevr_swatch_table) — a hand-drawn icon
    would have essentially no visual correspondence to a Blender render, so
    the anchor has to be real pixels from the same renderer/lighting engine,
    not a synthetic stand-in. Deployable — the lookup table is built once
    from attributes, available at real inference too, unlike the
    reveal/centroid-mask variants below which need the current scene's
    ground truth.

    Args:
        swatch_table_path: path to a (N_combos, 3, S, S) tensor saved by
            experiments/build_clevr_swatch_table.py.
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
        swatch_table_path: str,
        swatch_channels: int = 32,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels, hidden_size, grid_size)
        self.swatch_cnn = nn.Sequential(
            nn.Conv2d(3, swatch_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(swatch_channels, swatch_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.swatch_proj = nn.Linear(swatch_channels, out_dim)
        nn.init.zeros_(self.swatch_proj.weight)
        nn.init.zeros_(self.swatch_proj.bias)
        swatch_table = torch.load(swatch_table_path, map_location="cpu", weights_only=True)
        self.register_buffer("swatch_table", swatch_table, persistent=False)
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
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, N, out_dim)

        idx = _clevr_swatch_indices(condition)  # (B, N)
        B, N = idx.shape
        swatches = self.swatch_table[idx.reshape(-1)]  # (B*N, 3, S, S)
        swatch_feat = self.swatch_cnn(swatches).flatten(1)  # (B*N, swatch_channels)
        swatch_emb = self.swatch_proj(swatch_feat).view(B, N, -1)  # (B, N, out_dim)
        obj_tokens = obj_tokens + swatch_emb

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


class ObjectFeatureEncoderV1Reveal(ConditionEncoderBase):
    """ObjectFeatureEncoderV1 plus a third token block encoding
    ``spatial_conditions`` — a real-image reveal around a handful of
    objects' true positions (see datasets.clevr_dataset.make_reveal_from_scene).
    Diagnostic only: requires the real target image, unavailable at real
    generation time.
    """

    condition_keys: list[str] = ["embedding_conditions", "x_noisy", "spatial_conditions"]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        latent_channels: int,
        hidden_size: int,
        grid_size: int,
        reveal_channels: int = 3,
        reveal_enc_channels: int = 64,
        reveal_hidden_channels: Optional[list[int]] = None,
        reveal_factor: int = 32,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels, hidden_size, grid_size)
        self.reveal_enc, self.reveal_proj = _build_spatial_enc(
            reveal_channels, reveal_enc_channels, reveal_hidden_channels or [64, 128], hidden_size, reveal_factor
        )
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        spatial_conditions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, N, out_dim)

        noisy_for_enc = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = x_noisy * keep[:, None, None, None]
        img_tokens = self.latent_encoder(noisy_for_enc)  # (B, G², hidden)

        reveal_feat = self.reveal_proj(self.reveal_enc(spatial_conditions)).flatten(2).transpose(1, 2)  # (B, g², hidden)

        enc_emb = torch.cat([obj_tokens, img_tokens, reveal_feat], dim=1)
        if self.with_timestep_emb and timesteps is not None:
            enc_emb = enc_emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=enc_emb)


class ObjectFeatureEncoderV1CentroidMask(ConditionEncoderBase):
    """ObjectFeatureEncoderV1 plus a third token block encoding
    ``spatial_conditions`` — a per-attribute Gaussian-blob mask at every
    object's TRUE position (see datasets.clevr_dataset.make_mask_from_scene).
    Diagnostic only: the true positions are exactly what the model is
    supposed to invent from relations, not something available at real
    generation time — this tests whether a perfect position signal (of the
    kind mode="absolute" gives as raw coordinates) helps when given instead
    as a spatial image, matching the noisy latent's own representation.
    """

    condition_keys: list[str] = ["embedding_conditions", "x_noisy", "spatial_conditions"]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        latent_channels: int,
        hidden_size: int,
        grid_size: int,
        mask_channels: int = 16,
        mask_enc_channels: int = 64,
        mask_hidden_channels: Optional[list[int]] = None,
        mask_factor: int = 4,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels, hidden_size, grid_size)
        self.mask_enc, self.mask_proj = _build_spatial_enc(
            mask_channels, mask_enc_channels, mask_hidden_channels or [64, 128], hidden_size, mask_factor
        )
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        spatial_conditions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, N, out_dim)

        noisy_for_enc = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = x_noisy * keep[:, None, None, None]
        img_tokens = self.latent_encoder(noisy_for_enc)  # (B, G², hidden)

        mask_feat = self.mask_proj(self.mask_enc(spatial_conditions)).flatten(2).transpose(1, 2)  # (B, g², hidden)

        enc_emb = torch.cat([obj_tokens, img_tokens, mask_feat], dim=1)
        if self.with_timestep_emb and timesteps is not None:
            enc_emb = enc_emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=enc_emb)


class ObjectFeatureEncoderV1RevealFused(ConditionEncoderBase):
    """Channel-concat variant of ObjectFeatureEncoderV1Reveal: the reveal
    image is resized to the noisy latent's spatial size and concatenated
    channel-wise with it, then run through ONE combined CNN — instead of a
    separate token block. Half the extra tokens (max_objects + grid_size²,
    not + 2*grid_size²).

    This is safe here in a way MNIST-Sudoku's channel-concat V1 encoder
    wasn't: that one assumed the condition image and x_noisy were pixel-
    aligned when they weren't (scaled/offset independently). Here the
    reveal image and x_noisy are genuinely co-registered — both derived
    from the exact same real target scene, with no independent transform
    applied to one but not the other.
    """

    condition_keys: list[str] = ["embedding_conditions", "x_noisy", "spatial_conditions"]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        latent_channels: int,
        hidden_size: int,
        grid_size: int,
        reveal_channels: int = 3,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels + reveal_channels, hidden_size, grid_size)
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        spatial_conditions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, N, out_dim)

        noisy_for_enc = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = x_noisy * keep[:, None, None, None]

        reveal_resized = spatial_conditions
        if spatial_conditions.shape[-2:] != noisy_for_enc.shape[-2:]:
            reveal_resized = F.interpolate(spatial_conditions, size=noisy_for_enc.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([noisy_for_enc, reveal_resized], dim=1)
        img_tokens = self.latent_encoder(combined)  # (B, G², hidden)

        enc_emb = torch.cat([obj_tokens, img_tokens], dim=1)
        if self.with_timestep_emb and timesteps is not None:
            enc_emb = enc_emb + self.timestep_mlp(timesteps).unsqueeze(1)
        return TRMInput(enc_emb=enc_emb)


class ObjectFeatureEncoderV1CentroidMaskFused(ConditionEncoderBase):
    """Channel-concat variant of ObjectFeatureEncoderV1CentroidMask: the
    centroid/attribute mask is concatenated channel-wise with the noisy
    latent (already the same spatial size by construction — mask_size =
    image_size // 8 matches the VAE's own downsampling) and run through ONE
    combined CNN, instead of a separate token block. Safe for the same
    reason as ObjectFeatureEncoderV1RevealFused: the mask and x_noisy are
    genuinely co-registered, both built from the same real scene.
    """

    condition_keys: list[str] = ["embedding_conditions", "x_noisy", "spatial_conditions"]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        latent_channels: int,
        hidden_size: int,
        grid_size: int,
        mask_channels: int = 16,
        noisy_dropout_p_max: float = 0.0,
        num_train_timesteps: int = 1000,
        with_timestep_emb: bool = False,
    ):
        super().__init__()
        self.object_encoder = ObjectFeatureEncoder(in_dim, hidden_dim, out_dim, with_timestep_emb=False)
        self.latent_encoder = ClevrLatentEncoder(latent_channels + mask_channels, hidden_size, grid_size)
        self.noisy_dropout_p_max = noisy_dropout_p_max
        self.num_train_timesteps = num_train_timesteps
        self.with_timestep_emb = with_timestep_emb
        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=out_dim)

    def forward(
        self,
        condition: torch.Tensor,
        x_noisy: torch.Tensor,
        spatial_conditions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> TRMInput:
        obj_tokens = self.object_encoder(condition).enc_emb  # (B, N, out_dim)

        noisy_for_enc = x_noisy
        if self.training and self.noisy_dropout_p_max > 0.0 and timesteps is not None:
            t_norm = timesteps.float() / self.num_train_timesteps
            p = self.noisy_dropout_p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy_for_enc = x_noisy * keep[:, None, None, None]

        mask_resized = spatial_conditions
        if spatial_conditions.shape[-2:] != noisy_for_enc.shape[-2:]:
            mask_resized = F.interpolate(spatial_conditions, size=noisy_for_enc.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([noisy_for_enc, mask_resized], dim=1)
        img_tokens = self.latent_encoder(combined)  # (B, G², hidden)

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
    """Slices the observation history down to the first n_obs_steps entries.

    Matches upstream diffusion_policy's lowdim policies: no learned encoder,
    the raw (normalized) observation history is used directly as conditioning
    (ConditionalUnet1D flattens it into one FiLM vector itself;
    TransformerForDiffusion consumes it as n_obs_steps separate cond tokens).

    embedding_conditions comes from the dataset sampled over the full action
    horizon (matching the `images` target length), not just the observable
    history — e.g. upstream's own dataset classes do the same, and it's the
    policy/model that slices `nobs[:, :n_obs_steps]` before conditioning, not
    the dataset. This encoder does that slicing.

    Input:  embedding_conditions (B, horizon, obs_dim)
    Output: (B, n_obs_steps, obs_dim)
    """

    condition_keys: list[str] = ["embedding_conditions"]

    def __init__(self, n_obs_steps: int):
        super().__init__()
        self.n_obs_steps = n_obs_steps

    def forward(self, embedding_conditions: torch.Tensor, timesteps=None, **_) -> torch.Tensor:
        return embedding_conditions[:, :self.n_obs_steps]


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

    spatial_conditions/embedding_conditions come from the dataset sampled over
    the full action horizon (matching the `images` target length), not just
    the observable history — it's the policy/model that slices
    `nobs[:, :n_obs_steps]` before conditioning, not the dataset. This encoder
    does that slicing (before running the vision encoder, so frames beyond
    n_obs_steps are never even encoded).

    Input:
      spatial_conditions   (B, horizon, C, H, W)     — single camera view, or
                            (B, horizon, V, C, H, W)  — V camera views (kept
                            separate, e.g. ToolHangImageDataset)
      embedding_conditions (B, horizon, obs_dim)     — optional raw low-dim obs
                            (e.g. agent_pos / robot proprioception)
    Output: (B, n_obs_steps, V * resnet_feature_dim + obs_dim)
    """

    condition_keys: list[str] = ["spatial_conditions", "embedding_conditions"]

    def __init__(self, n_obs_steps: int, use_group_norm: bool = True, pretrained: bool = False, crop_shape=None):
        super().__init__()
        import torchvision

        self.n_obs_steps = n_obs_steps
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
        x = spatial_conditions[:, :self.n_obs_steps]
        if embedding_conditions is not None:
            embedding_conditions = embedding_conditions[:, :self.n_obs_steps]
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


# ── Thinker-facing wrappers around the action-sequence encoders ──────────────
#
# The two encoders above return a plain (B, n_obs_steps, D) tensor for direct
# use as a painter's global_cond (models/action_painters.py). The TRM thinker
# instead expects a TRMInput (per ConditionEncoderBase's contract above) — this
# is a SEPARATE condition encoder from the frozen painter's own baked-in one
# (exactly like spatial_image_v0 is separate from the frozen MNIST DiT/UNet's
# own encoder in the existing thinker experiments), used only to give the
# thinker something to reason over. These wrappers compose the existing
# encoders rather than duplicating their tensor math.


class LowdimObsTRMConditionEncoder(ConditionEncoderBase):
    """Thinker-facing wrapper around LowdimObsConditionEncoder.

    SpatialTRMInner._input_embeddings treats a floating-point input as an
    already-computed embedding at (B, seq_len, hidden_size), added directly
    to z_H (also (B, seq_len, hidden_size)) — it does no projection or
    broadcasting of its own. seq_len here is the thinker's reasoning grid
    (= the action horizon, e.g. 16), which has no natural 1:1
    correspondence with the n_obs_steps (e.g. 2) observed history steps —
    there's no reason obs step i should align with output horizon position
    i. So instead of keeping per-obs-step tokens (which is what
    LowdimObsConditionEncoder itself does for FiLM/prefix-token painter
    conditioning), this flattens the whole obs history into a single
    conditioning vector — mirroring how ConditionalUnet1D's own FiLM
    conditioning already flattens obs history into one vector — and
    broadcasts that same vector to every reasoning position.

    Output: TRMInput with enc_emb (B, seq_len, hidden_size).
    """

    condition_keys: list[str] = ["embedding_conditions"]

    def __init__(self, n_obs_steps: int, in_dim: int, hidden_size: int, seq_len: int):
        super().__init__()
        self._inner = LowdimObsConditionEncoder(n_obs_steps)
        self.proj = nn.Linear(n_obs_steps * in_dim, hidden_size)
        self.seq_len = seq_len

    @property
    def n_obs_steps(self) -> int:
        return self._inner.n_obs_steps

    def forward(self, embedding_conditions: torch.Tensor, timesteps=None, **_) -> TRMInput:
        raw = self._inner(embedding_conditions, timesteps=timesteps)  # (B, n_obs_steps, in_dim)
        flat = raw.reshape(raw.shape[0], -1)  # (B, n_obs_steps * in_dim)
        token = self.proj(flat)  # (B, hidden_size)
        enc_emb = token.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq_len, hidden_size)
        return TRMInput(enc_emb=enc_emb)


class ImageObsTRMConditionEncoder(ConditionEncoderBase):
    """Thinker-facing wrapper around ImageObsConditionEncoder.

    SpatialTRMInner._input_embeddings treats a floating-point input as an
    already-computed embedding at (B, seq_len, hidden_size), added directly
    to z_H (also (B, seq_len, hidden_size)). seq_len here is the thinker's
    reasoning grid (= the action horizon, e.g. 16), which has no natural
    1:1 correspondence with the n_obs_steps (e.g. 2) observed history steps.
    So instead of keeping per-obs-step tokens (what ImageObsConditionEncoder
    itself produces for FiLM/prefix-token painter conditioning), this
    flattens the whole obs history into a single conditioning vector and
    broadcasts it to every reasoning position — same reasoning as
    LowdimObsTRMConditionEncoder.

    in_dim must be set to the per-obs-step feature width
    (V*resnet_feature_dim+obs_dim, task-specific since it depends on the
    number of camera views) — the Linear projection's actual input width is
    n_obs_steps*in_dim after flattening across obs steps.

    Output: TRMInput with enc_emb (B, seq_len, hidden_size).
    """

    condition_keys: list[str] = ["spatial_conditions", "embedding_conditions"]

    def __init__(
        self,
        n_obs_steps: int,
        in_dim: int,
        hidden_size: int,
        seq_len: int,
        use_group_norm: bool = True,
        pretrained: bool = False,
        crop_shape=None,
    ):
        super().__init__()
        self._inner = ImageObsConditionEncoder(
            n_obs_steps,
            use_group_norm=use_group_norm,
            pretrained=pretrained,
            crop_shape=crop_shape,
        )
        self.proj = nn.Linear(n_obs_steps * in_dim, hidden_size)
        self.seq_len = seq_len

    @property
    def n_obs_steps(self) -> int:
        return self._inner.n_obs_steps

    def forward(
        self,
        spatial_conditions: torch.Tensor,
        embedding_conditions=None,
        timesteps=None,
        **_,
    ) -> TRMInput:
        raw = self._inner(
            spatial_conditions,
            embedding_conditions=embedding_conditions,
            timesteps=timesteps,
        )  # (B, n_obs_steps, in_dim)
        flat = raw.reshape(raw.shape[0], -1)  # (B, n_obs_steps * in_dim)
        token = self.proj(flat)  # (B, hidden_size)
        enc_emb = token.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq_len, hidden_size)
        return TRMInput(enc_emb=enc_emb)
