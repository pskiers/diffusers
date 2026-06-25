"""
models/painter_base.py — Generic painter base classes with DataSample interface.

Class hierarchy:

    PainterBase (abstract)
    ├── UNetPainter            standalone trainable UNet painter (pixel or latent)
    │   └── ControlNetSteeredUNetPainter
    │           frozen UNet from a trained checkpoint; steering residuals injected
    │           from an external ControlNetTranslator — no trainable components here
    └── DiTPainter             standalone trainable DiT painter (latent)
        └── CrossAttnSteeredDiTPainter
                frozen DiT; accepts cross-attn steering from a translator
                subclass to add IP-Adapter per-block attn (trainable components)

Training workflow:
    1. Train UNetPainter (or DiTPainter) standalone.
    2. Load checkpoint → pass to ControlNetSteeredUNetPainter (or CrossAttnSteeredDiTPainter).
    3. Pass steered painter to ThinkerFrozenPainterBase.

All painters implement:
    forward(sample: DataSample, steering: Optional[ThinkerSteering] = None)
        -> DiffusionPrediction
    null_condition_sample(sample: DataSample) -> DataSample

Latent-space painters additionally expose:
    encode(images: Tensor) -> latents
    decode(latents: Tensor) -> images
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets.data_sample import DataSample
from models.base import BaseModel
from models.diffusion_utils import apply_noisy_swap
from models.interfaces import DiffusionPrediction, ThinkerSteering
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from configs.schemas import PainterOptimConfig, TrainConfig, EvalConfig

# ── Abstract base ─────────────────────────────────────────────────────────────


class PainterBase(nn.Module):
    """Common interface for all painter classes.

    condition_keys declares which DataSample fields the painter reads as
    its own direct conditioning (separate from ThinkerSteering which comes
    in via the `steering` argument).  Default [] means unconditional.

    Set ``self.condition_keys = [...]`` in __init__ to override per instance.
    """

    condition_keys: list[str] = []

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        raise NotImplementedError

    def null_condition_sample(self, sample: DataSample) -> DataSample:
        """Return a copy with all condition_keys fields zeroed for CFG uncond pass."""
        updates = {
            k: torch.zeros_like(getattr(sample, k)) for k in self.condition_keys if getattr(sample, k, None) is not None
        }
        return dataclasses.replace(sample, **updates)


# ── Generic trainable UNet painter ───────────────────────────────────────────


class UNetPainter(PainterBase, BaseModel):
    """Standalone trainable UNet painter.

    Wraps a pre-built diffusers UNet (pixel-space or latent-space).  Pass a VAE
    to enable latent diffusion; then ``encode`` / ``decode`` are available and
    ``_prepare_training_sample`` encodes images to latents automatically.

    condition_keys = [] (unconditional) by default.  Subclasses that need
    cross-attention or other UNet conditioning:
      - set ``self.condition_keys = [<field>, ...]`` in __init__
      - override ``_build_unet_kwargs(sample)`` to extract those fields

    Args:
        unet:            pre-built diffusers UNet (UNet2DModel etc.).
        scheduler:       diffusion noise scheduler (DDPMScheduler etc.).
        optim_cfg:       optimizer hyperparams.
        train_cfg:       training hyperparams (cfg_prob, mse_loss_weight, …).
        eval_cfg:        eval hyperparams.
        vae:             optional AutoencoderKL for latent diffusion.
        scaling_factor:  VAE scaling factor (typically 0.18215 for SDXL VAE).
    """

    condition_keys: list[str] = []

    def __init__(
        self,
        unet: nn.Module,
        scheduler,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        vae: Optional[nn.Module] = None,
        scaling_factor: float = 0.18215,
        condition_encoder: Optional[nn.Module] = None,
        condition_field: str = "embedding_conditions",
        eval_callbacks: Optional[list] = None,
    ):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.condition_encoder = condition_encoder
        self.condition_field = condition_field
        if condition_encoder is not None:
            self.condition_keys = [condition_field]
        self.eval_callbacks: list = eval_callbacks or []

    # ── Latent helpers ───────────────────────────────────────────────────────

    @property
    def _noise_shape(self) -> tuple:
        """(C, H, W) shape of the tensor to generate (no batch dim)."""
        s = self.unet.config.sample_size
        c = self.unet.config.in_channels
        return (c, s, s)

    def _decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → [0, 1] pixel images for logging."""
        if self.vae is not None:
            imgs = self.vae.decode(latents / self.scaling_factor).sample
            return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
        return latents.clamp(0.0, 1.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to scaled latents. Requires VAE."""
        if self.vae is None:
            raise RuntimeError("encode() requires a VAE")
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents to pixel images. Requires VAE."""
        if self.vae is None:
            raise RuntimeError("decode() requires a VAE")
        return self.vae.decode(latents / self.scaling_factor).sample

    # ── UNet conditioning override point ────────────────────────────────────

    def _build_unet_kwargs(self, sample: DataSample) -> dict:
        """Build extra kwargs for unet.forward from DataSample condition fields.

        When condition_encoder is set at init, automatically encodes
        sample.<condition_field> → encoder_hidden_states.
        Override in subclasses for custom conditioning logic.
        """
        if self.condition_encoder is not None:
            val = getattr(sample, self.condition_field)
            return {"encoder_hidden_states": self.condition_encoder(val)}
        return {}

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        unet_kwargs = self._build_unet_kwargs(sample)
        if steering is not None:
            unet_kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.unet(sample.x_noisy, sample.timesteps, **unet_kwargs).sample
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    # ── Training helpers ────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: dict, device: torch.device) -> tuple[DataSample, torch.Tensor]:
        """Build (DataSample, target) from a batch dict.

        If the painter has a VAE, images are encoded to latents before adding
        noise.  Override in subclasses for custom preprocessing.
        """
        images = mb["images"].to(device)
        bsz = images.shape[0]

        if self.vae is not None:
            images = self.encode(images)  # pixel → latent (no_grad inside encode)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy = self.scheduler.add_noise(images, noise, timesteps)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else images

        noisy, target = apply_noisy_swap(
            images=images,
            noisy=noisy,
            target=target,
            timesteps=timesteps,
            scheduler=self.scheduler,
            swap_cfg=self.train_cfg.noisy_swap,
        )

        sample_kwargs: dict = {"x_noisy": noisy, "timesteps": timesteps}
        for k in self.condition_keys:
            if k in mb and k not in ("x_noisy", "timesteps"):
                sample_kwargs[k] = mb[k].to(device)

        return DataSample(**sample_kwargs), target

    def _apply_cfg_dropout(self, sample: DataSample, drop: torch.Tensor) -> DataSample:
        """Zero condition fields for samples where drop=True (CFG training)."""
        updates: dict = {}
        for k in self.condition_keys:
            val = getattr(sample, k, None)
            if val is not None:
                mask = drop.view((drop.shape[0],) + (1,) * (val.ndim - 1)).to(val.device)
                updates[k] = val * (~mask).to(val.dtype)
        return dataclasses.replace(sample, **updates) if updates else sample

    # ── Training / eval / optimizer ─────────────────────────────────────────

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        optim = torch.optim.AdamW(self.parameters(), lr=0, weight_decay=self.optim_cfg.weight_decay)
        return [
            ScheduledOptimizer(
                optim,
                base_lr=self.optim_cfg.lr,
                warmup_steps=self.optim_cfg.warmup_steps,
                num_steps=num_steps,
                min_ratio=self.optim_cfg.lr_min_ratio,
            )
        ]

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size,
        global_step,
        **kwargs,
    ):
        K = len(micro_batches)
        device = accelerator.device
        total_diff_loss = 0.0

        for mb in micro_batches:
            sample, target = self._prepare_training_sample(mb, device)

            if self.train_cfg.cfg_prob > 0:
                drop = torch.rand(sample.x_noisy.shape[0], device=device) < self.train_cfg.cfg_prob
                sample = self._apply_cfg_dropout(sample, drop)

            result = self(sample)
            diff_loss = F.mse_loss(result.pred.float(), target.float())
            accelerator.backward(self.train_cfg.mse_loss_weight * diff_loss / (global_batch_size * K))
            total_diff_loss += diff_loss.item()

        accelerator.clip_grad_norm_(self.parameters(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)

        return {"diff_loss": total_diff_loss / K}, lr, global_step + 1

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()
        diff_losses = []
        for i, batch in tqdm(enumerate(dataloader), "Eval", total=max_batches):
            if i >= max_batches:
                break
            sample, target = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            diff_losses.append(F.mse_loss(result.pred.float(), target.float()).item())
        self.train()
        metrics = {"diff_loss": float(np.mean(diff_losses))} if diff_losses else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        return metrics

    def compile_submodules(self):
        self.unet = torch.compile(self.unet, fullgraph=False)
        if self.condition_encoder is not None:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)


# ── Frozen UNet wrapper for TRM steering ─────────────────────────────────────


class ControlNetSteeredUNetPainter(PainterBase):
    """Frozen UNet wrapper that receives ControlNet-style residuals from a translator.

    Created from a trained UNetPainter: deep-copies and freezes its UNet.
    The VAE (if any) is shared without copy — it was already frozen during
    UNetPainter training and is not re-trained here.

    No trainable components are added.  All steering capacity lives in the
    external ControlNetTranslator (its ConditioningPyramid).

    condition_keys is copied from the source painter at init time so that
    null_condition_sample() can zero the right DataSample fields for CFG.

    Usage::

        steered = ControlNetSteeredUNetPainter(trained_painter)
        model = ThinkerWithFrozenPainterControlNetV2(painter=steered, ...)
    """

    def __init__(self, painter: UNetPainter):
        super().__init__()

        # Deep-copy so the steered painter owns independent frozen weights.
        self.unet = copy.deepcopy(painter.unet)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # Share (not copy) the frozen VAE.
        self.vae = painter.vae
        if self.vae is not None:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        self.scaling_factor = painter.scaling_factor
        self._prediction_type = painter.scheduler.config.prediction_type
        # Snapshot so the source painter can be GC'd after init.
        self.condition_keys = list(painter.condition_keys)

    # ── Latent helpers ───────────────────────────────────────────────────────

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("encode() requires a VAE")
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("decode() requires a VAE")
        return self.vae.decode(latents / self.scaling_factor).sample

    def _build_unet_kwargs(self, sample: DataSample) -> dict:
        """Build extra kwargs for unet.forward from DataSample.

        Default: empty dict (unconditional, or conditioning via steering only).
        Override in subclasses when the frozen UNet needs additional conditioning
        beyond ControlNet residuals (e.g. cross-attention text tokens).
        """
        return {}

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        unet_kwargs = self._build_unet_kwargs(sample)
        if steering is not None:
            unet_kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.unet(sample.x_noisy, sample.timesteps, **unet_kwargs).sample
        return DiffusionPrediction(pred=noise_pred, pred_type=self._prediction_type)

    def compile_submodules(self):
        self.unet = torch.compile(self.unet, fullgraph=False)


# ── Generic trainable DiT painter ────────────────────────────────────────────


class DiTPainter(PainterBase, BaseModel):
    """Standalone trainable DiT painter for latent diffusion.

    Wraps a diffusers Transformer2DModel + AutoencoderKL.  Conditioning is
    routed through cross-attention (encoder_hidden_states); override
    ``_build_encoder_hidden_states`` in subclasses for conditional models.

    condition_keys = [] (unconditional) by default.  Set
    ``self.condition_keys = [...]`` and override
    ``_build_encoder_hidden_states(sample)`` for conditional DiTs.

    Args:
        dit:             Transformer2DModel (or compatible DiT backbone).
        vae:             AutoencoderKL for encoding/decoding latents.
        scaling_factor:  VAE scaling factor.
        scheduler:       diffusion noise scheduler.
        optim_cfg:       optimizer hyperparams.
        train_cfg:       training hyperparams.
        eval_cfg:        eval hyperparams.
    """

    condition_keys: list[str] = []

    def __init__(
        self,
        dit: nn.Module,
        vae: nn.Module,
        scaling_factor: float,
        scheduler,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        condition_encoder: Optional[nn.Module] = None,
        condition_field: str = "embedding_conditions",
        eval_callbacks: Optional[list] = None,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.scheduler = scheduler
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.condition_encoder = condition_encoder
        self.condition_field = condition_field
        if condition_encoder is not None:
            self.condition_keys = [condition_field]
        self.eval_callbacks: list = eval_callbacks or []

        for p in self.vae.parameters():
            p.requires_grad_(False)

    # ── Latent helpers ───────────────────────────────────────────────────────

    @property
    def _noise_shape(self) -> tuple:
        """(C, H, W) shape of the tensor to generate (no batch dim)."""
        s = self.dit.config.sample_size
        c = self.dit.config.in_channels
        return (c, s, s)

    def _decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → [0, 1] pixel images for logging."""
        return self.decode(latents)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → [0, 1] pixel images."""
        return ((self.vae.decode(latents / self.scaling_factor).sample + 1.0) / 2.0).clamp(0.0, 1.0)

    # ── Conditioning override point ──────────────────────────────────────────

    def _build_encoder_hidden_states(self, sample: DataSample) -> Optional[torch.Tensor]:
        """Build encoder_hidden_states for DiT cross-attention from DataSample.

        When condition_encoder is set at init, automatically encodes
        sample.<condition_field>.  Override for custom conditioning logic.
        """
        if self.condition_encoder is not None:
            val = getattr(sample, self.condition_field)
            return self.condition_encoder(val)
        return None

    def _build_encoder_attention_mask(self, sample: DataSample) -> Optional[torch.Tensor]:
        """Build encoder_attention_mask for DiT cross-attention from DataSample.

        When condition_encoder is set, uses sample.embedding_mask (if present).
        Override for custom masking logic.
        """
        if self.condition_encoder is not None and sample.embedding_mask is not None:
            return sample.embedding_mask.bool()
        return None

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        kwargs: dict = {}
        enc_hs = self._build_encoder_hidden_states(sample)
        if enc_hs is not None:
            kwargs["encoder_hidden_states"] = enc_hs
        enc_mask = self._build_encoder_attention_mask(sample)
        if enc_mask is not None:
            kwargs["encoder_attention_mask"] = enc_mask
        if steering is not None:
            kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.dit(sample.x_noisy, timestep=sample.timesteps, **kwargs).sample
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    # ── Training helpers ────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: dict, device: torch.device) -> tuple[DataSample, torch.Tensor]:
        images = mb["images"].to(device)
        bsz = images.shape[0]
        z = self.encode(images)

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy_z = self.scheduler.add_noise(z, noise, timesteps)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else z

        sample_kwargs: dict = {"x_noisy": noisy_z, "timesteps": timesteps}
        for k in self.condition_keys:
            if k in mb and k not in ("x_noisy", "timesteps"):
                sample_kwargs[k] = mb[k].to(device)

        return DataSample(**sample_kwargs), target

    def _apply_cfg_dropout(self, sample: DataSample, drop: torch.Tensor) -> DataSample:
        updates: dict = {}
        for k in self.condition_keys:
            val = getattr(sample, k, None)
            if val is not None:
                mask = drop.view((drop.shape[0],) + (1,) * (val.ndim - 1)).to(val.device)
                updates[k] = val * (~mask).to(val.dtype)
        return dataclasses.replace(sample, **updates) if updates else sample

    # ── Training / eval / optimizer ─────────────────────────────────────────

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        trainable = [p for p in self.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=0, weight_decay=self.optim_cfg.weight_decay)
        return [
            ScheduledOptimizer(
                optim,
                base_lr=self.optim_cfg.lr,
                warmup_steps=self.optim_cfg.warmup_steps,
                num_steps=num_steps,
                min_ratio=self.optim_cfg.lr_min_ratio,
            )
        ]

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size,
        global_step,
        **kwargs,
    ):
        K = len(micro_batches)
        device = accelerator.device
        total_diff_loss = 0.0

        for mb in micro_batches:
            sample, target = self._prepare_training_sample(mb, device)

            if self.train_cfg.cfg_prob > 0:
                drop = torch.rand(sample.x_noisy.shape[0], device=device) < self.train_cfg.cfg_prob
                sample = self._apply_cfg_dropout(sample, drop)

            result = self(sample)
            diff_loss = F.mse_loss(result.pred.float(), target.float())
            accelerator.backward(diff_loss / (global_batch_size * K))
            total_diff_loss += diff_loss.item()

        accelerator.clip_grad_norm_([p for p in self.parameters() if p.requires_grad], 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)

        return {"diff_loss": total_diff_loss / K}, lr, global_step + 1

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()
        diff_losses = []
        for i, batch in tqdm(enumerate(dataloader), "Eval", total=max_batches):
            if i >= max_batches:
                break
            sample, target = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            diff_losses.append(F.mse_loss(result.pred.float(), target.float()).item())
        self.train()
        metrics = {"diff_loss": float(np.mean(diff_losses))} if diff_losses else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        return metrics

    def compile_submodules(self):
        self.dit = torch.compile(self.dit, fullgraph=False)
        if self.condition_encoder is not None:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)


# ── Frozen DiT wrapper for TRM steering ──────────────────────────────────────


class CrossAttnSteeredDiTPainter(PainterBase):
    """Frozen DiT wrapper that receives cross-attention steering from a translator.

    Wraps a trained DiTPainter: deep-copies and freezes the DiT.  The VAE is
    shared without copy (always frozen).

    The translator produces a ThinkerSteering whose to_painter_kwargs() returns
    ``{"encoder_hidden_states": <tokens>}``, which is passed to the frozen DiT.

    For IP-Adapter per-block conditioning, subclass and:
      1. Add ip_cross_attn ModuleList and any projection heads (trainable).
      2. Register forward hooks on dit.transformer_blocks in __init__.
      3. Use steering to pass the IP-Adapter KV tokens to the hooks.

    condition_keys is proxied from the source DiTPainter.
    """

    def __init__(self, painter: DiTPainter):
        super().__init__()

        self.dit = copy.deepcopy(painter.dit)
        for p in self.dit.parameters():
            p.requires_grad_(False)

        self.vae = painter.vae
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.scaling_factor = painter.scaling_factor
        self._prediction_type = painter.scheduler.config.prediction_type
        self.condition_keys = list(painter.condition_keys)

    # ── Latent helpers ───────────────────────────────────────────────────────

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return ((self.vae.decode(latents / self.scaling_factor).sample + 1.0) / 2.0).clamp(0.0, 1.0)

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        kwargs: dict = {}
        if steering is not None:
            kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.dit(sample.x_noisy, timestep=sample.timesteps, **kwargs).sample
        return DiffusionPrediction(pred=noise_pred, pred_type=self._prediction_type)

    def compile_submodules(self):
        self.dit = torch.compile(self.dit, fullgraph=False)
