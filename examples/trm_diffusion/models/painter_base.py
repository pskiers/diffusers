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
                frozen DiT; accepts cross-attn steering (CrossAttnSteering) from a translator
            └── IPAdapterSteeredDiTPainter
                    frozen DiT with per-block trainable IP cross-attention;
                    accepts IPAdapterSteering — adds thinker conditioning on top of
                    the frozen condition encoder rather than replacing it

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

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusers import UNet2DConditionModel, UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
from hydra.utils import instantiate

from configs.schemas import PainterOptimConfig, TrainConfig, EvalConfig
from models.utility_models import strip_compiled_prefix
from datasets.data_sample import DataSample
from models.base import BaseModel
from models.diffusion_utils import apply_noisy_swap
from models.interfaces import DiffusionPrediction, IPAdapterSteering, ThinkerSteering
from models.losses import LossBase, build_loss
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step

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

    def _batch_to_sample(self, batch: DataSample, device: torch.device) -> DataSample:
        """Build a static-condition DataSample from a batch for use in sampling.

        Copies all non-runtime DataSample fields (excludes x_noisy, timesteps,
        target, enc_x_noisy which are set per-step by the sampling loop).
        This ensures sample_grids can determine batch size even for unconditional
        models, and thinker models receive puzzle_id / solution / solution_mask.
        """
        _RUNTIME = {"x_noisy", "timesteps", "target", "enc_x_noisy"}
        kwargs: dict = {}
        for f in dataclasses.fields(DataSample):
            if f.name in _RUNTIME:
                continue
            val = batch.get(f.name) if hasattr(batch, "get") else None
            if val is not None:
                kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        return DataSample(**kwargs)


# ── Generic trainable UNet painter ───────────────────────────────────────────


class UNetPainter(PainterBase, BaseModel):
    """Standalone trainable UNet painter.

    Wraps a diffusers UNet (pixel-space or latent-space).  Pass a VAE to enable
    latent diffusion; then ``encode`` is available and ``_prepare_training_sample``
    encodes images to latents automatically.

    condition_keys is derived from condition_encoder.condition_keys at runtime
    (fields that will be zeroed for CFG dropout).  [] means unconditional.

    Args:
        unet:              Hydra config for a diffusers UNet (UNet2DModel etc.).
        scheduler:         diffusion noise scheduler (DDPMScheduler etc.).
        optim_cfg:         optimizer hyperparams.
        train_cfg:         training hyperparams.
        eval_cfg:          eval hyperparams.
        vae:               optional Hydra config for an AutoencoderKL.
        condition_encoder: optional Hydra config for a condition encoder;
                           must expose condition_keys and accept DataSample,
                           returning a dict of UNet kwargs.
        eval_callbacks:    optional list of Hydra configs for EvalCallbackBase.
    """

    def __init__(
        self,
        unet,
        scheduler,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        vae=None,
        condition_encoder=None,
        eval_callbacks=None,
        painter_dtype: Optional[str] = None,
        sampling_pipeline=None,
    ):
        super().__init__()
        self.unet: nn.Module = instantiate(unet)
        self.scheduler = scheduler
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None
        self.vae: Optional[nn.Module] = instantiate(vae) if vae is not None else None
        self.scaling_factor = self.vae.config.scaling_factor if self.vae is not None else 1.0
        self.condition_encoder: Optional[nn.Module] = (
            instantiate(condition_encoder) if condition_encoder is not None else None
        )
        self.eval_callbacks: list = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.loss_fn: LossBase = build_loss(train_cfg, scheduler)
        self.painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype]
            if painter_dtype is not None
            else None
        )

    @property
    def condition_keys(self) -> list[str]:
        return self.condition_encoder.condition_keys if self.condition_encoder is not None else []

    # ── Latent helpers ───────────────────────────────────────────────────────

    @property
    def noise_shape(self) -> tuple:
        """(C, H, W) shape of the tensor to generate (no batch dim)."""
        s = self.unet.config.sample_size
        c = self.unet.config.in_channels
        return (c, s, s)

    def decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → [0, 1] pixel images for logging."""
        if self.vae is not None:
            imgs = self.vae.decode(latents / self.scaling_factor).sample
            return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
        return latents.clamp(0.0, 1.0)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        """Convert dataset batch images → [0, 1] for display.

        Latent-space models receive [-1, 1] images from the dataset (VAE convention).
        Pixel-space models receive [0, 1] images directly.
        """
        if self.vae is not None:
            return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        return images.clamp(0.0, 1.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to scaled latents. Requires VAE."""
        if self.vae is None:
            raise RuntimeError("encode() requires a VAE")
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        kwargs: dict = {}
        if self.condition_encoder is not None:
            args = [getattr(sample, k) for k in self.condition_encoder.condition_keys]
            kwargs.update(self.condition_encoder(*args, timesteps=sample.timesteps).to_painter_kwargs(sample.embedding_mask))
        if steering is not None:
            kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.unet(sample.x_noisy, sample.timesteps, **kwargs).sample
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    # ── Training helpers ────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        """Build a DataSample with x_noisy, timesteps, and target from a batch.

        If the painter has a VAE, images are encoded to latents before adding
        noise.  All other batch fields are preserved so the condition encoder
        can read whatever it needs (e.g. embedding_conditions, embedding_mask).
        """
        images = mb.images.to(device)
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

        # Carry all fields from the input batch so the condition encoder can
        # read embedding_conditions, embedding_mask, etc.
        kwargs: dict = {}
        for f in dataclasses.fields(DataSample):
            if f.name in ("x_noisy", "timesteps", "target", "enc_x_noisy"):
                continue
            val = getattr(mb, f.name)
            if val is not None:
                kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        kwargs.update({"x_noisy": noisy, "timesteps": timesteps, "target": target})
        return DataSample(**kwargs)

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
        loss_sums: dict[str, float] = {}

        for mb in micro_batches:
            sample = self._prepare_training_sample(mb, device)

            if self.train_cfg.cfg_prob > 0:
                drop = torch.rand(sample.x_noisy.shape[0], device=device) < self.train_cfg.cfg_prob
                sample = self._apply_cfg_dropout(sample, drop)

            result = self(sample)
            total_loss, components = self.loss_fn(result.pred, result.logits, sample)
            accelerator.backward(total_loss / (global_batch_size * K))
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        accelerator.clip_grad_norm_(self.parameters(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)

        return {k: v / K for k, v in loss_sums.items()}, lr, global_step + 1

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()
        loss_sums: dict[str, float] = {}
        n_batches = 0
        dl_iter = iter(dataloader)
        for i in tqdm(range(max_batches), "Eval"):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            sample = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            _, components = self.loss_fn(result.pred, result.logits, sample)
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            n_batches += 1
        del dl_iter  # release workers before callbacks re-iterate the dataloader
        self.train()
        metrics = {k: v / n_batches for k, v in loss_sums.items()} if n_batches else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        return metrics

    def compile_submodules(self):
        self.unet = torch.compile(self.unet, fullgraph=False)
        if self.condition_encoder is not None:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)


# ── UNet2DModel with ControlNet residual injection ────────────────────────────


class ControlPainterUNet(UNet2DModel):
    """UNet2DModel extended with ControlNet-style residual injection.

    Identical to UNet2DModel except forward() accepts
    down_block_additional_residuals and mid_block_additional_residual, added to
    the skip connections before the up-blocks (standard ControlNet math).

    When all UNet parameters are frozen and the noisy input has no grad, the
    down/mid blocks produce no-grad tensors — PyTorch skips activation storage
    for those passes automatically.
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
        emb = self.time_embedding(self.time_proj(timestep).to(dtype=self.dtype))

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


# ── Frozen UNet wrapper for TRM steering ─────────────────────────────────────


class ControlNetSteeredUNetPainter(UNetPainter):
    """Frozen UNetPainter loaded from a checkpoint for ControlNet-style thinker steering.

    Builds the same architecture as the training run (same Hydra config), loads
    weights from a checkpoint produced by train_trm.py, and freezes the UNet and
    condition_encoder.  The unet is converted to ControlPainterUNet so that
    steering residuals can be passed directly as kwargs.

    Args:
        checkpoint: path to a checkpoint_*.pt file saved by train_trm.py
                    (must contain a "model_state" key).
        **kwargs:   forwarded verbatim to UNetPainter.__init__.
    """

    def __init__(self, checkpoint: str, **kwargs):
        super().__init__(**kwargs)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=True)
        # UNet2DConditionModel already accepts ControlNet residual kwargs natively.
        # Only swap to ControlPainterUNet for plain UNet2DModel (pixel-space UNets).
        if not isinstance(self.unet, UNet2DConditionModel):
            self.unet.__class__ = ControlPainterUNet
        for p in self.unet.parameters():
            p.requires_grad_(False)
        if self.condition_encoder is not None:
            for p in self.condition_encoder.parameters():
                p.requires_grad_(False)


# ── Generic trainable DiT painter ────────────────────────────────────────────


class DiTPainter(PainterBase, BaseModel):
    """Standalone trainable DiT painter for latent diffusion.

    Wraps a diffusers Transformer2DModel + AutoencoderKL.  Conditioning is
    routed through the condition_encoder, which accepts a DataSample and
    returns a dict of DiT kwargs (encoder_hidden_states, encoder_attention_mask).

    condition_keys is derived from condition_encoder.condition_keys at runtime
    (fields that will be zeroed for CFG dropout).  [] means unconditional.

    Args:
        dit:               Hydra config for a Transformer2DModel.
        vae:               Hydra config for an AutoencoderKL (always frozen).
        scheduler:         diffusion noise scheduler.
        optim_cfg:         optimizer hyperparams.
        train_cfg:         training hyperparams.
        eval_cfg:          eval hyperparams.
        condition_encoder: optional Hydra config for a condition encoder;
                           must expose condition_keys and accept DataSample,
                           returning a dict of DiT kwargs.
        eval_callbacks:    optional list of Hydra configs for EvalCallbackBase.
    """

    dit: nn.Module
    vae: nn.Module
    condition_encoder: Optional[nn.Module]
    eval_callbacks: list
    loss_fn: LossBase

    def __init__(
        self,
        dit,
        vae,
        scheduler,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        condition_encoder=None,
        eval_callbacks=None,
        painter_dtype: Optional[str] = None,
        vae_pixel_range: str = "[-1,1]",
        sampling_pipeline=None,
    ):
        super().__init__()
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None
        self.dit: nn.Module = instantiate(dit)
        self.vae: nn.Module = instantiate(vae)
        self.scaling_factor = self.vae.config.scaling_factor
        self.scheduler = scheduler
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.condition_encoder: Optional[nn.Module] = (
            instantiate(condition_encoder) if condition_encoder is not None else None
        )
        self.eval_callbacks: list = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.loss_fn: LossBase = build_loss(train_cfg, scheduler)
        self.painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype]
            if painter_dtype is not None
            else None
        )
        # "[-1,1]": VAE was trained on images in [-1,1] (standard SD convention).
        # "[0,1]":  VAE was trained on images in [0,1] (custom MNIST VAE).
        self._vae_tanh = vae_pixel_range == "[-1,1]"

        for p in self.vae.parameters():
            p.requires_grad_(False)

    @property
    def condition_keys(self) -> list[str]:
        return self.condition_encoder.condition_keys if self.condition_encoder is not None else []

    # ── Latent helpers ───────────────────────────────────────────────────────

    @property
    def noise_shape(self) -> tuple:
        """(C, H, W) shape of the tensor to generate (no batch dim)."""
        s = self.dit.config.sample_size
        c = self.dit.config.in_channels
        return (c, s, s)

    def decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → [0, 1] pixel images for logging."""
        pixels = self.vae.decode(latents / self.scaling_factor).sample
        if self._vae_tanh:
            return ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)
        return pixels.clamp(0.0, 1.0)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        """Convert dataset batch images → [0, 1] for display."""
        if self._vae_tanh:
            return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        return images.clamp(0.0, 1.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        kwargs: dict = {}
        if self.condition_encoder is not None:
            args = [getattr(sample, k) for k in self.condition_encoder.condition_keys]
            kwargs.update(self.condition_encoder(*args, timesteps=sample.timesteps).to_painter_kwargs(sample.embedding_mask))
        if steering is not None:
            kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.dit(sample.x_noisy, timestep=sample.timesteps, **kwargs).sample
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    # ── Training helpers ────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        """Build a DataSample with x_noisy, timesteps, and target from a batch.

        All other batch fields are preserved so the condition encoder can read
        embedding_conditions, embedding_mask, etc.
        """
        images = mb.images.to(device)
        bsz = images.shape[0]
        z = self.encode(images)

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy_z = self.scheduler.add_noise(z, noise, timesteps)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else z

        kwargs: dict = {}
        for f in dataclasses.fields(DataSample):
            if f.name in ("x_noisy", "timesteps", "target", "enc_x_noisy"):
                continue
            val = getattr(mb, f.name)
            if val is not None:
                kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        kwargs.update({"x_noisy": noisy_z, "timesteps": timesteps, "target": target})
        return DataSample(**kwargs)

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
        loss_sums: dict[str, float] = {}

        for mb in micro_batches:
            sample = self._prepare_training_sample(mb, device)

            if self.train_cfg.cfg_prob > 0:
                drop = torch.rand(sample.x_noisy.shape[0], device=device) < self.train_cfg.cfg_prob
                sample = self._apply_cfg_dropout(sample, drop)

            result = self(sample)
            total_loss, components = self.loss_fn(result.pred, result.logits, sample)
            accelerator.backward(total_loss / (global_batch_size * K))
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        accelerator.clip_grad_norm_([p for p in self.parameters() if p.requires_grad], 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)

        return {k: v / K for k, v in loss_sums.items()}, lr, global_step + 1

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()
        loss_sums: dict[str, float] = {}
        n_batches = 0
        dl_iter = iter(dataloader)
        for i in tqdm(range(max_batches), "Eval"):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            sample = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            _, components = self.loss_fn(result.pred, result.logits, sample)
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            n_batches += 1
        del dl_iter  # release workers before callbacks re-iterate the dataloader
        self.train()
        metrics = {k: v / n_batches for k, v in loss_sums.items()} if n_batches else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        return metrics

    def compile_submodules(self):
        self.dit = torch.compile(self.dit, fullgraph=False)
        if self.condition_encoder is not None:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)


# ── Frozen DiT wrapper for TRM steering ──────────────────────────────────────


class CrossAttnSteeredDiTPainter(DiTPainter):
    """Frozen DiTPainter loaded from a checkpoint for cross-attention thinker steering.

    Builds the same architecture as the training run (same Hydra config), loads
    weights from a checkpoint produced by train_trm.py, and freezes the DiT and
    condition_encoder.  The VAE remains frozen via the parent class.

    Can be instantiated directly from Hydra config — no factory wrapper needed.
    Thinker steering tokens are injected via ThinkerSteering in the inherited
    forward(), which also runs the condition_encoder if present.

    Args:
        checkpoint: path to a checkpoint_*.pt file saved by train_trm.py
                    (must contain a "model_state" key).
        **kwargs:   forwarded verbatim to DiTPainter.__init__.
    """

    def __init__(self, checkpoint: str, **kwargs):
        super().__init__(**kwargs)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=True)
        for p in self.dit.parameters():
            p.requires_grad_(False)
        if self.condition_encoder is not None:
            for p in self.condition_encoder.parameters():
                p.requires_grad_(False)


# ── IP-Adapter frozen DiT wrapper ─────────────────────────────────────────────


class _IPAdapterDiTBlock(nn.Module):
    """Wraps a single frozen DiT block with a trainable IP cross-attention layer.

    The inner block runs normally (scene-graph tokens via encoder_hidden_states).
    After the block, IP tokens from the thinker are injected via a separate
    Q/K/V attention whose output is added to the block output.

    Zero-init on to_out_ip means no effect at the start of training — the model
    begins identical to the frozen painter.

    ip_hidden_states is passed in via cross_attention_kwargs["ip_hidden_states"]
    so Transformer2DModel threads it to each block without modification.
    The dict is never mutated; a new dict is created for the inner block call.
    """

    def __init__(self, block: nn.Module, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.block = block
        self._n_heads = n_heads
        self.to_k_ip = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v_ip = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out_ip = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.to_out_ip.weight)

    def _ip_attn(self, hidden_states: torch.Tensor, ip_hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden_states.shape
        N = ip_hidden_states.shape[1]
        H, d = self._n_heads, D // self._n_heads

        q = hidden_states.view(B, S, H, d).transpose(1, 2)  # (B, H, S, d)
        k = self.to_k_ip(ip_hidden_states).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        v = self.to_v_ip(ip_hidden_states).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, d)
        return self.to_out_ip(out.transpose(1, 2).reshape(B, S, D))

    def forward(self, hidden_states: torch.Tensor, cross_attention_kwargs=None, **kwargs) -> torch.Tensor:
        ip_hs = None
        inner_cak = cross_attention_kwargs
        if cross_attention_kwargs is not None and "ip_hidden_states" in cross_attention_kwargs:
            ip_hs = cross_attention_kwargs["ip_hidden_states"]
            inner_cak = {k: v for k, v in cross_attention_kwargs.items() if k != "ip_hidden_states"} or None

        out = self.block(hidden_states, cross_attention_kwargs=inner_cak, **kwargs)
        if ip_hs is not None:
            out = out + self._ip_attn(out, ip_hs)
        return out


class IPAdapterSteeredDiTPainter(CrossAttnSteeredDiTPainter):
    """Frozen DiTPainter with per-block IP-Adapter conditioning.

    Extends CrossAttnSteeredDiTPainter by replacing each transformer block
    with an _IPAdapterDiTBlock that adds a trainable IP cross-attention on top.

    The frozen condition encoder still runs and provides scene-graph tokens via
    the normal cross-attention.  IP tokens from the thinker (via IPAdapterSteering)
    are injected additively at every block — adding thinker conditioning on top
    of the base conditioning rather than replacing it.

    Trainable parameters: to_k_ip / to_v_ip / to_out_ip per block (3 linears
    × num_blocks).  Everything else is frozen from the loaded checkpoint.

    torch.compile compatible: no forward hooks; all tensor ops are inside nn.Module
    calls.  The dict extraction in _IPAdapterDiTBlock.forward creates a graph break
    per block under fullgraph=False, but the IP attention itself compiles cleanly.

    Args:
        checkpoint: forwarded to CrossAttnSteeredDiTPainter (checkpoint path).
        **kwargs:   forwarded verbatim to CrossAttnSteeredDiTPainter.__init__.
    """

    def __init__(self, checkpoint: str, **kwargs) -> None:
        super().__init__(checkpoint=checkpoint, **kwargs)
        H = self.dit.config.num_attention_heads
        D = H * self.dit.config.attention_head_dim
        self.dit.transformer_blocks = nn.ModuleList(
            [_IPAdapterDiTBlock(blk, D, H) for blk in self.dit.transformer_blocks]
        )
        # Snapshot the DiT's patch positional embeddings (1, N_patches, D).
        # Added to IP tokens in forward() so cross-attention has the same spatial
        # coordinates as the DiT queries — cells near patch (i,j) naturally attend
        # to the token for cell (i,j) without needing CE supervision to learn it.
        self.register_buffer(
            "_ip_pos_embed",
            self.dit.pos_embed.pos_embed.detach().clone(),
        )

    def forward(self, sample: DataSample, steering=None) -> DiffusionPrediction:
        if steering is not None and isinstance(steering, IPAdapterSteering):
            pos = self._ip_pos_embed.to(steering.ip_hidden_states.dtype)
            steering = IPAdapterSteering(
                ip_hidden_states=steering.ip_hidden_states + pos
            )
        return super().forward(sample, steering)
