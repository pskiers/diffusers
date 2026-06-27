"""
models/clevr_painters.py — Latent DiT painter for CLEVR scene generation.

Two-stage training:
  1. Train StandaloneClevrDiT on CLEVR (relative mode) → ~90% relation accuracy.
  2. Freeze this model, train ThinkerWithFrozenClevrDiTV0/V1 on reduced mode.

Architecture:
  - Frozen SDXL VAE (stabilityai/sd-vae-ft-mse): 256x256 RGB → 4x32x32 latents.
  - Transformer2DModel DiT (PixArt ada_norm_single): denoises in latent space.
  - ClevrObjectEncoder MLP: (B, max_objects, object_feat_dim) → (B, max_objects, cond_embed_dim)
    for DiT cross-attention.  Zero-init output so null condition = all-zeros.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, Transformer2DModel
from tqdm.auto import tqdm

from configs.schemas import ClevrDiTConfig, ClevrDiTOptimConfig, EvalConfig, TrainConfig
from datasets.data_sample import DataSample
from models.base import BaseModel
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.utility_models import TimestepMLP

# ── Object encoder ────────────────────────────────────────────────────────────


class ClevrObjectEncoder(nn.Module):
    """
    MLP that maps raw per-object feature vectors to DiT cross-attention tokens.

    Input:  DataSample with embedding_conditions (B, max_objects, object_feat_dim)
            and optional embedding_mask (B, max_objects).
    Output: dict with "encoder_hidden_states" and optionally "encoder_attention_mask".

    The final linear is zero-initialised so the null conditioning signal
    (all-zero input) starts out exactly zero — meaning CFG dropout at the
    start of training produces well-defined unconditioned predictions.

    condition_keys = ["embedding_conditions"] — only this field is zeroed for
    CFG dropout; embedding_mask is structural and is NOT zeroed.
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

    def forward(self, sample: DataSample) -> dict:
        hidden = self.net(sample.embedding_conditions)
        if self.with_timestep_emb and sample.timesteps is not None:
            hidden = hidden + self.timestep_mlp(sample.timesteps).unsqueeze(1)
        result: dict = {"encoder_hidden_states": hidden}
        if sample.embedding_mask is not None:
            result["encoder_attention_mask"] = sample.embedding_mask.bool()
        return result


# ── Standalone CLEVR DiT painter ──────────────────────────────────────────────


class StandaloneClevrDiT(BaseModel):
    """
    Latent diffusion model for CLEVR scene generation.

    The VAE is loaded from a public HuggingFace hub checkpoint and frozen.
    The DiT denoises in VAE latent space, conditioned on per-object embeddings
    via cross-attention.

    For CFG training, the entire condition sequence is dropped with probability
    train_cfg.cfg_prob (replaced with zeros).  At inference the null condition
    is torch.zeros(B, max_objects, cond_embed_dim).

    Object slots that are padding (mask=0) are excluded from cross-attention
    via encoder_attention_mask.
    """

    def __init__(
        self,
        model_cfg: ClevrDiTConfig,
        optim_cfg: ClevrDiTOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        scheduler: Any,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.scheduler = scheduler

        # ── VAE (frozen) ───────────────────────────────────────────────────────
        vae = AutoencoderKL.from_pretrained(model_cfg.vae_model_name)
        self.vae = vae
        self.scaling_factor: float = float(vae.config.scaling_factor)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # ── DiT backbone ───────────────────────────────────────────────────────
        inner_dim = model_cfg.num_attention_heads * model_cfg.attention_head_dim
        self.dit = Transformer2DModel(
            num_attention_heads=model_cfg.num_attention_heads,
            attention_head_dim=model_cfg.attention_head_dim,
            in_channels=model_cfg.latent_channels,
            out_channels=model_cfg.latent_channels,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            cross_attention_dim=inner_dim,
            sample_size=model_cfg.latent_size,
            patch_size=model_cfg.patch_size,
            norm_type="ada_norm_single",
            attention_bias=True,
            activation_fn="gelu-approximate",
            caption_channels=model_cfg.cond_embed_dim,
        )

        # ── Condition encoder ──────────────────────────────────────────────────
        self.object_encoder = ClevrObjectEncoder(
            in_dim=model_cfg.object_feat_dim,
            hidden_dim=model_cfg.cond_embed_dim,
            out_dim=model_cfg.cond_embed_dim,
        )

    # ── VAE helpers ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) ∈ [-1,1] → scaled latents (B, 4, H/8, W/8)."""
        return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latents → pixel images (B, 3, H, W) ∈ [0,1]."""
        return ((self.vae.decode(z / self.scaling_factor).sample + 1.0) / 2.0).clamp(0.0, 1.0)

    # ── Optimizer ──────────────────────────────────────────────────────────────

    def build_optimizers(self, world_size: int, num_steps: int) -> list:
        params = list(self.dit.parameters()) + list(self.object_encoder.parameters())
        optim = torch.optim.AdamW(params, lr=0, weight_decay=self.optim_cfg.weight_decay)
        return [
            ScheduledOptimizer(
                optim,
                base_lr=self.optim_cfg.lr,
                warmup_steps=self.optim_cfg.warmup_steps,
                num_steps=num_steps,
                min_ratio=self.optim_cfg.lr_min_ratio,
            )
        ]

    def _trainable_params(self):
        return list(self.dit.parameters()) + list(self.object_encoder.parameters())

    def compile_submodules(self):
        self.dit = torch.compile(self.dit)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: torch.Tensor,  # (B, max_objects, object_feat_dim)
        masks: Optional[torch.Tensor] = None,  # (B, max_objects) 1=real 0=pad
    ) -> torch.Tensor:
        cond_emb = self.object_encoder(conditions)  # (B, max_objects, cond_embed_dim)

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(cond_emb.shape[0], device=cond_emb.device) < self.train_cfg.cfg_prob
            cond_emb = cond_emb * (~drop[:, None, None])
            if masks is not None:
                masks = masks * (~drop[:, None]).float()

        attn_mask = masks.bool() if masks is not None else None
        noise_pred = self.dit(
            noisy_z,
            timestep=timesteps,
            encoder_hidden_states=cond_emb,
            encoder_attention_mask=attn_mask,
        ).sample
        return noise_pred

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step, **kwargs):
        K = len(micro_batches)
        device = accelerator.device
        total_loss = 0.0

        for mb in micro_batches:
            images = mb["images"].to(device)
            conditions = mb["embedding_conditions"].to(device)
            masks = mb["embedding_mask"].to(device) if "embedding_mask" in mb else None

            z = self._encode(images)
            B = z.shape[0]
            noise = torch.randn_like(z)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long
            )
            noisy_z = self.scheduler.add_noise(z, noise, timesteps)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z

            noise_pred = self(noisy_z, timesteps, conditions, masks)
            loss = F.mse_loss(noise_pred.float(), target.float())
            total_loss += loss.item()
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self._trainable_params(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)
        global_step += 1

        return {"diff_loss": total_loss / K}, lr, global_step

    # ── Eval step ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        step = kwargs.get("step", None)
        max_batches = kwargs.get("max_batches", 50)
        device = accelerator.device
        self.eval()

        # Validation loss
        val_losses = []
        for i, batch in enumerate(tqdm(dataloader, desc="Eval loss", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            conditions = batch["embedding_conditions"].to(device)
            masks = batch["embedding_mask"].to(device) if "embedding_mask" in batch else None
            z = self._encode(images)
            B = z.shape[0]
            noise = torch.randn_like(z)
            ts = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
            noisy_z = self.scheduler.add_noise(z, noise, ts)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z
            noise_pred = self(noisy_z, ts, conditions, masks)
            val_losses.append(F.mse_loss(noise_pred.float(), target.float()).item())

        result = {"diff_loss": float(np.mean(val_losses))} if val_losses else {}

        # Sample a few images for visual inspection (main process only)
        if accelerator.is_main_process:
            ddim = DDIMScheduler(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                beta_schedule=self.scheduler.config.beta_schedule,
                prediction_type=self.scheduler.config.prediction_type,
            )
            ddim.set_timesteps(self.eval_cfg.num_ddim_steps)
            cfg_scale = self.eval_cfg.cfg_scale

            sample_images = []
            n_log = self.eval_cfg.num_log_images
            for batch in dataloader:
                if len(sample_images) >= n_log:
                    break
                conditions = batch["embedding_conditions"].to(device)
                masks = batch["embedding_mask"].to(device) if "embedding_mask" in batch else None
                B = conditions.shape[0]
                B = min(B, n_log - len(sample_images))
                conditions = conditions[:B]
                masks = masks[:B] if masks is not None else None

                z = torch.randn(
                    B,
                    self.model_cfg.latent_channels,
                    self.model_cfg.latent_size,
                    self.model_cfg.latent_size,
                    device=device,
                )
                null_cond = torch.zeros_like(conditions)

                for t in ddim.timesteps:
                    ts = torch.full((B,), t, device=device, dtype=torch.long)
                    noise_pred = self(z, ts, conditions, masks)
                    if cfg_scale > 1.0:
                        noise_uncond = self(z, ts, null_cond, None)
                        noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
                    z = ddim.step(noise_pred, t, z).prev_sample

                imgs = self._decode(z)  # (B, 3, H, W)
                sample_images.extend(imgs.cpu().unbind(0))

            if sample_images and step is not None:
                try:
                    import wandb
                    from torchvision.utils import make_grid

                    grid = make_grid(torch.stack(sample_images[:n_log]), nrow=4)
                    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    tracker = accelerator.get_tracker("wandb", unwrap=True)
                    tracker.log({"val/samples": wandb.Image(grid_np)}, step=step)
                except Exception:
                    pass

        self.train()
        return result
