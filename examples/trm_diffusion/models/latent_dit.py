"""
models/latent_dit.py — Latent Diffusion Transformer for MNIST Sudoku.

Uses diffusers' Transformer2DModel (PixArt-Alpha style: ada_norm_single + cross-attention)
as the denoising backbone, operating on VAE-encoded latents.

Conditioning: 81 sudoku tokens → ConditionEncoder → (B, 81, cond_embed_dim) →
              Transformer2DModel cross-attention via caption_channels projection.

patch_size is a pure config knob.  Cross-attention means ConditionEncoder output
adapts to any patch count without architectural changes — replacing ConditionEncoder
with a CNN or other encoder also requires no changes to the DiT backbone.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, Transformer2DModel
from tqdm.auto import tqdm

from configs.schemas import EvalConfig, LatentDiTConfig, LatentDiTOptimConfig, TrainConfig
from datasets.mnist_sudoku_dataset import get_solution_tokens
from eval.mnist_eval import evaluate_grids, load_or_train_classifier, make_panel_image
from models.base import BaseModel
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step

# ── Condition encoder ──────────────────────────────────────────────────────────


class ConditionEncoder(nn.Module):
    """
    Maps 81 sudoku token indices → (B, 81, out_dim) for DiT cross-attention.

    Token vocabulary: 0 = null/dropout, 1 = blank cell, 2-10 = given digits 1-9.
    padding_idx=0 keeps the null token embedding exactly zero, so CFG null
    conditioning is simply all-zero embeddings (no special null vector needed).

    This module is intentionally decoupled from the DiT backbone so it can be
    swapped for a CNN, image encoder, or any other conditioning source.
    """

    def __init__(self, vocab_size: int, embed_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, out_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(tokens))  # (B, n_tokens, out_dim)


# ── Latent DiT model ───────────────────────────────────────────────────────────


class LatentDiT(BaseModel):
    """
    Latent diffusion model for MNIST Sudoku, using a patch-based DiT backbone.

    Frozen VAE encodes pixel images to latents; the DiT denoises in that space.
    Conditioning on the given-cell sudoku tokens is via cross-attention in every
    transformer block (Transformer2DModel with caption_channels).
    """

    def __init__(
        self,
        model_cfg: LatentDiTConfig,
        optim_cfg: LatentDiTOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        scheduler: Any,
        vae,  # AutoencoderKL, frozen
        scaling_factor: float,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.scheduler = scheduler
        self.scaling_factor = scaling_factor
        self.cell_size = model_cfg.cell_size
        self.painter_size = model_cfg.painter_size

        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad_(False)

        inner_dim = model_cfg.num_attention_heads * model_cfg.attention_head_dim
        self.dit = Transformer2DModel(
            num_attention_heads=model_cfg.num_attention_heads,
            attention_head_dim=model_cfg.attention_head_dim,
            in_channels=model_cfg.latent_channels,
            out_channels=model_cfg.latent_channels,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            cross_attention_dim=inner_dim,  # K/V projections sized to inner_dim
            sample_size=model_cfg.latent_size,
            patch_size=model_cfg.patch_size,
            norm_type="ada_norm_single",  # PixArt-style: continuous timestep, no class labels
            attention_bias=True,
            activation_fn="gelu-approximate",
            caption_channels=model_cfg.cond_embed_dim,  # projects cond_emb → inner_dim
        )

        self.cond_encoder = ConditionEncoder(
            vocab_size=model_cfg.vocab_size,
            embed_dim=model_cfg.cond_embed_dim,
            out_dim=model_cfg.cond_embed_dim,
        )

        self.eval_clf = None
        if eval_cfg.classifier_path is not None:
            self.eval_clf = load_or_train_classifier(eval_cfg.classifier_path, None, model_cfg.cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def build_optimizers(self, world_size: int, num_steps: int) -> list:
        params = list(self.dit.parameters()) + list(self.cond_encoder.parameters())
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

    def compile_submodules(self):
        self.dit = torch.compile(self.dit)

    def _trainable_params(self):
        return list(self.dit.parameters()) + list(self.cond_encoder.parameters())

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scaling_factor).sample.clamp(0.0, 1.0)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        condition_tokens: torch.Tensor,  # (B, 81) long, 0=null 1=blank 2-10=digits
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        cond_emb = self.cond_encoder(condition_tokens)  # (B, 81, cond_embed_dim)

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(cond_emb.shape[0], device=cond_emb.device) < self.train_cfg.cfg_prob
            cond_emb = cond_emb * (~drop[:, None, None])

        noise_pred = self.dit(noisy_z, timestep=timesteps, encoder_hidden_states=cond_emb).sample
        return noise_pred, None  # (noise_pred, sudoku_logits=None)

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step, **kwargs):
        K = len(micro_batches)
        device = accelerator.device
        total_loss = 0.0

        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device)
            condition_tokens = get_solution_tokens(solution)

            z = self._encode(images)
            B = z.shape[0]
            noise = torch.randn_like(z)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long
            )
            noisy_z = self.scheduler.add_noise(z, noise, timesteps)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z

            noise_pred, _ = self(noisy_z, timesteps, condition_tokens)
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
        max_batches = kwargs.get("max_batches", 100)
        device = accelerator.device
        self.eval()

        # Validation loss
        val_losses = []
        for i, batch in enumerate(tqdm(dataloader, desc="Eval loss", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            solution = batch["solution"].to(device)
            condition_tokens = get_solution_tokens(solution)
            B = images.shape[0]

            z = self._encode(images)
            noise = torch.randn_like(z)
            ts = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
            noisy_z = self.scheduler.add_noise(z, noise, ts)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z
            noise_pred, _ = self(noisy_z, ts, condition_tokens)
            val_losses.append(F.mse_loss(noise_pred.float(), target.float()).item())

        result = {"diff_loss": float(np.mean(val_losses))} if val_losses else {}

        # Sampling eval with classifier scoring
        if self.eval_clf is not None and accelerator.is_main_process:
            ddim = DDIMScheduler(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                beta_schedule=self.scheduler.config.beta_schedule,
                prediction_type=self.scheduler.config.prediction_type,
            )
            ddim.set_timesteps(self.eval_cfg.num_ddim_steps)
            cfg_scale = self.eval_cfg.cfg_scale

            all_cell_acc, all_puzzle_acc = [], []
            n_done, n_total = 0, self.eval_cfg.num_samples
            panel_images = []

            for batch in tqdm(dataloader, desc="Sampling", leave=False):
                if n_done >= n_total:
                    break
                solutions = batch["solution"]
                condition_tokens = get_solution_tokens(solutions.to(device))
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = condition_tokens.shape[0]

                # DDIM loop in latent space
                z = torch.randn(
                    B,
                    self.model_cfg.latent_channels,
                    self.model_cfg.latent_size,
                    self.model_cfg.latent_size,
                    device=device,
                )
                null_tokens = torch.zeros_like(condition_tokens) if cfg_scale > 1.0 else None

                for t in ddim.timesteps:
                    ts = torch.full((B,), t, device=device, dtype=torch.long)
                    noise_pred, _ = self(z, ts, condition_tokens)
                    if cfg_scale > 1.0:
                        noise_uncond, _ = self(z, ts, null_tokens)
                        noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
                    z = ddim.step(noise_pred, t, z).prev_sample

                generated = self._decode(z)

                acc = evaluate_grids(
                    generated,
                    solutions,
                    self.eval_clf,
                    self.cell_size,
                    given_masks=given_masks,
                )
                all_cell_acc.append(acc["cell_acc"])
                all_puzzle_acc.append(acc["puzzle_acc"])
                n_done += B

                n_log = self.eval_cfg.num_log_images
                if len(panel_images) < n_log and conditions_pixel is not None:
                    for j in range(min(n_log - len(panel_images), B)):
                        panel_images.append(
                            make_panel_image(
                                condition=conditions_pixel[j],
                                generated=generated[j].cpu(),
                                solution=solutions[j].numpy(),
                            )
                        )

            result["cell_acc"] = float(np.mean(all_cell_acc))
            result["puzzle_acc"] = float(np.mean(all_puzzle_acc))

            if panel_images and step is not None:
                try:
                    import wandb

                    tracker = accelerator.get_tracker("wandb", unwrap=True)
                    tracker.log(
                        {"val/examples": [wandb.Image(img) for img in panel_images]},
                        step=step,
                    )
                except Exception:
                    pass

        self.train()
        return result
