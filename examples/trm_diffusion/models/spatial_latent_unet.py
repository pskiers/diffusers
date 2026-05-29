"""
models/spatial_latent_unet.py — Spatially-varying latent diffusion model.

Architecture:
  - Frozen VAE: pixel images ↔ latent space
  - ConditionEncoder: 81 sudoku tokens → cross-attention context
  - UNet2DConditionModel:
      in_channels  = latent_channels + 1  (noisy latent + normalised T_field)
      out_channels = latent_channels + 1  (x0 prediction + log-variance scalar map)

Training — mixed-batch refinement:
  Path A (prob 1 - p_refine): smooth Perlin T_field → spatial noise → student NLL loss.
  Path B (prob     p_refine):  same Perlin → EMA teacher uncertainty → structured T_field
                                → fresh noise → student NLL loss.
  p_refine ramps 0 → p_refine_max over p_refine_warmup_steps (starts all Path A).

The EMA teacher is managed internally; the outer training loop should disable EMAHelper.

Inference: confidence-driven variable-rate denoising — certain pixels converge faster.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm

from datasets.mnist_sudoku_dataset import get_solution_tokens
from eval.mnist_eval import evaluate_grids, load_or_train_classifier, make_panel_image
from models.latent_dit import ConditionEncoder
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.spatial_diffusion_utils import (
    add_noise_spatial,
    ddim_step_spatial,
    gaussian_nll_loss,
    make_t_field,
    smooth_noise_field,
)

# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class SpatialLatentConfig:
    vae_checkpoint: str
    latent_channels: int = 4
    latent_size: int = 36

    # UNet architecture
    down_block_types: tuple = field(default_factory=lambda: ("DownBlock2D", "CrossAttnDownBlock2D"))
    up_block_types: tuple = field(default_factory=lambda: ("CrossAttnUpBlock2D", "UpBlock2D"))
    block_out_channels: tuple = field(default_factory=lambda: (64, 128))
    layers_per_block: int = 2
    attention_head_dim: int = 8
    norm_num_groups: int = 16

    # Conditioning
    vocab_size: int = 11  # 0=null, 1=blank, 2-10=digits 1-9
    cond_embed_dim: int = 256

    # Spatial noise
    f_spatial: float = 0.25  # spatial frequency; lower → larger blobs
    tau_init: float = 20.0  # Perlin amplitude for T_init (timestep units)
    tau_student: float = 30.0  # Perlin amplitude for T_student in Path B
    n_octaves: int = 1

    # Path B schedule (perlin mode only)
    p_refine_max: float = 0.7
    p_refine_warmup_steps: int = 20000

    # ── teacher_guided mode ───────────────────────────────────────────────────
    # noise_mode: "perlin" (original Path A/B) | "teacher_guided" (Stage 1/2)
    noise_mode: str = "perlin"

    # Stage 1: uniform scalar warm-up before teacher is useful
    stage1_steps: int = 5000

    # Augmentation roulette probabilities (must sum to 1.0)
    aug_prob_vanilla: float = 0.20  # raw U_teacher
    aug_prob_power: float = 0.30  # U_teacher ^ gamma  (sharpens peaks)
    aug_prob_threshold: float = 0.30  # quantise to n+1 levels at random thresholds
    aug_prob_perlin: float = 0.20  # multiply by Perlin noise

    # Power: gamma ~ U(power_gamma_min, power_gamma_max)
    power_gamma_min: float = 2.0
    power_gamma_max: float = 4.0

    # Thresholding: sample n ~ randint(n_min, n_max) split points from
    #   U(val_min, val_max), then quantise mask to n+1 uniform levels
    threshold_n_min: int = 1  # 1 split → binary (2 levels)
    threshold_n_max: int = 3  # 3 splits → 4 levels
    threshold_val_min: float = 0.2
    threshold_val_max: float = 0.8

    # EMA teacher
    teacher_ema_rate: float = 0.999

    # Eval
    cell_size: int = 16
    painter_size: int = 144


@dataclass
class SpatialLatentOptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1


# ── Model ──────────────────────────────────────────────────────────────────────


class SpatialLatentUNet(nn.Module):
    """
    Latent diffusion with spatially-varying timestep fields and heteroscedastic uncertainty.
    """

    def __init__(
        self,
        model_cfg: SpatialLatentConfig,
        optim_cfg: SpatialLatentOptimConfig,
        scheduler: Any,
        vae: nn.Module,
        scaling_factor: float,
        eval_clf=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler = scheduler
        self.scaling_factor = scaling_factor
        self.cell_size = model_cfg.cell_size
        self.painter_size = model_cfg.painter_size

        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad_(False)

        ac = torch.as_tensor(scheduler.alphas_cumprod, dtype=torch.float32).clone()
        self.register_buffer("alphas_cumprod", ac)

        C = model_cfg.latent_channels
        self.unet = UNet2DConditionModel(
            sample_size=model_cfg.latent_size,
            in_channels=C + 1,
            out_channels=C + 1,
            down_block_types=list(model_cfg.down_block_types),
            up_block_types=list(model_cfg.up_block_types),
            block_out_channels=list(model_cfg.block_out_channels),
            layers_per_block=model_cfg.layers_per_block,
            cross_attention_dim=model_cfg.cond_embed_dim,
            norm_num_groups=model_cfg.norm_num_groups,
            attention_head_dim=model_cfg.attention_head_dim,
            act_fn="silu",
        )

        self.cond_encoder = ConditionEncoder(
            vocab_size=model_cfg.vocab_size,
            embed_dim=model_cfg.cond_embed_dim,
            out_dim=model_cfg.cond_embed_dim,
        )

        # EMA teacher — deep copy, frozen, updated after every optimizer step
        self.teacher = copy.deepcopy(self.unet)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_cond_enc = copy.deepcopy(self.cond_encoder)
        for p in self.teacher_cond_enc.parameters():
            p.requires_grad_(False)

        self.eval_clf = eval_clf
        if self.eval_clf is not None:
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def build_optimizers(self, world_size: int, num_steps: int) -> list:
        params = list(self.unet.parameters()) + list(self.cond_encoder.parameters())
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
        self.unet = torch.compile(self.unet)

    def _trainable_params(self):
        return list(self.unet.parameters()) + list(self.cond_encoder.parameters())

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scaling_factor).sample.clamp(0.0, 1.0)

    def _augment_mask(self, U: torch.Tensor, lH: int, lW: int, device: torch.device) -> torch.Tensor:
        """
        Apply one randomly-chosen augmentation to uncertainty mask U in [0,1].
        Draws from the roulette defined in model_cfg.aug_prob_*.
        """
        cfg = self.model_cfg
        roll = torch.rand(1).item()
        cum_v = cfg.aug_prob_vanilla
        cum_p = cum_v + cfg.aug_prob_power
        cum_t = cum_p + cfg.aug_prob_threshold

        if roll < cum_v:
            # Vanilla — unchanged
            return U

        if roll < cum_p:
            # Power scaling: sharpen peaks, crush valleys
            gamma = cfg.power_gamma_min + torch.rand(1).item() * (cfg.power_gamma_max - cfg.power_gamma_min)
            return U.pow(gamma)

        if roll < cum_t:
            # Random quantisation: sample n split points, map to n+1 uniform levels
            n = torch.randint(cfg.threshold_n_min, cfg.threshold_n_max + 1, (1,)).item()
            splits = sorted(
                cfg.threshold_val_min + torch.rand(n).tolist()[i] * (cfg.threshold_val_max - cfg.threshold_val_min)
                for i in range(n)
            )
            boundaries = torch.tensor(splits, device=device, dtype=torch.float32)
            bucket = torch.bucketize(U.squeeze(1), boundaries)  # (B, lH, lW) int
            return (bucket.float() / n).unsqueeze(1)  # (B, 1, lH, lW) in [0,1]

        # Perlin modulation: add organic blur to edges
        B = U.shape[0]
        perlin = smooth_noise_field(B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves)
        return (U * perlin).clamp(0.0, 1.0)

    def _p_refine(self, step: int) -> float:
        cfg = self.model_cfg
        if cfg.p_refine_warmup_steps <= 0:
            return cfg.p_refine_max
        return min(cfg.p_refine_max, step / cfg.p_refine_warmup_steps * cfg.p_refine_max)

    @torch.no_grad()
    def _update_teacher(self):
        mu = self.model_cfg.teacher_ema_rate
        for tp, sp in zip(self.teacher.parameters(), self.unet.parameters()):
            tp.data.mul_(mu).add_(sp.data, alpha=1.0 - mu)
        for tp, sp in zip(self.teacher_cond_enc.parameters(), self.cond_encoder.parameters()):
            tp.data.mul_(mu).add_(sp.data, alpha=1.0 - mu)

    def _run_unet(
        self,
        z: torch.Tensor,
        T_field: torch.Tensor,  # (B, 1, lH, lW) long
        cond_emb: torch.Tensor,
        unet: UNet2DConditionModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single UNet forward. Returns (x0_pred, log_var)."""
        T_max = self.scheduler.config.num_train_timesteps
        t_norm = T_field.float() / (T_max - 1)  # (B, 1, lH, lW) → [0,1]
        t_scalar = T_field.float().mean(dim=(1, 2, 3)).long()  # (B,)
        out = unet(torch.cat([z, t_norm], dim=1), timestep=t_scalar, encoder_hidden_states=cond_emb).sample
        C = self.model_cfg.latent_channels
        return out[:, :C], out[:, C:]  # x0_pred (B,C,H,W), log_var (B,1,H,W)

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(
        self,
        micro_batches: list,
        accelerator,
        optimizers: list,
        global_batch_size: int,
        global_step: int,
        cfg_prob: float = 0.0,
    ) -> tuple[dict, float, int]:
        if self.model_cfg.noise_mode == "teacher_guided":
            return self._train_step_teacher_guided(
                micro_batches,
                accelerator,
                optimizers,
                global_batch_size,
                global_step,
                cfg_prob,
            )
        return self._train_step_perlin(
            micro_batches,
            accelerator,
            optimizers,
            global_batch_size,
            global_step,
            cfg_prob,
        )

    def _train_step_perlin(
        self,
        micro_batches,
        accelerator,
        optimizers,
        global_batch_size,
        global_step,
        cfg_prob,
    ):
        """Original Path A / Path B with Perlin T_field."""
        K = len(micro_batches)
        device = accelerator.device
        T_max = self.scheduler.config.num_train_timesteps
        p_refine = self._p_refine(global_step)
        total_loss = 0.0

        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device)
            condition_tokens = get_solution_tokens(solution)
            B = images.shape[0]

            z0 = self._encode(images)
            _, _, lH, lW = z0.shape

            cond_emb = self.cond_encoder(condition_tokens)
            if cfg_prob > 0:
                drop = torch.rand(B, device=device) < cfg_prob
                cond_emb = cond_emb * (~drop[:, None, None])

            t_base = torch.randint(0, T_max, (B,), device=device).float()
            perlin_init = smooth_noise_field(
                B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves
            )
            T_init = make_t_field(t_base, self.model_cfg.tau_init, perlin_init, T_max)

            if torch.rand(1).item() < p_refine:
                # Path B: teacher uncertainty → structured T_student
                z_init = add_noise_spatial(z0, torch.randn_like(z0), T_init, self.alphas_cumprod)
                with torch.no_grad():
                    _, log_var_t = self._run_unet(z_init, T_init, self.teacher_cond_enc(condition_tokens), self.teacher)
                U_teacher = log_var_t.sigmoid()
                perlin_final = smooth_noise_field(
                    B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves
                )
                T_student = make_t_field(t_base, self.model_cfg.tau_student, U_teacher * perlin_final, T_max)
                z_s = add_noise_spatial(z0, torch.randn_like(z0), T_student, self.alphas_cumprod)
                x0_pred, log_var = self._run_unet(z_s, T_student, cond_emb, self.unet)
            else:
                # Path A
                z_noisy = add_noise_spatial(z0, torch.randn_like(z0), T_init, self.alphas_cumprod)
                x0_pred, log_var = self._run_unet(z_noisy, T_init, cond_emb, self.unet)

            loss = gaussian_nll_loss(z0, x0_pred, log_var)
            total_loss += loss.item()
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self._trainable_params(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        self._update_teacher()
        global_step += 1
        return {"nll_loss": total_loss / K, "p_refine": p_refine}, lr, global_step

    def _train_step_teacher_guided(
        self,
        micro_batches,
        accelerator,
        optimizers,
        global_batch_size,
        global_step,
        cfg_prob,
    ):
        """Stage 1/2 teacher-guided mode with augmentation roulette."""
        K = len(micro_batches)
        device = accelerator.device
        T_max = self.scheduler.config.num_train_timesteps
        in_stage1 = global_step < self.model_cfg.stage1_steps
        total_loss = 0.0

        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device)
            condition_tokens = get_solution_tokens(solution)
            B = images.shape[0]

            z0 = self._encode(images)
            _, _, lH, lW = z0.shape

            cond_emb = self.cond_encoder(condition_tokens)
            if cfg_prob > 0:
                drop = torch.rand(B, device=device) < cfg_prob
                cond_emb = cond_emb * (~drop[:, None, None])

            # Uniform base timestep (same for all pixels in this sample)
            t_base = torch.randint(0, T_max, (B,), device=device).float()
            T_uniform = t_base[:, None, None, None].expand(B, 1, lH, lW).long()

            if in_stage1:
                # Stage 1: plain uniform noise, no teacher
                z_noisy = add_noise_spatial(z0, torch.randn_like(z0), T_uniform, self.alphas_cumprod)
                x0_pred, log_var = self._run_unet(z_noisy, T_uniform, cond_emb, self.unet)
            else:
                # Stage 2: uniform noise → teacher → augment mask → structured T_student
                z_uniform = add_noise_spatial(z0, torch.randn_like(z0), T_uniform, self.alphas_cumprod)
                with torch.no_grad():
                    _, log_var_t = self._run_unet(
                        z_uniform, T_uniform, self.teacher_cond_enc(condition_tokens), self.teacher
                    )
                U_raw = log_var_t.sigmoid()  # (B, 1, lH, lW) in (0,1)
                U_aug = self._augment_mask(U_raw, lH, lW, device)  # augmented mask

                T_student = make_t_field(t_base, self.model_cfg.tau_student, U_aug, T_max)
                z_s = add_noise_spatial(z0, torch.randn_like(z0), T_student, self.alphas_cumprod)
                x0_pred, log_var = self._run_unet(z_s, T_student, cond_emb, self.unet)

            loss = gaussian_nll_loss(z0, x0_pred, log_var)
            total_loss += loss.item()
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self._trainable_params(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        self._update_teacher()
        global_step += 1
        return (
            {
                "nll_loss": total_loss / K,
                "stage": 1 if in_stage1 else 2,
            },
            lr,
            global_step,
        )

    # ── Eval step ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        step = kwargs.get("step", None)
        max_batches = kwargs.get("max_batches", 100)
        num_ddim_steps = kwargs.get("num_ddim_steps", 20)
        num_samples = kwargs.get("num_samples", 512)
        cfg_scale = kwargs.get("cfg_scale", 1.0)
        num_log_images = kwargs.get("num_log_images", 8)

        device = accelerator.device
        T_max = self.scheduler.config.num_train_timesteps
        self.eval()

        # Validation loss (Path A, no teacher)
        val_losses = []
        for i, batch in enumerate(tqdm(dataloader, desc="Eval loss", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            solution = batch["solution"].to(device)
            B = images.shape[0]
            z0 = self._encode(images)
            _, _, lH, lW = z0.shape
            cond_emb = self.cond_encoder(get_solution_tokens(solution))

            t_base = torch.randint(0, T_max, (B,), device=device).float()
            perlin = smooth_noise_field(B, lH, lW, self.model_cfg.f_spatial, device)
            T_field = make_t_field(t_base, self.model_cfg.tau_init, perlin, T_max)
            z_noisy = add_noise_spatial(z0, torch.randn_like(z0), T_field, self.alphas_cumprod)
            x0_pred, log_var = self._run_unet(z_noisy, T_field, cond_emb, self.unet)
            val_losses.append(gaussian_nll_loss(z0, x0_pred, log_var).item())

        result = {"nll_loss": float(np.mean(val_losses))} if val_losses else {}

        # Sampling + accuracy eval
        if self.eval_clf is not None and accelerator.is_main_process:
            all_cell_acc, all_puzzle_acc = [], []
            n_done, n_total = 0, num_samples
            panel_images = []
            dt = T_max / num_ddim_steps
            C = self.model_cfg.latent_channels
            lH = lW = self.model_cfg.latent_size

            for batch in tqdm(dataloader, desc="Sampling", leave=False):
                if n_done >= n_total:
                    break
                solutions = batch["solution"]
                condition_tokens = get_solution_tokens(solutions.to(device))
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = condition_tokens.shape[0]

                cond_emb = self.cond_encoder(condition_tokens)
                null_emb = self.cond_encoder(torch.zeros_like(condition_tokens)) if cfg_scale > 1.0 else None

                z = torch.randn(B, C, lH, lW, device=device)
                T_field = torch.full((B, 1, lH, lW), T_max - 1, device=device, dtype=torch.long)

                for _ in range(num_ddim_steps):
                    T_old = T_field.clone()
                    x0_pred, log_var = self._run_unet(z, T_field, cond_emb, self.unet)

                    if cfg_scale > 1.0:
                        x0_uncond, _ = self._run_unet(z, T_field, null_emb, self.unet)
                        x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)

                    # Confident pixels advance faster
                    uncertainty = log_var.sigmoid()
                    T_new = (T_field.float() - dt * (1.0 - uncertainty)).clamp(0, T_max - 1).round().long()
                    z = ddim_step_spatial(z, x0_pred, T_old, T_new, self.alphas_cumprod)
                    T_field = T_new

                    if T_field.max() == 0:
                        break

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

                if len(panel_images) < num_log_images and conditions_pixel is not None:
                    for j in range(min(num_log_images - len(panel_images), B)):
                        panel_images.append(
                            make_panel_image(
                                condition=conditions_pixel[j],
                                generated=generated[j].cpu(),
                                solution=solutions[j].cpu().numpy(),
                            )
                        )

            result["cell_acc"] = float(np.mean(all_cell_acc))
            result["puzzle_acc"] = float(np.mean(all_puzzle_acc))

            # ── Uniform DDIM sampling (all pixels same timestep) ───────────────
            # Provides a direct comparison baseline for the spatial denoising.
            all_cell_acc_u, all_puzzle_acc_u = [], []
            n_done_u = 0
            panel_images_u = []
            # Evenly-spaced timesteps T_max-1 → 0
            uniform_ts = torch.linspace(T_max - 1, 0, num_ddim_steps + 1).long()

            for batch in tqdm(dataloader, desc="Sampling (uniform)", leave=False):
                if n_done_u >= n_total:
                    break
                solutions = batch["solution"]
                condition_tokens = get_solution_tokens(solutions.to(device))
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = condition_tokens.shape[0]

                cond_emb = self.cond_encoder(condition_tokens)
                null_emb = self.cond_encoder(torch.zeros_like(condition_tokens)) if cfg_scale > 1.0 else None

                z = torch.randn(B, C, lH, lW, device=device)

                for s in range(num_ddim_steps):
                    t_val = uniform_ts[s].item()
                    t_next = uniform_ts[s + 1].item()
                    T_old = torch.full((B, 1, lH, lW), t_val, device=device, dtype=torch.long)
                    T_new = torch.full((B, 1, lH, lW), t_next, device=device, dtype=torch.long)

                    x0_pred, _ = self._run_unet(z, T_old, cond_emb, self.unet)
                    if cfg_scale > 1.0:
                        x0_uncond, _ = self._run_unet(z, T_old, null_emb, self.unet)
                        x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)

                    z = ddim_step_spatial(z, x0_pred, T_old, T_new, self.alphas_cumprod)

                generated_u = self._decode(z)
                acc_u = evaluate_grids(
                    generated_u, solutions, self.eval_clf, self.cell_size,
                    given_masks=given_masks,
                )
                all_cell_acc_u.append(acc_u["cell_acc"])
                all_puzzle_acc_u.append(acc_u["puzzle_acc"])
                n_done_u += B

                if len(panel_images_u) < num_log_images and conditions_pixel is not None:
                    for j in range(min(num_log_images - len(panel_images_u), B)):
                        panel_images_u.append(
                            make_panel_image(
                                condition=conditions_pixel[j],
                                generated=generated_u[j].cpu(),
                                solution=solutions[j].cpu().numpy(),
                            )
                        )

            result["cell_acc_uniform"] = float(np.mean(all_cell_acc_u))
            result["puzzle_acc_uniform"] = float(np.mean(all_puzzle_acc_u))

            if panel_images and step is not None:
                try:
                    import wandb

                    tracker = accelerator.get_tracker("wandb", unwrap=True)
                    tracker.log(
                        {
                            "val/examples_spatial": [wandb.Image(img) for img in panel_images],
                            "val/examples_uniform": [wandb.Image(img) for img in panel_images_u],
                        },
                        step=step,
                    )
                except Exception:
                    pass

        self.train()
        return result
