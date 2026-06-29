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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm

from datasets.mnist_sudoku_dataset import get_solution_tokens  # kept for eval metrics only
from eval.mnist_eval import evaluate_grids, make_panel_image
from models.condition_encoders import ObjectFeatureEncoder
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.spatial_diffusion_utils import (
    add_noise_spatial,
    add_noise_spatial_c,
    compute_denoising_speed,
    ddim_step_spatial,
    ddim_step_spatial_c,
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
    aug_prob_vanilla: float = 0.20  # raw U_teacher → standard make_t_field
    aug_prob_power: float = 0.30  # trajectory simulation (see below)
    aug_prob_threshold: float = 0.30  # rank-based adaptive T assignment
    aug_prob_perlin: float = 0.20  # U_teacher × Perlin noise → make_t_field

    # Power (denoising trajectory simulation):
    #   T = T_max * (1 - progress * (1 - U_teacher))
    #   progress ∈ [progress_min, progress_max] sampled per step.
    #   Simulates the T_field after `progress` fraction of confidence-driven denoising:
    #   confident pixels (low U) converge toward T=0; uncertain pixels stay near T_max.
    progress_min: float = 0.1
    progress_max: float = 0.9

    # Thresholding: rank pixels by U_teacher, split into n equal buckets,
    #   assign n randomly sampled T values (ascending → low U gets low T).
    #   Adaptive by construction — no fixed value range needed.
    threshold_n_min: int = 1
    threshold_n_max: int = 3

    # Teacher evaluation timestep for power & threshold augmentations.
    #   U_teacher is only informative at high noise levels; these augmentations
    #   always evaluate the teacher in [T_max * teacher_t_min_frac, T_max).
    #   Vanilla and perlin still use the batch's t_base.
    teacher_t_min_frac: float = 0.6

    # Model type: "unet" (UNet2DConditionModel + cross-attention) or "dit" (SpatialDiT + channel concat)
    model_type: str = "unet"

    # DiT-specific (only used when model_type="dit")
    patch_size: int = 4
    n_heads: int = 8
    attention_head_dim: int = 64
    n_layers: int = 6
    mlp_ratio: float = 4.0
    t_freq_dim: int = 256
    dropout: float = 0.0

    # Timestep mode
    # "discrete"   — T_field values are integers in [0, T_max-1], alpha_bar looked up
    #                from scheduler's alphas_cumprod table (original behaviour).
    # "continuous" — T_field values are floats in [0, T_max), alpha_bar computed
    #                analytically from the squaredcos_cap_v2 formula, giving smooth
    #                spatial noise gradients.
    continuous_time: bool = False

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

        self.cond_encoder = ObjectFeatureEncoder(
            in_dim=model_cfg.vocab_size,
            hidden_dim=model_cfg.cond_embed_dim,
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

    def _make_t_field(self, t_base: torch.Tensor, tau: float, perlin: torch.Tensor, T_max: int) -> torch.Tensor:
        """Return T_field as long (discrete) or float (continuous) depending on config."""
        t = t_base[:, None, None, None].float() + tau * perlin
        t = t.clamp(0, T_max - 1)
        return t if self.model_cfg.continuous_time else t.long()

    def _add_noise(self, x0: torch.Tensor, noise: torch.Tensor, T_field: torch.Tensor, T_max: int) -> torch.Tensor:
        if self.model_cfg.continuous_time:
            return add_noise_spatial_c(x0, noise, T_field, T_max)
        return add_noise_spatial(x0, noise, T_field, self.alphas_cumprod)

    def _ddim_step(
        self, z: torch.Tensor, x0_pred: torch.Tensor, T_old: torch.Tensor, T_new: torch.Tensor, T_max: int
    ) -> torch.Tensor:
        if self.model_cfg.continuous_time:
            return ddim_step_spatial_c(z, x0_pred, T_old, T_new, T_max)
        return ddim_step_spatial(z, x0_pred, T_old, T_new, self.alphas_cumprod)

    def _select_aug(self, roll: float) -> str:
        """Roulette selection — returns one of 'vanilla', 'power', 'threshold', 'perlin'."""
        cfg = self.model_cfg
        cum_v = cfg.aug_prob_vanilla
        cum_p = cum_v + cfg.aug_prob_power
        cum_t = cum_p + cfg.aug_prob_threshold
        if roll < cum_v:
            return "vanilla"
        if roll < cum_p:
            return "power"
        if roll < cum_t:
            return "threshold"
        return "perlin"

    def _teacher_t(self, aug_type: str, t_base: torch.Tensor, T_max: int, device: torch.device) -> torch.Tensor:
        """
        Choose the timestep at which to evaluate the teacher.
        Power and threshold use a high-noise T so the teacher's uncertainty map is
        spatially rich; vanilla and perlin use the batch's t_base (their T_field
        formula anchors on t_base anyway).
        """
        if aug_type in ("power", "threshold"):
            t_lo = int(T_max * self.model_cfg.teacher_t_min_frac)
            return torch.randint(t_lo, T_max, t_base.shape, device=device).float()
        return t_base

    def _augment_mask(
        self,
        U: torch.Tensor,  # (B, 1, lH, lW) teacher uncertainty in [0,1]
        aug_type: str,
        t_base: torch.Tensor,  # (B,) — used only for vanilla / perlin
        lH: int,
        lW: int,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns a T_field (B, 1, lH, lW) directly — callers no longer call _make_t_field.

        vanilla  — make_t_field(t_base, tau_student, U)
        power    — T = T_max * (1 - progress * (1 - U)), independent of t_base
        threshold— sort pixels by U rank, assign n sampled T values (ascending),
                   vectorised via argsort + scatter
        perlin   — make_t_field(t_base, tau_student, U * Perlin)
        """
        cfg = self.model_cfg
        B = U.shape[0]

        if aug_type == "vanilla":
            return self._make_t_field(t_base, cfg.tau_student, U, T_max)

        if aug_type == "power":
            progress = cfg.progress_min + torch.rand(1).item() * (cfg.progress_max - cfg.progress_min)
            # Normalize U per-batch so the most/least confident pixels always map to
            # T=T_max*(1-progress) / T=T_max regardless of the absolute confidence level.
            U_min = U.reshape(B, -1).min(dim=1).values[:, None, None, None]
            U_max = U.reshape(B, -1).max(dim=1).values[:, None, None, None]
            U_norm = (U - U_min) / (U_max - U_min + 1e-6)
            T_field = (T_max * (1.0 - progress * (1.0 - U_norm))).clamp(0, T_max - 1)
            return T_field if cfg.continuous_time else T_field.long()

        if aug_type == "threshold":
            n = torch.randint(cfg.threshold_n_min, cfg.threshold_n_max + 1, (1,)).item()
            U_flat = U.reshape(B, -1).float()  # (B, N)
            N = U_flat.shape[1]
            order = U_flat.argsort(dim=1)  # (B, N) ascending U — low U = confident

            # n T values per sample, sorted ascending (confident rank → low T)
            T_vals = (torch.rand(B, n, device=device) * T_max).sort(dim=1).values  # (B, n)

            # Build sorted T map then scatter back to original pixel order
            T_sorted = torch.zeros(B, N, device=device)
            bucket_size = N // n
            for i in range(n):
                lo = i * bucket_size
                hi = (i + 1) * bucket_size if i < n - 1 else N
                T_sorted[:, lo:hi] = T_vals[:, i : i + 1].expand(B, hi - lo)

            T_original = torch.zeros(B, N, device=device)
            T_original.scatter_(1, order, T_sorted)  # pixel order[b,pos] gets T_sorted[b,pos]
            T_field = T_original.reshape(B, 1, lH, lW).clamp(0, T_max - 1)
            return T_field if cfg.continuous_time else T_field.long()

        # perlin
        perlin = smooth_noise_field(B, lH, lW, cfg.f_spatial, device, n_octaves=cfg.n_octaves)
        return self._make_t_field(t_base, cfg.tau_student, (U * perlin).clamp(0, 1), T_max)

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
        total_comps: dict = {}

        for mb in micro_batches:
            images = mb["images"].to(device)
            condition_tokens = mb["puzzle_tokens"].to(device)  # given cells only — not full solution
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
            T_init = self._make_t_field(t_base, self.model_cfg.tau_init, perlin_init, T_max)

            if torch.rand(1).item() < p_refine:
                z_init = self._add_noise(z0, torch.randn_like(z0), T_init, T_max)
                with torch.no_grad():
                    _, log_var_t = self._run_unet(z_init, T_init, self.teacher_cond_enc(condition_tokens), self.teacher)
                U_teacher = log_var_t.sigmoid()
                perlin_final = smooth_noise_field(
                    B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves
                )
                T_student = self._make_t_field(t_base, self.model_cfg.tau_student, U_teacher * perlin_final, T_max)
                z_s = self._add_noise(z0, torch.randn_like(z0), T_student, T_max)
                x0_pred, log_var = self._run_unet(z_s, T_student, cond_emb, self.unet)
            else:
                z_noisy = self._add_noise(z0, torch.randn_like(z0), T_init, T_max)
                x0_pred, log_var = self._run_unet(z_noisy, T_init, cond_emb, self.unet)

            loss, comps = gaussian_nll_loss(z0, x0_pred, log_var)
            total_loss += loss.item()
            for k, v in comps.items():
                total_comps[k] = total_comps.get(k, 0.0) + v
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self._trainable_params(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        self._update_teacher()
        global_step += 1
        return (
            {
                "nll_loss": total_loss / K,
                "p_refine": p_refine,
                **{k: v / K for k, v in total_comps.items()},
            },
            lr,
            global_step,
        )

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
        total_comps: dict = {}

        for mb in micro_batches:
            images = mb["images"].to(device)
            condition_tokens = mb["puzzle_tokens"].to(device)  # given cells only — not full solution
            B = images.shape[0]

            z0 = self._encode(images)
            _, _, lH, lW = z0.shape

            cond_emb = self.cond_encoder(condition_tokens)
            if cfg_prob > 0:
                drop = torch.rand(B, device=device) < cfg_prob
                cond_emb = cond_emb * (~drop[:, None, None])

            # Uniform base timestep (same for all pixels in this sample)
            t_base = torch.randint(0, T_max, (B,), device=device).float()
            T_uniform_f = t_base[:, None, None, None].expand(B, 1, lH, lW)
            T_uniform = T_uniform_f if self.model_cfg.continuous_time else T_uniform_f.long()

            if in_stage1:
                # Stage 1: plain uniform noise, no teacher
                z_noisy = self._add_noise(z0, torch.randn_like(z0), T_uniform, T_max)
                x0_pred, log_var = self._run_unet(z_noisy, T_uniform, cond_emb, self.unet)
            else:
                # Stage 2: roll augmentation first, then evaluate teacher at the
                # appropriate T (high-noise for power/threshold; t_base for others).
                aug_type = self._select_aug(torch.rand(1).item())
                t_teacher = self._teacher_t(aug_type, t_base, T_max, device)
                T_teacher_f = t_teacher[:, None, None, None].expand(B, 1, lH, lW)
                T_teacher = T_teacher_f if self.model_cfg.continuous_time else T_teacher_f.long()

                z_teacher = self._add_noise(z0, torch.randn_like(z0), T_teacher, T_max)
                with torch.no_grad():
                    _, log_var_t = self._run_unet(
                        z_teacher, T_teacher, self.teacher_cond_enc(condition_tokens), self.teacher
                    )
                U_raw = log_var_t.sigmoid()
                T_student = self._augment_mask(U_raw, aug_type, t_base, lH, lW, T_max, device)
                z_s = self._add_noise(z0, torch.randn_like(z0), T_student, T_max)
                x0_pred, log_var = self._run_unet(z_s, T_student, cond_emb, self.unet)

            loss, comps = gaussian_nll_loss(z0, x0_pred, log_var)
            total_loss += loss.item()
            for k, v in comps.items():
                total_comps[k] = total_comps.get(k, 0.0) + v
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self._trainable_params(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        self._update_teacher()
        global_step += 1
        return (
            {
                "nll_loss": total_loss / K,
                "stage": 1 if in_stage1 else 2,
                **{k: v / K for k, v in total_comps.items()},
            },
            lr,
            global_step,
        )

    # ── T-field visualisation ──────────────────────────────────────────────────

    def _tfield_panels(
        self,
        images: torch.Tensor,  # (N, 1, H, W) pixel images [0,1]
        T_field: torch.Tensor,  # (N, 1, lH, lW) float or long
        T_max: int,
    ) -> list:
        """
        Build one panel per sample: [original | overlay | heatmap].
        T_field is upsampled to image resolution, colourised with 'plasma',
        then blended 50/50 with the greyscale image for the overlay column.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm

        B, _, H, W = images.shape
        t_norm = T_field.float() / (T_max - 1)  # (N,1,lH,lW) in [0,1]
        t_up = F.interpolate(t_norm, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)  # (N,H,W)
        cmap = cm.get_cmap("plasma")
        sep = np.full((H, 4, 3), 180, dtype=np.uint8)
        panels = []
        for i in range(B):
            img = (images[i, 0].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            img_rgb = np.stack([img] * 3, axis=-1)  # (H,W,3)
            heat = (cmap(t_up[i].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)
            overlay = ((img_rgb / 2.0) + (heat / 2.0)).astype(np.uint8)
            panels.append(np.concatenate([img_rgb, sep, overlay, sep, heat], axis=1))
        return panels

    @torch.no_grad()
    def _log_tfield_panels(self, dataloader, accelerator, step, n_samples, T_max, device):
        """
        Visualise T_field for each active augmentation type (teacher_guided mode)
        or the raw Perlin field (perlin mode) and log to wandb.
        Does NOT include the standard uniform denoising field.
        """
        cfg = self.model_cfg
        batch = next(iter(dataloader))
        images = batch["images"][:n_samples].to(device)
        B = images.shape[0]
        _, _, H, W = images.shape

        condition_tokens = batch["puzzle_tokens"][:n_samples].to(device)
        z0 = self._encode(images)
        _, _, lH, lW = z0.shape

        t_base = torch.randint(0, T_max, (B,), device=device).float()
        t_cond = self.teacher_cond_enc(condition_tokens)

        def _eval_teacher_at(t_vec):
            """Evaluate teacher at uniform t_vec (B,), return U = log_var.sigmoid()."""
            T_f = t_vec[:, None, None, None].expand(B, 1, lH, lW)
            T_f = T_f if cfg.continuous_time else T_f.long()
            z_n = self._add_noise(z0, torch.randn_like(z0), T_f, T_max)
            _, lv = self._run_unet(z_n, T_f, t_cond, self.teacher)
            return lv.sigmoid(), T_f

        # ── Teacher-guided mode: one panel set per active aug type ─────────────
        if cfg.noise_mode == "teacher_guided":
            # U at t_base (for vanilla/perlin) and at t_high (for power/threshold)
            t_high = self._teacher_t("power", t_base, T_max, device)  # representative high T
            U_base, _ = _eval_teacher_at(t_base)
            U_high, _ = _eval_teacher_at(t_high)

            aug_tfields = {}
            if cfg.aug_prob_vanilla > 0:
                aug_tfields["vanilla"] = self._augment_mask(U_base, "vanilla", t_base, lH, lW, T_max, device)
            if cfg.aug_prob_power > 0:
                aug_tfields["power"] = self._augment_mask(U_high, "power", t_base, lH, lW, T_max, device)
            if cfg.aug_prob_threshold > 0:
                aug_tfields["threshold"] = self._augment_mask(U_high, "threshold", t_base, lH, lW, T_max, device)
            if cfg.aug_prob_perlin > 0:
                aug_tfields["perlin_aug"] = self._augment_mask(U_base, "perlin", t_base, lH, lW, T_max, device)

        # ── Perlin mode: show Path A and Path B Perlin fields ──────────────────
        else:
            aug_tfields = {}
            perlin_a = smooth_noise_field(B, lH, lW, cfg.f_spatial, device, n_octaves=cfg.n_octaves)
            aug_tfields["perlin_pathA"] = self._make_t_field(t_base, cfg.tau_init, perlin_a, T_max)

            if cfg.p_refine_max > 0:
                # Simulate Path B: use teacher uncertainty × perlin
                T_init = aug_tfields["perlin_pathA"]
                z_init = self._add_noise(z0, torch.randn_like(z0), T_init, T_max)
                t_cond = self.teacher_cond_enc(condition_tokens)
                _, log_var_t = self._run_unet(z_init, T_init, t_cond, self.teacher)
                U_teacher = log_var_t.sigmoid()
                perlin_b = smooth_noise_field(B, lH, lW, cfg.f_spatial, device, n_octaves=cfg.n_octaves)
                aug_tfields["perlin_pathB"] = self._make_t_field(t_base, cfg.tau_student, U_teacher * perlin_b, T_max)

        # ── Build wandb log dict ────────────────────────────────────────────────
        log_dict = {}
        for aug_name, T_field in aug_tfields.items():
            panels = self._tfield_panels(images, T_field, T_max)
            try:
                import wandb

                log_dict[f"val/tfield_{aug_name}"] = [wandb.Image(p) for p in panels]
            except Exception:
                pass

        if log_dict:
            try:
                import wandb

                tracker = accelerator.get_tracker("wandb", unwrap=True)
                tracker.log(log_dict, step=step)
            except Exception:
                pass

    # ── Eval step ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        step = kwargs.get("step", None)
        max_batches = kwargs.get("max_batches", 100)
        num_ddim_steps = kwargs.get("num_ddim_steps", 20)
        num_samples = kwargs.get("num_samples", 512)
        cfg_scale = kwargs.get("cfg_scale", 1.0)
        num_log_images = kwargs.get("num_log_images", 8)
        guidance_alpha = float(kwargs.get("guidance_alpha", 1.0))
        guidance_power = float(kwargs.get("guidance_power", 1.0))
        guidance_top_m = kwargs.get("guidance_top_m", None)
        if guidance_top_m is not None:
            guidance_top_m = float(guidance_top_m)

        device = accelerator.device
        T_max = self.scheduler.config.num_train_timesteps
        self.eval()

        # Validation loss (Path A, no teacher)
        val_losses, val_comps = [], {}
        for i, batch in enumerate(tqdm(dataloader, desc="Eval loss", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            B = images.shape[0]
            z0 = self._encode(images)
            _, _, lH, lW = z0.shape
            cond_emb = self.cond_encoder(batch["puzzle_tokens"].to(device))

            t_base = torch.randint(0, T_max, (B,), device=device).float()
            perlin = smooth_noise_field(B, lH, lW, self.model_cfg.f_spatial, device)
            T_field = self._make_t_field(t_base, self.model_cfg.tau_init, perlin, T_max)
            z_noisy = self._add_noise(z0, torch.randn_like(z0), T_field, T_max)
            x0_pred, log_var = self._run_unet(z_noisy, T_field, cond_emb, self.unet)
            loss, comps = gaussian_nll_loss(z0, x0_pred, log_var)
            val_losses.append(loss.item())
            for k, v in comps.items():
                val_comps[k] = val_comps.get(k, 0.0) + v

        if val_losses:
            n = len(val_losses)
            result = {"nll_loss": float(np.mean(val_losses)), **{k: v / n for k, v in val_comps.items()}}
        else:
            result = {}

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
                condition_tokens = batch["puzzle_tokens"].to(device)
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = condition_tokens.shape[0]

                cond_emb = self.cond_encoder(condition_tokens)
                null_emb = self.cond_encoder(torch.zeros_like(condition_tokens)) if cfg_scale > 1.0 else None

                z = torch.randn(B, C, lH, lW, device=device)
                _t_init = float(T_max - 1)
                _dtype = torch.float32 if self.model_cfg.continuous_time else torch.long
                T_field = torch.full((B, 1, lH, lW), _t_init, device=device, dtype=_dtype)

                for _ in range(num_ddim_steps):
                    T_old = T_field.clone()
                    x0_pred, log_var = self._run_unet(z, T_field, cond_emb, self.unet)

                    if cfg_scale > 1.0:
                        x0_uncond, _ = self._run_unet(z, T_field, null_emb, self.unet)
                        x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)

                    # Confident pixels advance faster
                    uncertainty = log_var.sigmoid()
                    T_new = T_field.float() - compute_denoising_speed(
                        uncertainty, dt,
                        alpha=guidance_alpha,
                        power=guidance_power,
                        top_m=guidance_top_m,
                    )
                    T_new = T_new.clamp(0, T_max - 1)
                    if not self.model_cfg.continuous_time:
                        T_new = T_new.round().long()
                    z = self._ddim_step(z, x0_pred, T_old, T_new, T_max)
                    T_field = T_new

                    if T_field.max() < 0.5:
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
                condition_tokens = batch["puzzle_tokens"].to(device)
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = condition_tokens.shape[0]

                cond_emb = self.cond_encoder(condition_tokens)
                null_emb = self.cond_encoder(torch.zeros_like(condition_tokens)) if cfg_scale > 1.0 else None

                z = torch.randn(B, C, lH, lW, device=device)

                _dtype = torch.float32 if self.model_cfg.continuous_time else torch.long
                for s in range(num_ddim_steps):
                    t_val = uniform_ts[s].item()
                    t_next = uniform_ts[s + 1].item()
                    T_old = torch.full((B, 1, lH, lW), t_val, device=device, dtype=_dtype)
                    T_new = torch.full((B, 1, lH, lW), t_next, device=device, dtype=_dtype)

                    x0_pred, _ = self._run_unet(z, T_old, cond_emb, self.unet)
                    if cfg_scale > 1.0:
                        x0_uncond, _ = self._run_unet(z, T_old, null_emb, self.unet)
                        x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)

                    z = self._ddim_step(z, x0_pred, T_old, T_new, T_max)

                generated_u = self._decode(z)
                acc_u = evaluate_grids(
                    generated_u,
                    solutions,
                    self.eval_clf,
                    self.cell_size,
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

        # T-field visualisation (main process, wandb only)
        if accelerator.is_main_process and step is not None:
            self._log_tfield_panels(
                dataloader,
                accelerator,
                step,
                n_samples=num_log_images,
                T_max=T_max,
                device=device,
            )

        self.train()
        return result
