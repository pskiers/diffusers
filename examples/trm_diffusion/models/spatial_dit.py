"""
models/spatial_dit.py — Spatial DiT with per-token adaLN for spatially-varying diffusion.

Key design:
  PerTokenAdaLN   — drop-in replacement for AdaLayerNormZero where the timestep
                    embedding is (B, N, D) instead of (B, D).  When all token
                    embeddings are identical (uniform T_field) the output is
                    exactly equal to standard global adaLN.

  SpatialDiTBlock — mirrors BasicTransformerBlock(norm_type="ada_norm_zero")
                    from diffusers, using PerTokenAdaLN.  Contains a no-op hook
                    `_condition_forward` that subclasses can override to add
                    cross-attention or any other extra conditioning.

  SpatialDiT      — full backbone: patchify cat[noisy_z, puzzle_map], compute
                    per-patch sinusoidal T embeddings, run blocks, unpatchify.
                    puzzle_tokens → one-hot spatial map (channel concat, not
                    cross-attention) so conditioning cannot be bypassed.

  LatentSpatialDiT — wraps SpatialDiT with a frozen VAE, EMA teacher, and the
                    same Path A / Path B spatial training loop as SpatialLatentUNet.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from tqdm.auto import tqdm

from eval.mnist_eval import evaluate_grids, make_panel_image
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.spatial_diffusion_utils import (
    add_noise_spatial,
    add_noise_spatial_c,
    ddim_step_spatial,
    ddim_step_spatial_c,
    gaussian_nll_loss,
    smooth_noise_field,
)

# ── Per-token adaLN ────────────────────────────────────────────────────────────


class PerTokenAdaLN(nn.Module):
    """
    Per-token variant of AdaLayerNormZero (diffusers).

    Standard adaLN-Zero takes a global emb (B, D) and broadcasts shift/scale/gate
    to all tokens.  Here emb may be (B, N, D), giving each token its own
    modulation.  When emb is (B, D) (or all rows of (B, N, D) are equal) the
    result is identical to standard AdaLayerNormZero — backward-compatible.

    Zero-init on the linear layer so blocks start as identity (adaLN-Zero convention).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """
        x   : (B, N, D)
        emb : (B, N, D_t) per-token  OR  (B, D_t) global
        Returns normalised x and (gate_msa, shift_mlp, scale_mlp, gate_mlp),
        each (B, N, D) — matching the per-token shape of x.
        """
        mods = self.linear(self.silu(emb))  # (B, N, 6D) or (B, 6D)
        per_token = mods.dim() == 3
        if per_token:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        else:
            # Global: chunk on dim=1, then unsqueeze to broadcast — identical to
            # AdaLayerNormZero behaviour when the same emb is repeated for all tokens.
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=1)
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ── DiT block ──────────────────────────────────────────────────────────────────


class SpatialDiTBlock(nn.Module):
    """
    DiT block mirroring BasicTransformerBlock(norm_type='ada_norm_zero') from
    diffusers, using PerTokenAdaLN so timestep conditioning can be per-token.

    Extension point
    ---------------
    Override `_condition_forward(hidden_states, **kwargs) -> hidden_states` in
    a subclass to add cross-attention or any other extra conditioning step
    between self-attention and the feed-forward layer.  The base implementation
    is a no-op so the block behaves as a plain DiT block by default.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_bias: bool = True,
    ):
        super().__init__()
        self.ada_norm = PerTokenAdaLN(d_model)
        self.attn = Attention(
            query_dim=d_model,
            heads=n_heads,
            dim_head=d_model // n_heads,
            dropout=dropout,
            bias=attention_bias,
            out_bias=True,
        )
        # Second LayerNorm for the feed-forward (mirrors norm3 in diffusers block)
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(d_model, mult=mlp_ratio, activation_fn="gelu-approximate", dropout=dropout)

    def _condition_forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Override to add cross-attention or other conditioning."""
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, t_emb: torch.Tensor, **condition_kwargs) -> torch.Tensor:
        """
        hidden_states : (B, N, D)
        t_emb         : (B, N, D_t) per-token  OR  (B, D_t) global
        condition_kwargs: passed through to _condition_forward
        """
        norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_norm(hidden_states, t_emb)

        # Self-attention — gate is (B, N, D), matches attn_output shape exactly
        attn_output = self.attn(norm_hs)
        hidden_states = gate_msa * attn_output + hidden_states

        # Conditioning hook (no-op by default)
        hidden_states = self._condition_forward(hidden_states, **condition_kwargs)

        # Feed-forward with per-token shift/scale (mirrors ada_norm_zero FF step)
        norm_hs = self.norm_ff(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hs)
        hidden_states = gate_mlp * ff_output + hidden_states

        return hidden_states


# ── DiT backbone ───────────────────────────────────────────────────────────────


class SpatialDiT(nn.Module):
    """
    Patch-based DiT with per-patch sinusoidal timestep embedding and puzzle
    conditioning via channel concatenation.

    Forward inputs
    --------------
    noisy_z       : (B, C_latent, H, W)   noisy latent
    T_field       : (B, 1, H, W)          spatial timestep field (float or long)
    puzzle_tokens : (B, 81) long          0=null, 1=blank, 2-10=given digit

    The puzzle map is created from puzzle_tokens as a one-hot spatial map at
    latent resolution and concatenated with noisy_z before patchification.
    T_field is averaged per patch → sinusoidal + MLP → per-token embedding
    fed into each block's PerTokenAdaLN.

    Outputs (B, out_channels, H, W).
    """

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        vocab_size: int,
        latent_size: int,
        patch_size: int,
        n_heads: int,
        attention_head_dim: int,
        n_layers: int,
        t_freq_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.latent_size = latent_size
        d_model = n_heads * attention_head_dim
        n_patches = (latent_size // patch_size) ** 2
        in_channels = latent_channels + vocab_size  # puzzle map concatenated

        # Patch embedding (Conv2d equivalent via view + linear)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Per-patch sinusoidal timestep embedding (same as global but applied per token)
        self.time_proj = Timesteps(t_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(t_freq_dim, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([SpatialDiTBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])

        # Final per-token adaLN + output projection (zero-init → identity at start)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        self.proj_out = nn.Linear(d_model, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _puzzle_to_map(self, puzzle_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long → (B, vocab_size, latent_size, latent_size) one-hot."""
        B = puzzle_tokens.shape[0]
        onehot = F.one_hot(puzzle_tokens.clamp(0, self.vocab_size - 1), self.vocab_size).float()
        onehot = onehot.transpose(1, 2).reshape(B, self.vocab_size, 9, 9)
        return F.interpolate(onehot, size=(self.latent_size, self.latent_size), mode="nearest")

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, n_patches, C*p*p)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B, (H // p) * (W // p), C * p * p)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_patches, p*p*C_out) → (B, C_out, H, W)."""
        B, n, _ = x.shape
        p = self.patch_size
        g = self.latent_size // p
        C = self.out_channels
        x = x.reshape(B, g, g, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(B, C, g * p, g * p)

    def _patch_t_embed(self, T_field: torch.Tensor) -> torch.Tensor:
        """
        T_field (B, 1, H, W) → per-patch mean → sinusoidal + MLP → (B, N, d_model).
        When T_field is uniform the result is identical to global timestep conditioning.
        """
        B = T_field.shape[0]
        p = self.patch_size
        g = self.latent_size // p
        # Average T over each patch's spatial region
        T = T_field.squeeze(1).float()  # (B, H, W)
        T = T.reshape(B, g, p, g, p).mean(dim=(2, 4))  # (B, g, g)
        T_flat = T.reshape(B * g * g)  # (B*N,)
        freq = self.time_proj(T_flat)  # (B*N, t_freq_dim)
        emb = self.time_embed(freq)  # (B*N, d_model)
        return emb.reshape(B, g * g, -1)  # (B, N, d_model)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_z: torch.Tensor,
        T_field: torch.Tensor,
        puzzle_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # Build input: concat noisy latent + one-hot puzzle map
        puzzle_map = self._puzzle_to_map(puzzle_tokens)  # (B, vocab_size, H, W)
        inp = torch.cat([noisy_z, puzzle_map], dim=1)  # (B, C_in, H, W)

        # Patchify + positional embedding
        x = self.patch_embed(self._patchify(inp))  # (B, N, d_model)
        x = x + self.pos_embed

        # Per-patch timestep embedding
        t_emb = self._patch_t_embed(T_field)  # (B, N, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final per-token adaLN (mirrors final norm in standard DiT)
        shift, scale = self.final_mod(t_emb).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.proj_out(x)  # (B, N, p*p*C_out)

        return self._unpatchify(x)  # (B, C_out, H, W)


# ── Full latent model ──────────────────────────────────────────────────────────


class LatentSpatialDiT(nn.Module):
    """
    Latent spatial diffusion model using SpatialDiT as backbone.
    Same training interface as SpatialLatentUNet (train_step / eval_step).
    Uses channel-concatenated puzzle conditioning — no ConditionEncoder needed.
    """

    def __init__(
        self,
        model_cfg,  # SpatialLatentConfig (extended with DiT fields)
        optim_cfg,
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
        self.dit = SpatialDiT(
            latent_channels=C,
            out_channels=C + 1,  # x0_pred + log_var
            vocab_size=model_cfg.vocab_size,
            latent_size=model_cfg.latent_size,
            patch_size=model_cfg.patch_size,
            n_heads=model_cfg.n_heads,
            attention_head_dim=model_cfg.attention_head_dim,
            n_layers=model_cfg.n_layers,
            t_freq_dim=model_cfg.t_freq_dim,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )

        # EMA teacher — same architecture, frozen weights, EMA-updated after each step
        self.teacher = copy.deepcopy(self.dit)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.eval_clf = eval_clf
        if self.eval_clf is not None:
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def build_optimizers(self, world_size: int, num_steps: int) -> list:
        optim = torch.optim.AdamW(self.dit.parameters(), lr=0, weight_decay=self.optim_cfg.weight_decay)
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

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scaling_factor).sample.clamp(0.0, 1.0)

    def _run_model(
        self,
        z: torch.Tensor,
        T_field: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        model: SpatialDiT,
    ):
        """Run backbone, split output into (x0_pred, log_var)."""
        out = model(z, T_field, puzzle_tokens)  # (B, C+1, H, W)
        C = self.model_cfg.latent_channels
        return out[:, :C], out[:, C:]

    def _p_refine(self, step: int) -> float:
        cfg = self.model_cfg
        if cfg.p_refine_warmup_steps <= 0:
            return cfg.p_refine_max
        return min(cfg.p_refine_max, step / cfg.p_refine_warmup_steps * cfg.p_refine_max)

    @torch.no_grad()
    def _update_teacher(self):
        mu = self.model_cfg.teacher_ema_rate
        for tp, sp in zip(self.teacher.parameters(), self.dit.parameters()):
            tp.data.mul_(mu).add_(sp.data, alpha=1.0 - mu)

    def _make_t_field(self, t_base, tau, perlin, T_max):
        t = t_base[:, None, None, None].float() + tau * perlin
        t = t.clamp(0, T_max - 1)
        return t if self.model_cfg.continuous_time else t.long()

    def _add_noise(self, x0, noise, T_field, T_max):
        if self.model_cfg.continuous_time:
            return add_noise_spatial_c(x0, noise, T_field, T_max)
        return add_noise_spatial(x0, noise, T_field, self.alphas_cumprod)

    def _ddim_step(self, z, x0_pred, T_old, T_new, T_max):
        if self.model_cfg.continuous_time:
            return ddim_step_spatial_c(z, x0_pred, T_old, T_new, T_max)
        return ddim_step_spatial(z, x0_pred, T_old, T_new, self.alphas_cumprod)

    # ── Training step ──────────────────────────────────────────────────────────

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        global_batch_size,
        global_step,
        cfg_prob=0.0,
    ):
        K = len(micro_batches)
        device = accelerator.device
        T_max = self.scheduler.config.num_train_timesteps
        p_refine = self._p_refine(global_step)
        total_loss = 0.0
        total_comps: dict = {}

        for mb in micro_batches:
            images = mb["images"].to(device)
            puzzle_tokens = mb["puzzle_tokens"].to(device)  # (B, 81) — puzzle, not solution
            B = images.shape[0]

            # CFG dropout: replace with all-blank (token 1) conditioning
            if cfg_prob > 0 and torch.rand(1).item() < cfg_prob:
                puzzle_tokens = torch.ones_like(puzzle_tokens)

            z0 = self._encode(images)
            _, _, lH, lW = z0.shape
            t_base = torch.randint(0, T_max, (B,), device=device).float()

            if self.model_cfg.noise_mode == "teacher_guided":
                in_stage1 = global_step < self.model_cfg.stage1_steps
                T_uni_f = t_base[:, None, None, None].expand(B, 1, lH, lW)
                T_uni = T_uni_f if self.model_cfg.continuous_time else T_uni_f.long()

                if in_stage1:
                    z_n = self._add_noise(z0, torch.randn_like(z0), T_uni, T_max)
                    x0_pred, log_var = self._run_model(z_n, T_uni, puzzle_tokens, self.dit)
                else:
                    z_u = self._add_noise(z0, torch.randn_like(z0), T_uni, T_max)
                    with torch.no_grad():
                        _, log_var_t = self._run_model(z_u, T_uni, puzzle_tokens, self.teacher)
                    U_aug = self._augment_mask(log_var_t.sigmoid(), lH, lW, device)
                    T_s = self._make_t_field(t_base, self.model_cfg.tau_student, U_aug, T_max)
                    z_s = self._add_noise(z0, torch.randn_like(z0), T_s, T_max)
                    x0_pred, log_var = self._run_model(z_s, T_s, puzzle_tokens, self.dit)
            else:
                # Perlin mode
                perlin = smooth_noise_field(
                    B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves
                )
                T_init = self._make_t_field(t_base, self.model_cfg.tau_init, perlin, T_max)
                if torch.rand(1).item() < p_refine:
                    z_i = self._add_noise(z0, torch.randn_like(z0), T_init, T_max)
                    with torch.no_grad():
                        _, log_var_t = self._run_model(z_i, T_init, puzzle_tokens, self.teacher)
                    U_t = log_var_t.sigmoid()
                    perlin2 = smooth_noise_field(
                        B, lH, lW, self.model_cfg.f_spatial, device, n_octaves=self.model_cfg.n_octaves
                    )
                    T_s = self._make_t_field(t_base, self.model_cfg.tau_student, U_t * perlin2, T_max)
                    z_s = self._add_noise(z0, torch.randn_like(z0), T_s, T_max)
                    x0_pred, log_var = self._run_model(z_s, T_s, puzzle_tokens, self.dit)
                else:
                    z_n = self._add_noise(z0, torch.randn_like(z0), T_init, T_max)
                    x0_pred, log_var = self._run_model(z_n, T_init, puzzle_tokens, self.dit)

            loss, comps = gaussian_nll_loss(z0, x0_pred, log_var)
            total_loss += loss.item()
            for k, v in comps.items():
                total_comps[k] = total_comps.get(k, 0.0) + v
            accelerator.backward(loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self.dit.parameters(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        self._update_teacher()
        global_step += 1

        metrics = {"nll_loss": total_loss / K, **{k: v / K for k, v in total_comps.items()}}
        if self.model_cfg.noise_mode == "teacher_guided":
            metrics["stage"] = 1 if (global_step - 1) < self.model_cfg.stage1_steps else 2
        else:
            metrics["p_refine"] = p_refine
        return metrics, lr, global_step

    def _augment_mask(self, U, lH, lW, device):
        """Same augmentation roulette as SpatialLatentUNet._augment_mask."""
        cfg = self.model_cfg
        roll = torch.rand(1).item()
        cum_v = cfg.aug_prob_vanilla
        cum_p = cum_v + cfg.aug_prob_power
        cum_t = cum_p + cfg.aug_prob_threshold
        if roll < cum_v:
            return U
        if roll < cum_p:
            gamma = cfg.power_gamma_min + torch.rand(1).item() * (cfg.power_gamma_max - cfg.power_gamma_min)
            return U.pow(gamma)
        if roll < cum_t:
            n = torch.randint(cfg.threshold_n_min, cfg.threshold_n_max + 1, (1,)).item()
            splits = sorted(
                cfg.threshold_val_min + torch.rand(n).tolist()[i] * (cfg.threshold_val_max - cfg.threshold_val_min)
                for i in range(n)
            )
            boundaries = torch.tensor(splits, device=device, dtype=torch.float32)
            bucket = torch.bucketize(U.squeeze(1), boundaries)
            return (bucket.float() / n).unsqueeze(1)
        B = U.shape[0]
        perlin = smooth_noise_field(B, lH, lW, cfg.f_spatial, device, n_octaves=cfg.n_octaves)
        return (U * perlin).clamp(0.0, 1.0)

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

        # Validation loss
        val_losses, val_comps = [], {}
        for i, batch in enumerate(tqdm(dataloader, desc="Eval loss", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            puzzle_tokens = batch["puzzle_tokens"].to(device)
            B = images.shape[0]
            z0 = self._encode(images)
            _, _, lH, lW = z0.shape
            t_base = torch.randint(0, T_max, (B,), device=device).float()
            perlin = smooth_noise_field(B, lH, lW, self.model_cfg.f_spatial, device)
            T_field = self._make_t_field(t_base, self.model_cfg.tau_init, perlin, T_max)
            z_n = self._add_noise(z0, torch.randn_like(z0), T_field, T_max)
            x0_pred, log_var = self._run_model(z_n, T_field, puzzle_tokens, self.dit)
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

            for batch in tqdm(dataloader, desc="Sampling (spatial)", leave=False):
                if n_done >= n_total:
                    break
                solutions = batch["solution"]
                puzzle_tokens = batch["puzzle_tokens"].to(device)
                given_masks = batch.get("given_mask")
                conditions_pixel = batch.get("conditions")
                B = puzzle_tokens.shape[0]

                null_tokens = torch.ones_like(puzzle_tokens) if cfg_scale > 1.0 else None

                z = torch.randn(B, C, lH, lW, device=device)
                _dtype = torch.float32 if self.model_cfg.continuous_time else torch.long
                T_field = torch.full((B, 1, lH, lW), float(T_max - 1), device=device, dtype=_dtype)

                for _ in range(num_ddim_steps):
                    T_old = T_field.clone()
                    x0_pred, log_var = self._run_model(z, T_field, puzzle_tokens, self.dit)
                    if cfg_scale > 1.0:
                        x0_uncond, _ = self._run_model(z, T_field, null_tokens, self.dit)
                        x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)
                    uncertainty = log_var.sigmoid()
                    T_new = (T_field.float() - dt * (1.0 - uncertainty)).clamp(0, T_max - 1)
                    if not self.model_cfg.continuous_time:
                        T_new = T_new.round().long()
                    z = self._ddim_step(z, x0_pred, T_old, T_new, T_max)
                    T_field = T_new
                    if T_field.max() < 0.5:
                        break

                generated = self._decode(z)
                acc = evaluate_grids(generated, solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
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

            if panel_images and step is not None:
                try:
                    import wandb

                    tracker = accelerator.get_tracker("wandb", unwrap=True)
                    tracker.log({"val/examples": [wandb.Image(img) for img in panel_images]}, step=step)
                except Exception:
                    pass

        self.train()
        return result
