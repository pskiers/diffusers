"""
Embedded-TRM Bit-Diffusion for Sokoban board generation (ISOLATED implementation).

This is a standalone iteration in which the TRM recursion lives *strictly inside*
the Transformer block (``EmbeddedTRMBlock``). It does NOT import or modify the
Standard (``train_std.py``) or basic TRM (``train_trm.py``) model code; the only
shared dependencies are the data pipeline (``SokobanBitsDataset``) and the Sokoban
evaluation metrics, which are domain utilities rather than model implementations.

Architecture (parsed from schema.drawio.xml):
    board 12x12x3 -> patchify -> 144 x D sequence
    EmbeddedTRMBlock (TRM strictly inside the transformer block):
        N-loop : z = norm_z(f_z(x + y + z))        (n_inner iterations)
        y-step : y = norm_y(f_y(y + z))            (separate sub-stack, per schema)
        T-loop : T-1 no-grad iterations + 1 with-grad iteration (1-step gradient)
    x0 head + Q-head (ACT). Conditioning (timestep + class) via AdaLN-Zero.

Optimization scheme (see module docstring of training_step):
    The n_sup(t) deep-supervision loop is driven by the LightningModule, NOT by the
    nn.Module. Per supervision step we compute the loss, call backward() immediately
    (gradient accumulation), then detach the carry (y, z). A single clipped
    optimizer step is taken after the loop. This realizes "accumulate losses ->
    one global step" at O(1)-in-n_sup memory and with no BPTT across steps.

All inputs are shaped as in the previous version: 12x12 board with 3-bit numbers.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings, PatchEmbed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from sokoban.dataset.evaluate_sokoban_boards import generate_metrics
from sokoban.dataset.sokoban_dataset import SokobanBitsDataset


# --------------------------------------------------------------------------------------
# Schedules (standalone copies to keep this file isolated from train_trm.py)
# --------------------------------------------------------------------------------------
def n_sup_schedule(t: torch.Tensor, n_min: int, n_max: int) -> torch.Tensor:
    """Per-sample deep-supervision budget n_sup(t) = n_min + (n_max-n_min)*sin(pi*t).

    More refinement at medium noise (t~0.5 -> n_max), least at the extremes (t~0/1).
    """
    return (n_min + (n_max - n_min) * torch.sin(math.pi * t)).round().long().clamp(min=n_min, max=n_max)


def halt_threshold(t_val: float) -> float:
    """Time-dependent Q-head halting threshold used at inference only."""
    return 0.95 - 0.4 * t_val


# --------------------------------------------------------------------------------------
# Transformer sub-layer (DiT-style AdaLN-Zero) used as the recursion's block-function f
# --------------------------------------------------------------------------------------
class AdaLNZeroSubLayer(nn.Module):
    """One Attention + FeedForward unit with AdaLN-Zero conditioning on (B, L, D).

    Conditioning ``cond`` (B, D) is embedded once per forward and reused across all
    recursion iterations. The modulation projection is zero-initialized so the layer
    starts as the identity (gates = 0) -> stable warm-up for the recursive loops.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_mult: int,
        activation_fn: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(query_dim=dim, heads=num_heads, dim_head=head_dim, dropout=dropout, bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, mult=ffn_mult, activation_fn=activation_fn, dropout=dropout)

        # AdaLN-Zero: produce (shift/scale/gate) for MSA and MLP; zero-init -> identity start.
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(F.silu(cond)).chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + gate_msa[:, None] * self.attn(h)
        h = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.ff(h)
        return x


# --------------------------------------------------------------------------------------
# The Transformer block with the TRM recursion embedded INSIDE it
# --------------------------------------------------------------------------------------
class EmbeddedTRMBlock(nn.Module):
    """Transformer block whose forward IS the TRM recursion.

    Two latent states of shape (B, L, D):
      - z : low-level "scratch" updated n_inner times in the N-loop
      - y : high-level "solution" updated once per latent recursion
    Per the schema, the z-update and y-update use separate Attention/FFN sub-stacks.

    deep_recursion applies the 1-step-gradient policy: the first ``T-1`` latent
    recursions run under ``torch.no_grad`` (no graph), and only the final one builds
    a graph. This keeps the backward graph shallow regardless of T.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_mult: int,
        activation_fn: str,
        dropout: float,
        n_inner: int,
        T: int,
        num_inner_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_inner = n_inner
        self.T = T
        self.z_layers = nn.ModuleList(
            [AdaLNZeroSubLayer(dim, num_heads, head_dim, ffn_mult, activation_fn, dropout) for _ in range(num_inner_layers)]
        )
        self.y_layers = nn.ModuleList(
            [AdaLNZeroSubLayer(dim, num_heads, head_dim, ffn_mult, activation_fn, dropout) for _ in range(num_inner_layers)]
        )
        # Recurrent normalizers (affine) bound the magnitude of every state write.
        self.norm_z = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

    @staticmethod
    def _apply(layers: nn.ModuleList, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in layers:
            h = layer(h, cond)
        return h

    def _latent_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One latent recursion: n_inner z-updates (N-loop) followed by one y-update."""
        for _ in range(self.n_inner):
            z = self.norm_z(self._apply(self.z_layers, x + y + z, cond)).float()
        y = self.norm_y(self._apply(self.y_layers, y + z, cond)).float()
        return y, z

    def deep_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """T latent recursions with the 1-step gradient (T-1 no-grad + 1 with-grad)."""
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x, y, z, cond)
        y, z = self._latent_recursion(x, y, z, cond)  # only this iteration carries gradient
        return y, z


# --------------------------------------------------------------------------------------
# Full diffusion denoiser: patchify -> embedded-TRM block -> x0 head + Q-head
# --------------------------------------------------------------------------------------
class EmbeddedTRMDiffusion(nn.Module):
    """Standalone TRM-inside-transformer denoiser for bit-diffusion.

    forward() is pure (no optimizer / no loss): the training loop drives the n_sup
    supervision loop and owns optimization. This module only exposes the building
    blocks: ``embed_inputs`` -> ``step`` (one supervised refinement) -> ``decode`` / ``q_logit``.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_embeds_ada_norm: int,
        n_inner: int = 6,
        T: int = 3,
        num_inner_layers: int = 1,
        ffn_mult: int = 4,
        activation_fn: str = "gelu-approximate",
        dropout: float = 0.0,
        patch_size: int = 1,
        use_grid_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.h_p = resolution // patch_size
        self.w_p = resolution // patch_size
        seq_len = self.h_p * self.w_p

        # Patchify (Conv2d + fixed 2D sincos positional embedding).
        self.patch = PatchEmbed(
            height=resolution, width=resolution, patch_size=patch_size,
            in_channels=in_channels, embed_dim=dim, pos_embed_type="sincos",
        )
        # Optional learnable grid positional embedding (helps structured-grid tasks).
        self.grid_pos = nn.Parameter(torch.zeros(1, seq_len, dim)) if use_grid_pos_embed else None
        if self.grid_pos is not None:
            nn.init.trunc_normal_(self.grid_pos, std=0.02)

        # Single conditioning embedding (timestep + class) shared by AdaLN and the head.
        # class_dropout_prob=0: CFG label dropout is handled explicitly in the training loop.
        self.cond_embed = CombinedTimestepLabelEmbeddings(num_embeds_ada_norm, dim, class_dropout_prob=0.0)

        # The transformer block with the TRM recursion inside.
        self.block = EmbeddedTRMBlock(
            dim, num_attention_heads, attention_head_dim, ffn_mult, activation_fn, dropout, n_inner, T, num_inner_layers
        )

        # Fixed (non-trained) initial latent states.
        self.register_buffer("y_init", torch.randn(1, seq_len, dim))
        self.register_buffer("z_init", torch.randn(1, seq_len, dim))

        # x0 output head (AdaLN-style modulation -> unpatchify).
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_2 = nn.Linear(dim, patch_size * patch_size * out_channels)

        # Q-head: predicts board accuracy of the current x0 estimate (ACT halting).
        # bias=-2 -> sigmoid~0.12 at init, calibrates quickly toward the graded target.
        self.q_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.q_head.weight)
        with torch.no_grad():
            self.q_head.bias.fill_(-2.0)

    # -- building blocks -------------------------------------------------------------
    def embed_inputs(
        self, sample: torch.Tensor, timestep: torch.Tensor, class_labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, C, H, W) -> token features x (B, L, D) and conditioning cond (B, D)."""
        x = self.patch(sample)
        if self.grid_pos is not None:
            x = x + self.grid_pos
        x = x.float()
        cond = self.cond_embed(timestep, class_labels, hidden_dtype=x.dtype).float()
        return x, cond

    def get_initial_carry(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.y_init.expand(batch_size, -1, -1).clone().float()
        z = self.z_init.expand(batch_size, -1, -1).clone().float()
        return y, z

    def step(
        self, x: torch.Tensor, cond: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One supervised refinement step.

        Returns:
            y_final : attached (gradient-carrying) high-level state for head + Q-head
            (y, z)  : DETACHED carry for the next supervision step (breaks BPTT)
        """
        y_final, z_final = self.block.deep_recursion(x, y, z, cond)
        return y_final, (y_final.detach(), z_final.detach())

    def _unpatchify(self, h: torch.Tensor) -> torch.Tensor:
        p, c, hp, wp = self.patch_size, self.out_channels, self.h_p, self.w_p
        h = h.reshape(-1, hp, wp, p, p, c)
        h = torch.einsum("nhwpqc->nchpwq", h)
        return h.reshape(-1, c, hp * p, wp * p)

    def decode(self, y_final: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Map the high-level latent to a model output image (B, out_channels, H, W)."""
        shift, scale = self.proj_out_1(F.silu(cond)).chunk(2, dim=1)
        h = self.norm_out(y_final) * (1 + scale[:, None]) + shift[:, None]
        h = self.proj_out_2(h)
        return self._unpatchify(h)

    def q_logit(self, y_final: torch.Tensor) -> torch.Tensor:
        """Halting logit from mean-pooled latent -> (B,)."""
        return self.q_head(y_final.mean(dim=1)).squeeze(-1)

    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_steps: int = 1,
        early_stop: bool = False,
        q_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Inference-only refinement (optionally with per-sample Q-head halting)."""
        x, cond = self.embed_inputs(sample, timestep, class_labels)
        bsz = sample.shape[0]
        y, z = carry if carry is not None else self.get_initial_carry(bsz)

        if not early_stop:
            y_final = y
            for _ in range(n_steps):
                y_final, (y, z) = self.step(x, cond, y, z)
            return self.decode(y_final, cond), (y, z)

        active = torch.ones(bsz, dtype=torch.bool, device=x.device)
        y_store = y.clone()
        for i in range(n_steps):
            y_final, (y, z) = self.step(x, cond, y, z)
            y_store = torch.where(active[:, None, None], y_final, y_store)  # freeze halted samples
            if i < n_steps - 1:
                halt = self.q_logit(y_final).sigmoid() > q_threshold
                active = active & (~halt)
                if not active.any():
                    break
        return self.decode(y_store, cond), (y, z)


# --------------------------------------------------------------------------------------
# LightningModule: owns the n_sup loop, loss accumulation and the single optimizer step
# --------------------------------------------------------------------------------------
class SokobanEmbeddedTRMDiffusion(L.LightningModule):
    """Bit-diffusion trainer for the embedded-TRM denoiser.

    Conditioning modes: "unconditional", "num_boxes", "k_steps".
    """

    def __init__(
        self,
        model: EmbeddedTRMDiffusion,
        noise_scheduler: DDPMScheduler,
        conditioning: str = "unconditional",
        num_classes: int = 0,
        num_bits: int = 3,
        resolution: int = 12,
        inference_steps: int = 300,
        self_cond: bool = True,
        cfg_drop_rate: float = 0.1,
        guidance_scale: float = 4.0,
        time_shift_xi: float = 0.0,
        n_sup_min: int = 2,
        n_sup_max: int = 6,
        q_loss_weight: float = 0.15,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        num_epochs: int = 300,
        eval_every_n_epochs: int = 50,
        num_eval_samples: int = 128,
        k_values: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_scheduler"])
        # Manual optimization: we accumulate per-step gradients then step once per batch.
        self.automatic_optimization = False

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.conditioning = conditioning
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.resolution = resolution
        self.inference_steps = inference_steps
        self.self_cond = self_cond
        self.cfg_drop_rate = cfg_drop_rate
        self.guidance_scale = guidance_scale
        self.time_shift_xi = time_shift_xi
        self.n_sup_min = n_sup_min
        self.n_sup_max = n_sup_max
        self.q_loss_weight = q_loss_weight
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_eval_samples = num_eval_samples

        n_params = sum(p.numel() for p in model.parameters())
        print(f"EmbeddedTRMDiffusion parameters: {n_params:,}")

        if conditioning == "k_steps" and k_values:
            k_to_class = torch.zeros(max(k_values) + 1, dtype=torch.long)
            for idx, k in enumerate(k_values):
                k_to_class[k] = idx
            self.register_buffer("k_to_class", k_to_class)
        else:
            self.k_to_class = None

    # -- conditioning / input helpers ------------------------------------------------
    def _extract_class_labels(self, batch: dict) -> torch.Tensor:
        if self.conditioning == "num_boxes":
            return (batch["num_boxes"] - 1).long()
        if self.conditioning == "k_steps":
            return self.k_to_class[batch["k"]]  # type: ignore[index]
        return torch.full(
            (batch["target"].shape[0],), self.num_classes, dtype=torch.long, device=batch["target"].device
        )

    def _apply_cfg_dropout(
        self, class_labels: torch.Tensor, cond_board: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cfg_drop_rate <= 0:
            return class_labels, cond_board
        drop = torch.rand(class_labels.shape[0], device=class_labels.device) < self.cfg_drop_rate
        class_labels = torch.where(drop, self.num_classes, class_labels)
        if cond_board is not None:
            cond_board = torch.where(drop.view(-1, 1, 1, 1), torch.zeros_like(cond_board), cond_board)
        return class_labels, cond_board

    def _build_model_input(
        self, x_t: torch.Tensor, x_pred: Optional[torch.Tensor], cond_board: Optional[torch.Tensor]
    ) -> torch.Tensor:
        parts = [x_t]
        if self.self_cond:
            parts.append(x_pred if x_pred is not None else torch.zeros_like(x_t))
        if cond_board is not None:
            parts.append(cond_board)
        return torch.cat(parts, dim=1)

    def _snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Per-sample Min-SNR-gamma weights (gamma=5) for x0-prediction MSE."""
        ac = self.noise_scheduler.alphas_cumprod.to(self.device)
        a = ac.gather(-1, timesteps)
        snr = a / (1.0 - a).clamp(min=1e-5)
        return snr.clamp(max=5.0)

    def _to_x0(self, model_out: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Recover an x0 estimate from the raw model output (for the Q-head target)."""
        if self.noise_scheduler.config.prediction_type == "sample":
            return model_out
        ac = self.noise_scheduler.alphas_cumprod.to(self.device).gather(-1, timesteps).view(-1, 1, 1, 1)
        return (x_t - (1.0 - ac).sqrt() * model_out) / ac.sqrt().clamp(min=1e-4)

    # -- training: accumulate-then-single-step --------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> None:
        """Deep-supervision over n_sup(t) steps with gradient accumulation.

        For each supervision step i:
          1. run one embedded-TRM refinement (1-step gradient inside the block);
          2. compute the per-sample diffusion + Q-head loss for step i;
          3. backward() IMMEDIATELY -> accumulates into .grad and frees step i's graph
             (this is why peak memory is O(1) in n_sup, not O(n_sup));
          4. the carry (y, z) returned by step() is already DETACHED -> no BPTT across
             supervision steps, so the per-step gradients are local 1-step gradients.
        A single clipped optimizer step is applied after the loop. NOTE: we detach the
        *carry*, never the loss tensor (a detached loss cannot be backpropagated).
        """
        opt = self.optimizers()
        x_bits = batch["target"]
        bsz = x_bits.shape[0]
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)

        num_train = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train, (bsz,), device=self.device).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)
        labels_tr, cond_tr = self._apply_cfg_dropout(class_labels, cond_board)

        # Optional self-conditioning: a cheap 1-step no-grad pass to seed x_pred.
        x_pred = None
        if self.self_cond and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                mi0 = self._build_model_input(x_t, None, cond_tr)
                x0_emb, cond0 = self.model.embed_inputs(mi0, timesteps, labels_tr)
                y0, z0 = self.model.get_initial_carry(bsz)
                yf0, _ = self.model.step(x0_emb, cond0, y0, z0)
                x_pred = self.model.decode(yf0, cond0).detach().clamp(-1.0, 1.0)

        model_input = self._build_model_input(x_t, x_pred, cond_tr)  # fixed (no-grad) inputs
        y, z = self.model.get_initial_carry(bsz)

        # Per-sample supervision budget from the noise-dependent schedule.
        t_norm = timesteps.float() / float(num_train - 1)
        n_sup = n_sup_schedule(t_norm, self.n_sup_min, self.n_sup_max)  # (B,)
        inv_budget = 1.0 / n_sup.float()                                # equal weight per sample
        max_steps = int(n_sup.max().item())

        sample_pred = self.noise_scheduler.config.prediction_type == "sample"
        snr_w = self._snr_weights(timesteps) if sample_pred else torch.ones(bsz, device=self.device)

        opt.zero_grad()
        diff_log = torch.zeros((), device=self.device)
        q_log = torch.zeros((), device=self.device)

        for i in range(max_steps):
            active = (i < n_sup).float()  # samples whose budget still includes step i

            # Re-embed inside the loop: patch-embed and conditioning depend on the
            # shared params, so each step must build its OWN graph rooted at the fixed
            # model_input. Computing them once outside would mean the first per-step
            # backward() frees that shared subgraph and the next step's backward() would
            # raise "backward through the graph a second time". model_input is a leaf
            # (no grad), so re-embedding is cheap and keeps each step's graph independent.
            x, cond = self.model.embed_inputs(model_input, timesteps, labels_tr)
            y_final, (y, z) = self.model.step(x, cond, y, z)  # carry returned DETACHED
            model_out = self.model.decode(y_final, cond)

            # Diffusion loss (per sample) -------------------------------------------
            if sample_pred:
                mse = F.mse_loss(model_out.float(), x_bits.float(), reduction="none").mean(dim=(1, 2, 3))
                diff_ps = snr_w * mse
            else:
                mse = F.mse_loss(model_out.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
                diff_ps = mse

            # Q-head loss (graded board-accuracy soft target) -----------------------
            with torch.no_grad():
                x0_est = self._to_x0(model_out, x_t, timesteps)
                cell_correct = (x0_est.sign() == x_bits).all(dim=1)         # (B, H, W)
                board_acc = cell_correct.float().mean(dim=(1, 2))           # (B,) in [0, 1]
            q_ps = F.binary_cross_entropy_with_logits(self.model.q_logit(y_final), board_acc, reduction="none")

            per_sample = diff_ps + self.q_loss_weight * q_ps
            # Weight each active sample by 1/budget and average over the batch so the
            # accumulated gradient equals the mean of per-sample, per-step 1-step gradients.
            step_loss = (per_sample * inv_budget * active).sum() / bsz

            self.manual_backward(step_loss)  # accumulate into .grad, free this step's graph
            diff_log = diff_log + (diff_ps * inv_budget * active).sum().detach() / bsz
            q_log = q_log + (q_ps * inv_budget * active).sum().detach() / bsz

        # Single global, gradient-clipped optimizer step for the whole batch.
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt.step()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train/loss", diff_log + self.q_loss_weight * q_log, prog_bar=True, sync_dist=True)
        self.log("train/diff_loss", diff_log, sync_dist=True)
        self.log("train/q_loss", q_log, sync_dist=True)
        self.log("train/max_steps", float(max_steps), sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x_bits = batch["target"]
        bsz = x_bits.shape[0]
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)

        num_train = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train, (bsz,), device=self.device).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)
        x_pred = torch.zeros_like(x_t) if self.self_cond else None
        model_input = self._build_model_input(x_t, x_pred, cond_board)

        out, _ = self.model(model_input, timesteps, class_labels, n_steps=self.n_sup_max, early_stop=False)
        if self.noise_scheduler.config.prediction_type == "sample":
            loss = F.mse_loss(out.float(), x_bits.float())
        else:
            loss = F.mse_loss(out.float(), noise.float())
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    # -- sampling --------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        cond_board: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        gs = self.guidance_scale if guidance_scale is None else guidance_scale
        use_cfg = gs > 1.0 and class_labels is not None and self.conditioning != "unconditional"
        sample_pred = self.noise_scheduler.config.prediction_type == "sample"
        num_train = self.noise_scheduler.config.num_train_timesteps

        x_t = torch.randn(batch_size, self.num_bits, self.resolution, self.resolution, device=device, generator=generator)
        x_pred = None
        out = x_t
        self.noise_scheduler.set_timesteps(self.inference_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device)
            if self.time_shift_xi > 0.0:
                t_model = torch.clamp(t_batch + int(self.time_shift_xi * num_train), max=num_train - 1)
            else:
                t_model = t_batch

            t_norm = max(0.0, min(1.0, t.item() / float(num_train - 1)))
            n_steps = int(n_sup_schedule(torch.tensor([t_norm], device=device), self.n_sup_min, self.n_sup_max).item())
            thr = halt_threshold(t_norm)

            if use_cfg:
                uncond_labels = torch.full_like(class_labels, self.num_classes)  # type: ignore[arg-type]
                uncond_board = torch.zeros_like(cond_board) if cond_board is not None else None
                mi_cond = self._build_model_input(x_t, x_pred, cond_board)
                mi_uncond = self._build_model_input(x_t, x_pred, uncond_board)
                mi = torch.cat([mi_cond, mi_uncond], dim=0)
                t_both = torch.cat([t_model, t_model], dim=0)
                labels_both = torch.cat([class_labels, uncond_labels], dim=0)  # type: ignore[arg-type]
                out_both, _ = self.model(mi, t_both, labels_both, n_steps=n_steps, early_stop=True, q_threshold=thr)
                out_cond, out_uncond = out_both.chunk(2, dim=0)
                out = out_uncond + gs * (out_cond - out_uncond)
            else:
                mi = self._build_model_input(x_t, x_pred, cond_board)
                out, _ = self.model(mi, t_model, class_labels, n_steps=n_steps, early_stop=True, q_threshold=thr)

            if sample_pred:
                out = out.clamp(-1.0, 1.0)
            x_t = self.noise_scheduler.step(out, t, x_t).prev_sample
            x_pred = out if (self.self_cond and sample_pred) else None

        return out

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, num_samples: Optional[int] = None) -> Tuple[dict, torch.Tensor]:
        device = self.device
        num_samples = num_samples or self.num_eval_samples
        batch_size = min(50, num_samples)

        gen = torch.Generator(device=device)
        gen.manual_seed(42)

        gen_boards: List[np.ndarray] = []
        target_boards: List[np.ndarray] = []
        gen_bits_all: List[torch.Tensor] = []
        num_boxes_labels: List[int] = []

        it = iter(val_dataloader)
        produced = 0
        while produced < num_samples:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(val_dataloader)
                batch = next(it)
            bsz = min(batch_size, num_samples - produced)
            x_bits = batch["target"][:bsz].to(device)
            class_labels = self._extract_class_labels(
                {k: v[:bsz] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            ).to(device)
            cond_board = batch.get("condition", None)
            if cond_board is not None:
                cond_board = cond_board[:bsz].to(device)

            gen_bits = self.generate_batch(
                batch_size=x_bits.shape[0], device=device, class_labels=class_labels,
                cond_board=cond_board, generator=gen,
            )
            gen_bits_all.append(gen_bits.cpu())
            gen_boards.append(SokobanBitsDataset.bits_to_tokens(gen_bits).cpu().numpy())
            target_boards.append(SokobanBitsDataset.bits_to_tokens(x_bits).cpu().numpy())
            if "num_boxes" in batch:
                num_boxes_labels.extend(batch["num_boxes"][:bsz].tolist())
            produced += x_bits.shape[0]

        gen_np = np.clip(np.concatenate(gen_boards, axis=0)[:num_samples], 0, 6)
        target_np = np.concatenate(target_boards, axis=0)[:num_samples]
        nb = np.array(num_boxes_labels[:num_samples]) if num_boxes_labels else None
        metrics = generate_metrics(generated_boards=gen_np, num_boxes_labels=nb, target_boards=target_np)
        return metrics, torch.cat(gen_bits_all, dim=0)[:num_samples]

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.eval_every_n_epochs != 0 or not self.trainer.is_global_zero:
            return
        val_dl = self.trainer.datamodule.val_dataloader()  # type: ignore[union-attr]
        metrics, _ = self.evaluate(val_dl)
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        warmup_steps = self.hparams.get("warmup_steps", 500)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# --------------------------------------------------------------------------------------
# EMA: with one optimizer step per batch, standard per-batch EMA is valid again.
# --------------------------------------------------------------------------------------
class EmbeddedEMACallback(Callback):
    """Diffusers EMA over model weights; swaps EMA weights in for validation."""

    def __init__(self, decay: float = 0.9999, inv_gamma: float = 1.0, power: float = 0.75) -> None:
        super().__init__()
        self.decay = decay
        self.inv_gamma = inv_gamma
        self.power = power
        self.ema_model: Optional[EMAModel] = None

    def on_fit_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.ema_model = EMAModel(
            pl_module.model.parameters(), decay=self.decay, use_ema_warmup=True,
            inv_gamma=self.inv_gamma, power=self.power,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        assert self.ema_model is not None
        self.ema_model.step(pl_module.model.parameters())

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        assert self.ema_model is not None
        self.ema_model.store(pl_module.model.parameters())
        self.ema_model.copy_to(pl_module.model.parameters())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        assert self.ema_model is not None
        self.ema_model.restore(pl_module.model.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self.ema_model is None:
            return
        for i, (name, _) in enumerate(pl_module.model.named_parameters()):
            key = f"model.{name}"
            if key in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.ema_model.shadow_params[i].clone()


# --------------------------------------------------------------------------------------
# Isolated data module (uses only the shared dataset, no model dependencies)
# --------------------------------------------------------------------------------------
class EmbeddedSokobanDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        val_data_path: Optional[str] = None,
        conditioning: str = "unconditional",
        total_train_size: int = 1000,
        total_eval_size: int = 200,
        batch_size: int = 32,
        num_workers: int = 4,
        num_bits: int = 3,
        k_values: Optional[List[int]] = None,
        use_dihedral_aug: bool = True,
        bot_removal_prob: float = 0.75,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.val_data_path = val_data_path or data_path
        self.conditioning = conditioning
        self.total_train_size = total_train_size
        self.total_eval_size = total_eval_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.k_values = k_values or [3, 8, 10]
        self.use_dihedral_aug = use_dihedral_aug
        self.bot_removal_prob = bot_removal_prob

    def setup(self, stage: Optional[str] = None) -> None:
        if self.conditioning == "num_boxes":
            self.train_ds = SokobanBitsDataset.for_conditioning_num_boxes_generation(self.data_path, self.total_train_size)
            self.val_ds = SokobanBitsDataset.for_conditioning_num_boxes_generation(self.val_data_path, self.total_eval_size)
        elif self.conditioning == "k_steps":
            self.train_ds = SokobanBitsDataset.for_new_k_steps_conditioning_generation(
                self.data_path, self.total_train_size, k_values=self.k_values, bot_removal_prob=self.bot_removal_prob
            )
            self.val_ds = SokobanBitsDataset.for_new_k_steps_conditioning_generation(
                self.val_data_path, self.total_eval_size, k_values=self.k_values, bot_removal_prob=self.bot_removal_prob
            )
        else:
            self.train_ds = SokobanBitsDataset.for_unconditional(
                self.data_path, self.total_train_size, bot_removal_prob=self.bot_removal_prob
            )
            self.val_ds = SokobanBitsDataset.for_unconditional(
                self.val_data_path, self.total_eval_size, bot_removal_prob=self.bot_removal_prob
            )
        for ds, aug in ((self.train_ds, self.use_dihedral_aug), (self.val_ds, False)):
            ds.num_bits = self.num_bits
            ds.use_dihedral_aug = aug

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


@hydra.main(version_base=None, config_path="config", config_name="embedded_trm_diffusion")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.get("seed", 42), workers=True)

    num_bits = cfg.num_bits
    # For k_steps the class label indexes the k-value list, so the embedding table must
    # be sized len(k_values) (+1 for the CFG/unconditional token), not cfg.num_classes.
    k_values = cfg.dataset.get("k_values", [3, 8, 10])
    num_classes = len(k_values) if cfg.conditioning == "k_steps" else cfg.num_classes
    use_self_cond = cfg.get("self_cond", True)
    self_cond_mult = 2 if use_self_cond else 1
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * self_cond_mult + num_bits
    else:
        in_channels = num_bits * self_cond_mult

    model = EmbeddedTRMDiffusion(
        resolution=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_attention_heads=cfg.model.num_attention_heads,
        attention_head_dim=cfg.model.attention_head_dim,
        num_embeds_ada_norm=num_classes + 1,
        n_inner=cfg.trm.n_inner,
        T=cfg.trm.T,
        num_inner_layers=cfg.trm.get("num_inner_layers", 1),
        ffn_mult=cfg.model.get("ffn_mult", 4),
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        patch_size=cfg.model.get("patch_size", 1),
        use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    lit_model = SokobanEmbeddedTRMDiffusion(
        model=model,
        noise_scheduler=noise_scheduler,
        conditioning=cfg.conditioning,
        num_classes=num_classes,
        num_bits=num_bits,
        resolution=cfg.resolution,
        inference_steps=cfg.get("inference_steps", 300),
        self_cond=use_self_cond,
        cfg_drop_rate=cfg.cfg_drop_rate,
        guidance_scale=cfg.guidance_scale,
        time_shift_xi=cfg.get("time_shift_xi", 0.0),
        n_sup_min=cfg.get("n_sup_min", 2),
        n_sup_max=cfg.get("n_sup_max", 6),
        q_loss_weight=cfg.get("q_loss_weight", 0.15),
        lr=cfg.lr,
        betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.get("warmup_steps", 500),
        num_epochs=cfg.num_epochs,
        eval_every_n_epochs=cfg.eval_every_n_epochs,
        num_eval_samples=cfg.num_eval_samples,
        k_values=k_values if cfg.conditioning == "k_steps" else None,
    )

    data_module = EmbeddedSokobanDataModule(
        data_path=cfg.dataset.data_path,
        val_data_path=cfg.dataset.get("val_data_path", None),
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=cfg.dataset.get("k_values", [3, 8, 10]),
        use_dihedral_aug=cfg.dataset.get("use_dihedral_aug", True),
        bot_removal_prob=cfg.dataset.get("bot_removal_prob", 0.75),
    )

    run_name = cfg.get("run_name", None)
    wandb_logger = WandbLogger(
        project="Sokoban-EmbeddedTRM-BitDiffusion",
        name=run_name,
        save_dir=cfg.output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = [
        EmbeddedEMACallback(decay=cfg.get("ema_decay", 0.9999), inv_gamma=1.0, power=0.75),
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="best-{epoch}-{step}", monitor="val/loss", mode="min", save_top_k=1, verbose=True,
        ),
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="periodic-{epoch}", every_n_epochs=cfg.save_every_n_epochs, save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.get("precision", "16-mixed"),
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    trainer.fit(lit_model, datamodule=data_module, ckpt_path=cfg.get("resume_from_checkpoint", None))


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
