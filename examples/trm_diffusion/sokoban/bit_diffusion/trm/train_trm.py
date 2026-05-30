"""
TRM Bit-Diffusion training for Sokoban board generation.

Implements the full TRM algorithm:
  - n_sup(t) schedule: more reasoning steps at medium noise, fewer at extremes
  - Loss computed + backward + optimizer.step() at EACH n_sup step (per-sample budget mask)
  - Q_head trained on "100% correct reconstruction" — halting only at inference
  - EMA: exponential moving average of model weights for stable evaluation
  - Carry recycling: reuse (y, z) from self-conditioning pass

Architecture:
  - n latent recursions per step (z-loop + y-update)
  - T deep recursion iterations (T-1 no-grad + 1 with-grad)
  - n_sup(t) supervised reasoning steps with per-sample budget

ACT at inference:
  - n_sup(t) schedule provides max steps per diffusion timestep
  - Q_head enables early stopping with time-dependent threshold(t) = 0.95 - 0.4*t
"""

import math
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from diffusers import Transformer2DModel
from diffusers.training_utils import EMAModel


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.sokoban_dataset import SokobanBitsDataset
from dataset.evaluate_sokoban_boards import generate_metrics

# Reuse data module and gamma schedule from standard training
from train import SokobanBitDataModule, gamma_schedule


# ---------------------------------------------------------------------------
# n_sup(t) schedule and inference threshold
# ---------------------------------------------------------------------------

def n_sup_schedule(t: torch.Tensor, n_min: int, n_max: int) -> torch.Tensor:
    """Compute per-sample n_sup budget based on noise level t.

    n_sup(t) = n_min + (n_max - n_min) * sin(pi * t)

    - t≈0 (clean): n_min steps (easy reconstruction)
    - t≈0.5 (medium noise): n_max steps (hardest regime, most reasoning needed)
    - t≈1 (pure noise): n_min steps (can't extract much signal anyway)

    Args:
        t: [B] noise levels in [0, 1]
        n_min: minimum supervised steps
        n_max: maximum supervised steps

    Returns:
        [B] integer tensor of per-sample step budgets
    """
    return (n_min + (n_max - n_min) * torch.sin(math.pi * t)).round().long().clamp(min=n_min, max=n_max)


def halt_threshold(t_val: float) -> float:
    """Time-dependent Q_head threshold for inference halting.

    threshold(t) = 0.95 - 0.4*t

    - t≈0 (final denoising): 0.95 — strict, need high confidence to stop early
    - t≈0.5 (middle): 0.75 — moderate
    - t≈1 (early denoising): 0.55 — lenient, rough estimate suffices
    """
    return 0.95 - 0.4 * t_val


# ---------------------------------------------------------------------------
# TRM Module with Q_head
# ---------------------------------------------------------------------------

class TRMDiT(nn.Module):
    """TRM recursive wrapper around Transformer2DModel (ada_norm_zero).

    Implements the TRM algorithm:
      - Inner (n iterations): z = norm_z(blocks(x_high + y + z))
      - Outer (1 y-update):   y = norm_y(blocks(y + z))
      - Deep (T repetitions): T-1 no-grad + 1 with-grad
      - Returns: (y_for_output, y.detach(), z.detach()) + Q_head(y)

    Q_head: predicts whether current output is 100% correct (for inference halting).
    Trained with BCE loss during training, used for early stopping only at inference.
    """

    def __init__(self, core_model: Transformer2DModel, resolution: int, n: int = 6, T: int = 3, n_sup_max: int = 12, use_grid_pos_embed: bool = True):
        super().__init__()
        self.core_model = core_model
        self.n = n
        self.T = T
        self.n_sup_max = n_sup_max
        self.use_grid_pos_embed = use_grid_pos_embed

        dim = core_model.config.num_attention_heads * core_model.config.attention_head_dim
        patch_size = getattr(core_model.config, "patch_size", 1)
        self.h_p = resolution // patch_size
        self.w_p = resolution // patch_size

        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)

        # Q_head: predicts confidence that current output is 100% correct
        # Initialized to output -5 → sigmoid(-5) ≈ 0.007 (start with "never halt")
        self.q_head = nn.Linear(dim, 1)
        with torch.no_grad():
            nn.init.zeros_(self.q_head.weight)
            self.q_head.bias.fill_(-5.0)

        # Learnable grid positional embedding (additive on top of sincos from PatchEmbed)
        # Useful for structured-grid tasks (Sokoban, Sudoku); disable for natural images
        seq_len = self.h_p * self.w_p
        if self.use_grid_pos_embed:
            self.grid_pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
            nn.init.trunc_normal_(self.grid_pos_embed, std=0.02)
        else:
            self.grid_pos_embed = None

        # Initial states (fixed random, not trained)
        self.register_buffer("y_init", torch.randn(1, seq_len, dim))
        self.register_buffer("z_init", torch.randn(1, seq_len, dim))

    def _get_initial_states(self, batch_size: int):
        y = self.y_init.expand(batch_size, -1, -1).clone()
        z = self.z_init.expand(batch_size, -1, -1).clone()
        return y, z

    def _run_blocks(self, hidden_states: torch.Tensor, timestep, class_labels):
        """Run all transformer blocks on hidden_states."""
        for block in self.core_model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=timestep,
                cross_attention_kwargs=None,
                class_labels=class_labels,
            )
        return hidden_states

    def _latent_recursion(self, x_high, y, z, timestep, class_labels):
        """One latent recursion: n z-iterations + 1 y-update."""
        for _ in range(self.n):
            z = self.norm_z(self._run_blocks(x_high + y + z, timestep, class_labels)).float()

        y = self.norm_y(self._run_blocks(y + z, timestep, class_labels)).float()
        return y, z

    def _deep_recursion(self, x_high, y, z, timestep, class_labels):
        """T latent recursions: T-1 no-grad + 1 with gradients.

        Returns:
            y_final: non-detached y for output_head and Q_head (gradients flow)
            y_detached: detached y for next n_sup step
            z_detached: detached z for next n_sup step
        """
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x_high, y, z, timestep, class_labels)

        y_final, z_final = self._latent_recursion(x_high, y, z, timestep, class_labels)
        return y_final, y_final.detach(), z_final.detach()

    # Scale continuous t∈[0,1] to give sinusoidal embeddings more frequency variation
    TIMESTEP_SCALE = 1000.0

    def _patchify(self, sample, timestep):
        """Convert image input to patch sequence via core_model internals.

        Scales timestep by TIMESTEP_SCALE for richer sinusoidal embeddings.
        Adds learnable grid positional embedding on top of fixed sincos.
        """
        scaled_t = timestep * self.TIMESTEP_SCALE
        hidden_states, encoder_hidden_states, timestep_out, embedded_timestep = (
            self.core_model._operate_on_patched_inputs(sample, None, scaled_t, None)
        )
        if self.grid_pos_embed is not None:
            hidden_states = hidden_states + self.grid_pos_embed
        return hidden_states.float(), timestep_out, embedded_timestep

    def _unpatchify(self, hidden_states, timestep, class_labels, embedded_timestep):
        """Convert patch sequence back to image via core_model internals."""
        return self.core_model._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep,
            height=self.h_p,
            width=self.w_p,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor = None,
        carry: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_steps: int | None = None,
    ):
        """Full TRM forward pass (runs n_steps supervised iterations, no halting).

        Args:
            carry: Optional (y, z) tuple from previous diffusion step.
            n_steps: Number of supervised steps to run. Defaults to n_sup_max.

        Returns:
            output: object with .sample attribute
            new_carry: (y_detached, z_detached) for next diffusion step
        """
        if n_steps is None:
            n_steps = self.n_sup_max
        bsz = sample.shape[0]
        x_high, ts, embedded_ts = self._patchify(sample, timestep)

        if carry is not None:
            y, z = carry[0].to(x_high.device), carry[1].to(x_high.device)
        else:
            y, z = self._get_initial_states(bsz)

        for _ in range(n_steps):
            y_final, y, z = self._deep_recursion(x_high, y, z, ts, class_labels)

        output = self._unpatchify(y_final, ts, class_labels, embedded_ts)
        return _TRMOutput(sample=output), (y.detach(), z.detach())

    def forward_with_early_stop(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor = None,
        threshold: float = 0.5,
        carry: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_steps: int | None = None,
    ):
        """TRM forward with Q_head-based early stopping.

        Runs up to n_steps supervised iterations. At each step, checks Q_head:
        if sigmoid(q_logit) > threshold (batch mean), stops early.

        Args:
            carry: Optional (y, z) tuple from previous diffusion step.
            threshold: Confidence threshold for halting (sigmoid scale).
            n_steps: Max supervised steps. Defaults to n_sup_max.

        Returns:
            output: object with .sample
            n_taken: number of supervised steps executed
            new_carry: (y_detached, z_detached) for next diffusion step
        """
        if n_steps is None:
            n_steps = self.n_sup_max
        bsz = sample.shape[0]
        x_high, ts, embedded_ts = self._patchify(sample, timestep)

        if carry is not None:
            y, z = carry[0].to(x_high.device), carry[1].to(x_high.device)
        else:
            y, z = self._get_initial_states(bsz)
        n_taken = 0

        for i in range(n_steps):
            y_final, y, z = self._deep_recursion(x_high, y, z, ts, class_labels)
            n_taken = i + 1

            # Q_head early stopping (not on last step — always complete at least n_steps)
            if i < n_steps - 1:
                q_logit = self.q_head(y_final.mean(dim=1))
                if q_logit.sigmoid().mean().item() > threshold:
                    break

        output = self._unpatchify(y_final, ts, class_labels, embedded_ts)
        return _TRMOutput(sample=output), n_taken, (y.detach(), z.detach())


class _TRMOutput:
    """Simple output container matching Transformer2DModel output interface."""
    __slots__ = ("sample",)

    def __init__(self, sample):
        self.sample = sample


# ---------------------------------------------------------------------------
# Lightning Module — Manual Optimization with per-step loss + EMA + Q_head
# ---------------------------------------------------------------------------

class SokobanTRMBitDiffusion(L.LightningModule):
    """TRM Bit-Diffusion with per-step optimization, Q_head, and EMA.

    Training:
      - n_sup(t) schedule determines per-sample step budget based on noise level
      - Loss computed + backward + optimizer.step() at each active step
      - Q_head trained on "100% correct after discretization" (not used for halting)
      - Per-sample budget mask: samples that exhausted their budget don't get loss

    Inference:
      - n_sup(t) schedule provides max steps per diffusion timestep
      - Q_head enables early stopping with time-dependent threshold(t) = 0.95 - 0.4*t
    """

    def __init__(
        self,
        model: TRMDiT,
        conditioning: str = "unconditional",
        num_classes: int = 4,
        num_bits: int = 3,
        resolution: int = 12,
        diffusion_steps: int = 1000,
        self_cond: bool = True,
        cfg_drop_rate: float = 0.1,
        guidance_scale: float = 4.0,
        time_shift_xi: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 300,
        eval_every_n_epochs: int = 50,
        num_eval_samples: int = 100,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_warmup: bool = True,
        ema_inv_gamma: float = 1.0,
        ema_power: float = 0.75,
        # Q_head
        q_loss_weight: float = 0.1,
        # n_sup(t) schedule
        n_sup_min: int = 2,
        n_sup_max: int = 12,
        # Carry recycling
        use_carry_recycling: bool = False,
        carry_recycle_prob: float = 0.5,
        use_carry_persistence: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False  # Manual opt for per-step loss

        self.model = model
        self.criterion = nn.MSELoss()

        self.conditioning = conditioning
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.resolution = resolution
        self.diffusion_steps = diffusion_steps
        self.self_cond = self_cond
        self.cfg_drop_rate = cfg_drop_rate
        self.guidance_scale = guidance_scale
        self.time_shift_xi = time_shift_xi
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_eval_samples = num_eval_samples

        # Q_head config
        self.q_loss_weight = q_loss_weight

        # n_sup(t) schedule config
        self.n_sup_min = n_sup_min
        self.n_sup_max = n_sup_max

        # EMA config (initialized in on_fit_start when params are on device)
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_warmup = ema_warmup
        self.ema_inv_gamma = ema_inv_gamma
        self.ema_power = ema_power
        self.ema: EMAModel | None = None

        # Carry recycling config
        self.use_carry_recycling = use_carry_recycling
        self.carry_recycle_prob = carry_recycle_prob
        self.use_carry_persistence = use_carry_persistence

    # -----------------------------------------------------------------------
    # EMA checkpoint persistence
    # -----------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint and self.use_ema:
            # Defer EMA restoration — model params may not be on device yet
            self._ema_state_to_load = checkpoint["ema_state_dict"]

    def on_fit_start(self):
        """Initialize EMA after model is on device."""
        if self.use_ema:
            self.ema = EMAModel(
                self.model.parameters(),
                decay=self.ema_decay,
                use_ema_warmup=self.ema_warmup,
                inv_gamma=self.ema_inv_gamma,
                power=self.ema_power,
            )
            self.ema.to(self.device)

            # Restore EMA state if resuming from checkpoint
            if hasattr(self, "_ema_state_to_load"):
                self.ema.load_state_dict(self._ema_state_to_load)
                del self._ema_state_to_load

    def apply_ema_weights(self):
        """Copy EMA weights into model parameters (for inference after loading)."""
        if self.ema is not None:
            self.ema.copy_to(self.model.parameters())

    # -----------------------------------------------------------------------
    # Input preparation helpers
    # -----------------------------------------------------------------------
    def _apply_cfg_dropout(self, class_labels: torch.Tensor, cond_board: torch.Tensor | None = None):
        if self.cfg_drop_rate <= 0 or class_labels is None:
            return class_labels, cond_board

        drop_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.cfg_drop_rate
        class_labels = torch.where(drop_mask, self.num_classes, class_labels)

        if cond_board is not None:
            cond_board = torch.where(drop_mask.view(-1, 1, 1, 1), torch.zeros_like(cond_board), cond_board)

        return class_labels, cond_board

    def _build_model_input(self, x_t: torch.Tensor, x_pred: torch.Tensor, cond_board: torch.Tensor | None = None):
        parts = [x_t, x_pred]
        if cond_board is not None:
            parts.append(cond_board)
        return torch.cat(parts, dim=1)

    def _extract_class_labels(self, batch):
        if self.conditioning == "num_boxes":
            return (batch["num_boxes"] - 1).long()
        elif self.conditioning == "k_steps":
            return (batch["k"] - 1).long()
        return torch.full((batch["target"].shape[0],), self.num_classes, dtype=torch.long)

    # -----------------------------------------------------------------------
    # Training — n_sup(t) schedule, per-sample budget mask, Q_head train-only
    # -----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x_bits = batch["target"]  # (B, num_bits, H, W) in {-1, 1}
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)
        B = x_bits.shape[0]

        # --- 1. Sample noise and build noisy input (once per batch) ---
        t = torch.rand(B, device=self.device)
        t_view = t.view(-1, 1, 1, 1)
        gamma_t = gamma_schedule(t_view)
        eps = torch.randn_like(x_bits)
        x_t = torch.sqrt(gamma_t) * x_bits + torch.sqrt(1.0 - gamma_t) * eps

        class_labels_train, cond_board_train = self._apply_cfg_dropout(class_labels, cond_board)

        # --- 2. Compute per-sample n_sup budget from schedule ---
        # n_sup(t) = n_min + (n_max - n_min) * sin(pi * t)
        budget = n_sup_schedule(t, self.n_sup_min, self.n_sup_max)  # [B] int
        max_steps = int(budget.max().item())

        # --- 3. Self-conditioning + carry recycling (single no-grad forward) ---
        x_pred = torch.zeros_like(x_t)
        carry_init = None

        use_self_cond = self.self_cond and torch.rand(1).item() > 0.5
        if use_self_cond:
            with torch.no_grad():
                model_input = self._build_model_input(x_t, x_pred, cond_board_train)
                out, (y_sc, z_sc) = self.model(
                    sample=model_input, timestep=t, class_labels=class_labels_train,
                    n_steps=max_steps,
                )
                x_pred = (out.sample if hasattr(out, "sample") else out).detach()
                if self.use_carry_recycling:
                    carry_init = (y_sc.detach(), z_sc.detach())
        elif self.use_carry_recycling and torch.rand(1).item() < self.carry_recycle_prob:
            with torch.no_grad():
                model_input = self._build_model_input(x_t, x_pred, cond_board_train)
                _, (y_sc, z_sc) = self.model(
                    sample=model_input, timestep=t, class_labels=class_labels_train,
                    n_steps=max_steps,
                )
                carry_init = (y_sc.detach(), z_sc.detach())

        # --- 4. Build model input ---
        model_input = self._build_model_input(x_t, x_pred, cond_board_train)

        # --- 5. Initialize carry (y, z) ---
        if carry_init is not None:
            y, z = carry_init
        else:
            y, z = self.model._get_initial_states(B)

        # --- 6. Supervised loop — only active samples computed each step ---
        # Loop for max_steps; only process samples still within their budget.
        # Q_head is trained at every active step but NOT used for halting.
        total_diff_loss = 0.0
        total_q_loss = 0.0
        total_active = 0

        for sup_step in range(max_steps):
            # Which samples still have budget left?
            active_mask = sup_step < budget  # [B] bool
            if not active_mask.any():
                break
            active_idx = active_mask.nonzero(as_tuple=True)[0]

            # Slice only active samples — skip inactive compute entirely
            model_input_a = model_input[active_idx]
            t_a = t[active_idx]
            cls_a = class_labels_train[active_idx]
            x_bits_a = x_bits[active_idx]
            y_a = y[active_idx]
            z_a = z[active_idx]

            # Patchify + deep recursion (only active samples)
            x_high_a, ts_a, embedded_ts_a = self.model._patchify(model_input_a, t_a)
            y_final_a, y_new_a, z_new_a = self.model._deep_recursion(
                x_high_a, y_a, z_a, ts_a, cls_a
            )

            # Update carry for active samples (y, z are detached — safe to index-assign)
            y[active_idx] = y_new_a
            z[active_idx] = z_new_a

            # Output head: unpatchify to image space
            x_0_pred_a = self.model._unpatchify(y_final_a, ts_a, cls_a, embedded_ts_a)

            # Q_head: confidence prediction (trained, not used for halting)
            q_logit_a = self.model.q_head(y_final_a.mean(dim=1))  # (n_active, 1)

            # --- Diffusion loss (all samples in this slice are active) ---
            diff_loss = F.mse_loss(x_0_pred_a, x_bits_a)

            # --- Q_head loss: target = 100% correct reconstruction ---
            with torch.no_grad():
                bit_accuracy = (x_0_pred_a.sign() == x_bits_a).float().mean(dim=(1, 2, 3))
                is_correct = (bit_accuracy >= 1.0).float()
            q_loss = F.binary_cross_entropy_with_logits(
                q_logit_a.squeeze(-1), is_correct
            )

            # --- Combined loss ---
            loss = diff_loss + self.q_loss_weight * q_loss

            # --- Backward + step ---
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0)
            opt.step()
            opt.zero_grad()

            # --- EMA update ---
            if self.ema is not None:
                self.ema.step(self.model.parameters())

            total_diff_loss += diff_loss.detach()
            total_q_loss += q_loss.detach()
            total_active += 1

        # --- Logging ---
        n_steps_done = max(total_active, 1)
        self.log("train/loss", total_diff_loss / n_steps_done, prog_bar=True, sync_dist=True)
        self.log("train/q_loss", total_q_loss / n_steps_done, sync_dist=True)
        self.log("train/max_steps", float(max_steps), sync_dist=True)
        self.log("train/avg_budget", budget.float().mean().item(), sync_dist=True)
        if self.ema is not None:
            self.log("train/ema_decay", self.ema.cur_decay_value or 0.0)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x_bits = batch["target"]
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)
        B = x_bits.shape[0]

        t = torch.rand(B, device=self.device)
        t_view = t.view(-1, 1, 1, 1)
        gamma_t = gamma_schedule(t_view)
        eps = torch.randn_like(x_bits)
        x_t = torch.sqrt(gamma_t) * x_bits + torch.sqrt(1.0 - gamma_t) * eps

        # No CFG dropout during validation
        x_pred = torch.zeros_like(x_t)
        model_input = self._build_model_input(x_t, x_pred, cond_board)

        # Use n_sup schedule for validation too
        budget = n_sup_schedule(t, self.n_sup_min, self.n_sup_max)
        max_steps = int(budget.max().item())

        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        out, _ = self.model(sample=model_input, timestep=t, class_labels=class_labels, n_steps=max_steps)
        x_pred_final = out.sample if hasattr(out, "sample") else out
        loss = self.criterion(x_pred_final, x_bits)

        if self.ema is not None:
            self.ema.restore(self.model.parameters())

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        """Step LR scheduler once per epoch (manual optimization)."""
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return
        if not self.trainer.is_global_zero:
            return
        self._run_sokoban_evaluation()

    # -----------------------------------------------------------------------
    # Sampling with n_sup(t) schedule + Q_head early stopping
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: torch.Tensor | None = None,
        cond_board: torch.Tensor | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Generate boards with n_sup(t) schedule and Q_head early stopping.

        At each diffusion step:
          - n_sup(t) determines max reasoning steps for this noise level
          - Q_head can halt early with time-dependent threshold(t) = 0.95 - 0.4*t
          - Carry is optionally persisted across diffusion steps
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        use_cfg = guidance_scale > 1.0 and class_labels is not None and self.conditioning != "unconditional"

        x_t = torch.randn(batch_size, self.num_bits, self.resolution, self.resolution, device=device)
        x_pred = torch.zeros_like(x_t)

        # Carry state persisted across diffusion steps
        carry = None
        carry_uncond = None

        for step in range(self.diffusion_steps):
            t_now_val = 1.0 - step / self.diffusion_steps
            t_next_val = max(1.0 - (step + 1.0) / self.diffusion_steps, 0.0)
            t_now = torch.full((batch_size,), t_now_val, device=device)

            # Asymmetric time intervals (Bit Diffusion paper Sec 3.2):
            # Feed model t' = t + xi to encourage stronger denoising
            t_model = torch.full((batch_size,), min(t_now_val + self.time_shift_xi, 1.0), device=device)

            if not self.self_cond:
                x_pred = torch.zeros_like(x_t)

            # Compute schedule for this timestep
            n_steps = int(n_sup_schedule(
                torch.tensor([t_now_val], device=device), self.n_sup_min, self.n_sup_max
            ).item())
            threshold = halt_threshold(t_now_val)

            if use_cfg:
                # Batch cond + uncond in one forward pass (2*B)
                uncond_labels = torch.full_like(class_labels, self.num_classes)  # type: ignore
                uncond_cond_board = torch.zeros_like(cond_board) if cond_board is not None else None

                model_input_cond = self._build_model_input(x_t, x_pred, cond_board)
                model_input_uncond = self._build_model_input(x_t, x_pred, uncond_cond_board)

                model_input_both = torch.cat([model_input_cond, model_input_uncond], dim=0)
                t_both = torch.cat([t_model, t_model], dim=0)
                labels_both = torch.cat([class_labels, uncond_labels], dim=0)  # type: ignore

                # Merge carry states along batch dim
                if self.use_carry_persistence and carry is not None and carry_uncond is not None:
                    carry_both = (
                        torch.cat([carry[0], carry_uncond[0]], dim=0),
                        torch.cat([carry[1], carry_uncond[1]], dim=0),
                    )
                else:
                    carry_both = None

                out_both, _, new_carry_both = self.model.forward_with_early_stop(
                    sample=model_input_both, timestep=t_both, class_labels=labels_both,
                    threshold=threshold, carry=carry_both, n_steps=n_steps,
                )

                x_pred_cond, x_pred_uncond = out_both.sample.chunk(2, dim=0)

                # Split carry back for next diffusion step
                carry = (new_carry_both[0][:batch_size], new_carry_both[1][:batch_size])
                carry_uncond = (new_carry_both[0][batch_size:], new_carry_both[1][batch_size:])

                x_pred = x_pred_uncond + guidance_scale * (x_pred_cond - x_pred_uncond)
            else:
                # Single forward pass (no CFG)
                model_input = self._build_model_input(x_t, x_pred, cond_board)
                step_carry = carry if self.use_carry_persistence else None

                out, _, new_carry = self.model.forward_with_early_stop(
                    sample=model_input, timestep=t_model, class_labels=class_labels,
                    threshold=threshold, carry=step_carry, n_steps=n_steps,
                )
                x_pred = out.sample
                carry = new_carry

            # DDPM step uses real t (not shifted) for state transition
            x_t = self._ddpm_step(x_t, x_pred, t_now_val, t_next_val)

        return x_pred

    def _ddpm_step(self, x_t, x_pred, t_now_val, t_next_val):
        gamma_now = gamma_schedule(torch.tensor(t_now_val, device=x_t.device))
        gamma_next = gamma_schedule(torch.tensor(t_next_val, device=x_t.device))

        alpha_t = gamma_now / gamma_next if gamma_next > 0 else gamma_now
        beta_t = 1.0 - alpha_t

        if t_next_val == 0.0:
            return x_pred

        coef1 = (math.sqrt(gamma_next) * beta_t) / (1.0 - gamma_now)
        coef2 = (math.sqrt(alpha_t) * (1.0 - gamma_next)) / (1.0 - gamma_now)
        mu = coef1 * x_pred + coef2 * x_t

        var = ((1.0 - gamma_next) / (1.0 - gamma_now)) * beta_t
        sigma = math.sqrt(max(var, 1e-10))

        z = torch.randn_like(x_t)
        return mu + sigma * z

    # -----------------------------------------------------------------------
    # Evaluation (uses EMA weights)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, val_dataloader, num_samples: int | None = None):
        device = self.device
        if num_samples is None:
            num_samples = self.num_eval_samples
        batch_size = min(50, num_samples)

        # Swap to EMA weights for generation
        if self.ema is not None:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        all_gen_bits = []
        all_gen_boards = []
        all_cond_boards = []
        all_target_boards = []
        all_k_values = []
        all_num_boxes_labels = []

        val_iter = iter(val_dataloader)
        generated_count = 0

        while generated_count < num_samples:
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                batch = next(val_iter)

            bsz = min(batch_size, num_samples - generated_count)
            x_bits = batch["target"][:bsz].to(device)
            class_labels = self._extract_class_labels({k: v[:bsz] for k, v in batch.items() if isinstance(v, torch.Tensor)})
            cond_board = batch.get("condition", None)
            if cond_board is not None:
                cond_board = cond_board[:bsz].to(device)
            if class_labels is not None:
                class_labels = class_labels.to(device)

            gen_bits = self.generate_batch(
                batch_size=x_bits.shape[0], device=device,
                class_labels=class_labels, cond_board=cond_board,
            )
            all_gen_bits.append(gen_bits.cpu())

            gen_int = SokobanBitsDataset.bits_to_tokens(gen_bits).cpu().numpy()
            all_gen_boards.append(gen_int)
            target_int = SokobanBitsDataset.bits_to_tokens(x_bits).cpu().numpy()
            all_target_boards.append(target_int)

            if cond_board is not None:
                cond_int = SokobanBitsDataset.bits_to_tokens(cond_board).cpu().numpy()
                all_cond_boards.append(cond_int)
            if "k" in batch:
                all_k_values.extend(batch["k"][:bsz].tolist())
            if "num_boxes" in batch:
                all_num_boxes_labels.extend(batch["num_boxes"][:bsz].tolist())

            generated_count += x_bits.shape[0]

        # Restore training weights
        if self.ema is not None:
            self.ema.restore(self.model.parameters())

        all_gen_np = np.clip(np.concatenate(all_gen_boards, axis=0)[:num_samples], 0, 6)
        all_target_np = np.concatenate(all_target_boards, axis=0)[:num_samples]
        cond_np = np.concatenate(all_cond_boards, axis=0)[:num_samples] if all_cond_boards else None
        k_vals = all_k_values[:num_samples] if all_k_values else None
        num_boxes_labels = np.array(all_num_boxes_labels[:num_samples]) if all_num_boxes_labels else None

        metrics = generate_metrics(
            generated_boards=all_gen_np,
            num_boxes_labels=num_boxes_labels,
            conditioning_boards=cond_np,
            target_boards=all_target_np,
            k_values=k_vals,
        )

        gen_bits_tensor = torch.cat(all_gen_bits, dim=0)[:num_samples]
        return metrics, gen_bits_tensor

    @torch.no_grad()
    def _run_sokoban_evaluation(self):
        val_dl = self.trainer.datamodule.val_dataloader()
        metrics, gen_bits = self.evaluate(val_dl)

        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)

        if isinstance(self.logger, WandbLogger):
            self._log_board_renders(gen_bits[:16])

    def _log_board_renders(self, gen_bits: torch.Tensor):
        import wandb
        val_ds = self.trainer.datamodule.val_ds
        rendered = []
        for i in range(gen_bits.shape[0]):
            img = val_ds.render_bit_boards(gen_bits[i])
            rendered.append(wandb.Image(img, caption=f"epoch={self.current_epoch}"))
        self.logger.experiment.log({"sokoban/generated_boards": rendered})

    # -----------------------------------------------------------------------
    # Optimizer (manual optimization — no automatic LR scheduling)
    # -----------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Main entry point (Hydra)
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="trm_diffusion")
def main(cfg: DictConfig):
    num_bits = cfg.num_bits
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * 3
    else:
        in_channels = num_bits * 2

    # Build core Transformer2DModel
    core_model = Transformer2DModel(
        sample_size=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_layers=cfg.model.num_layers,
        patch_size=cfg.model.patch_size,
        attention_head_dim=cfg.model.attention_head_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        cross_attention_dim=None,
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        num_embeds_ada_norm=cfg.num_classes + 1,
        norm_type="ada_norm_zero",
    )

    # Wrap in TRM
    model = TRMDiT(
        core_model=core_model,
        resolution=cfg.resolution,
        n=cfg.trm.n,
        T=cfg.trm.T,
        n_sup_max=cfg.n_sup_max,
        use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
    )

    # Lightning module
    lit_model = SokobanTRMBitDiffusion(
        model=model,
        conditioning=cfg.conditioning,
        num_classes=cfg.num_classes,
        num_bits=num_bits,
        resolution=cfg.resolution,
        diffusion_steps=cfg.diffusion_steps,
        self_cond=cfg.self_cond,
        cfg_drop_rate=cfg.cfg_drop_rate,
        guidance_scale=cfg.guidance_scale,
        time_shift_xi=cfg.get("time_shift_xi", 0.0),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_epochs=cfg.num_epochs,
        eval_every_n_epochs=cfg.eval_every_n_epochs,
        num_eval_samples=cfg.num_eval_samples,
        # EMA
        use_ema=cfg.get("use_ema", True),
        ema_decay=cfg.get("ema_decay", 0.9999),
        ema_warmup=cfg.get("ema_warmup", True),
        ema_inv_gamma=cfg.get("ema_inv_gamma", 1.0),
        ema_power=cfg.get("ema_power", 0.75),
        # Q_head
        q_loss_weight=cfg.get("q_loss_weight", 0.1),
        # n_sup(t) schedule
        n_sup_min=cfg.get("n_sup_min", 2),
        n_sup_max=cfg.get("n_sup_max", 12),
        # Carry recycling
        use_carry_recycling=cfg.get("use_carry_recycling", False),
        carry_recycle_prob=cfg.get("carry_recycle_prob", 0.5),
        use_carry_persistence=cfg.get("use_carry_persistence", False),
    )

    # Data
    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.data_path,
        val_data_path=cfg.dataset.get("val_data_path", None),
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=cfg.dataset.get("k_values", [1, 3, 5, 8, 10]),
        use_dihedral_aug=cfg.dataset.get("use_dihedral_aug", True),
    )

    # Logger
    wandb_logger = WandbLogger(
        project="Sokoban-TRM-BitDiffusion",
        name=cfg.get("run_name", None),
        save_dir=cfg.output_dir,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="best-{epoch}-{step}",
            monitor="sokoban/solvability",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="periodic-{epoch}",
            every_n_epochs=cfg.save_every_n_epochs,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer — no gradient_clip_val (handled manually in training_step)
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

    ckpt_path = cfg.get("resume_from_checkpoint", None)
    trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
