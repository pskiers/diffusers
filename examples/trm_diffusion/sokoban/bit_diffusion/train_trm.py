"""
TRM Bit-Diffusion training for Sokoban board generation.

Implements the full TRM algorithm:
  - n_sup(t) schedule: more reasoning steps at medium noise, fewer at extremes
  - Loss computed + backward + optimizer.step() at EACH n_sup step
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from typing import Optional, Tuple

from diffusers import Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sokoban.bit_diffusion.train_std import SokobanBitDataModule, EMACallback, SokobanBitDiffusion


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


class TRMDiT(nn.Module):
    """TRM recursive wrapper around Transformer2DModel (ada_norm_zero).

    Implements the TRM algorithm:
      - Inner (n iterations): z = norm_z(blocks(x_high + y + z))
      - Outer (1 y-update):   y = norm_y(blocks(y + z))
      - Deep (T repetitions): T-1 no-grad + 1 with-grad
      - Returns: (y_for_output, y.detach(), z.detach()) + Q_head(y)

    Q_head:
    - predicts whether current output is 100% correct (for inference halting).
    - Trained with BCE loss during training, used for early stopping only at inference.
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

        # Learnable grid positional embedding
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

    def _patchify(self, sample, timestep):
        """Convert image input to patch sequence via core_model internals.
        Adds learnable grid positional embedding on top of fixed sincos.
        """
        hidden_states, encoder_hidden_states, timestep_out, embedded_timestep = (
            self.core_model._operate_on_patched_inputs(sample, None, timestep, None)
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
        class_labels: Optional[torch.Tensor] = None,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_steps: Optional[int] = None,
    ):
        """Full TRM forward pass (runs n_steps supervised iterations, no halting)."""
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
        class_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_steps: Optional[int] = None,
    ):
        """TRM forward with Q_head-based early stopping."""
        if n_steps is None:
            n_steps = self.n_sup_max
        bsz = sample.shape[0]
        x_high, ts, embedded_ts = self._patchify(sample, timestep)

        if carry is not None:
            y, z = carry[0].to(x_high.device), carry[1].to(x_high.device)
        else:
            y, z = self._get_initial_states(bsz)

        # Per-sample active mask + storage for final y
        active_mask = torch.ones(bsz, dtype=torch.bool, device=x_high.device)
        y_final_store = y.clone()
        n_taken = torch.zeros(bsz, dtype=torch.long, device=x_high.device)

        for i in range(n_steps):
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                break

            # Run deep recursion only for active samples
            y_final_a, y_new_a, z_new_a = self._deep_recursion(
                x_high[active_idx], y[active_idx], z[active_idx], ts[active_idx],
                None if class_labels is None else class_labels[active_idx],
            )

            # store updated states
            y[active_idx] = y_new_a
            z[active_idx] = z_new_a
            y_final_store[active_idx] = y_final_a
            n_taken[active_idx] = i + 1

            # Q_head per-sample stopping (do not check on last iteration)
            if i < n_steps - 1:
                q_logits = self.q_head(y_final_a.mean(dim=1)).squeeze(-1)
                q_probs = q_logits.sigmoid()
                to_halt = q_probs > threshold
                if to_halt.any():
                    # Map halted indices back into batch
                    halted_idx = active_idx[to_halt.nonzero(as_tuple=True)[0]]
                    active_mask[halted_idx] = False

        # Unpatchify using stored y_final for full batch
        output = self._unpatchify(y_final_store, ts, class_labels, embedded_ts)
        return _TRMOutput(sample=output), n_taken, (y.detach(), z.detach())


class _TRMOutput:
    """Simple output container matching Transformer2DModel output interface."""
    __slots__ = ("sample",)

    def __init__(self, sample):
        self.sample = sample


class TRMEMACallback(EMACallback):
    """
    Rozszerzenie EMACallback dla TRM.
    Wyłącza automatyczny krok na końcu batcha. Krok EMA jest wymuszany ręcznie
    wewnętrz pętli n_sup, synchronicznie z opt.step().
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass


class SokobanTRMBitDiffusion(SokobanBitDiffusion):
    """TRM Bit-Diffusion with per-step optimization, Q_head, and EMA."""
    def __init__(
        self,
        # Q_head
        q_loss_weight: float = 0.1,
        # n_sup(t) schedule
        n_sup_min: int = 2,
        n_sup_max: int = 12,
        # Carry recycling
        use_carry_recycling: bool = False,
        carry_recycle_prob: float = 0.5,
        use_carry_persistence: bool = False,
        **base_kwargs,  # the same like in SokobanBitDiffusion
    ) -> None:
        super().__init__(**base_kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False # manual step after every n_sup

        self.q_loss_weight = q_loss_weight

        self.n_sup_min = n_sup_min
        self.n_sup_max = n_sup_max

        self.use_carry_recycling = use_carry_recycling
        self.carry_recycle_prob = carry_recycle_prob
        self.use_carry_persistence = use_carry_persistence

    def _trm_supervised_step(self, model_input, timesteps, class_labels_train, x_bits, noise, y, z, n_sup_steps, opt):
        total_diff_loss = 0.0
        total_q_loss = 0.0
        total_active = 0
        # BEZPIECZEŃSTWO: ensure gradients and optimizer state are clean before TRM loop
        opt.zero_grad()

        max_steps = int(n_sup_steps.max().item())
        for sup_step in range(max_steps):
            active_mask = sup_step < n_sup_steps
            # `.any()` returns a tensor scalar — use `.item()` for a Python bool
            if not active_mask.any().item():
                break

            active_idx = active_mask.nonzero(as_tuple=True)[0]
            model_input_a = model_input[active_idx]
            t_a = timesteps[active_idx]
            cls_a = class_labels_train[active_idx]
            x_bits_a = x_bits[active_idx]
            noise_a = noise[active_idx]
            y_a = y[active_idx]
            z_a = z[active_idx]

            x_high_a, ts_a, embedded_ts_a = self.model._patchify(model_input_a, t_a)
            y_final_a, y_new_a, z_new_a = self.model._deep_recursion(x_high_a, y_a, z_a, ts_a, cls_a)

            y[active_idx] = y_new_a
            z[active_idx] = z_new_a

            x_0_prediction_a = self.model._unpatchify(y_final_a, ts_a, cls_a, embedded_ts_a)

            # Strata Dyfuzji
            prediction_type = self.noise_scheduler.config.prediction_type
            if prediction_type == "sample":
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
                alpha_t = self._extract_into_tensor(alphas_cumprod, t_a, x_bits_a.shape)
                snr = alpha_t / (1.0 - alpha_t).clamp(min=1e-5)
                snr_weights = snr.clamp(max=5.0)
                loss_diff = snr_weights * F.mse_loss(x_0_prediction_a.float(), x_bits_a.float(), reduction="none")
                diffusion_loss = loss_diff.mean()
            elif prediction_type == "epsilon":
                diffusion_loss = F.mse_loss(x_0_prediction_a.float(), noise_a.float())
            else:
                raise ValueError(f"Unsupported prediction_type: {prediction_type}")

            # Q_head: predict the board-level accuracy of the current x0 estimate.
            # NOTE: with this bit encoding "all bits exactly correct" == "board exactly
            # correct", which is unreachable at medium/high noise -> the binary label is
            # ~always 0, the head collapses to q->0, and inference ACT never halts.
            # A graded (soft) target gives a dense, reachable signal at every noise level,
            # so sigmoid(q) becomes a calibrated confidence that drives the halting decision.
            q_logit_a = self.model.q_head(y_final_a.mean(dim=1)).squeeze(-1)
            with torch.no_grad():
                # a board cell is correct iff all of its bits have the right sign
                cell_correct = (x_0_prediction_a.sign() == x_bits_a).all(dim=1)  # (A, H, W)
                board_accuracy = cell_correct.float().mean(dim=(1, 2))           # (A,) in [0, 1]

            # Soft-label BCE: regress confidence onto board accuracy (noise difficulty is
            # already encoded in the target, so no explicit time gate is needed).
            q_loss = F.binary_cross_entropy_with_logits(q_logit_a, board_accuracy)

            # Złożenie straty i manualny krok optymalizatora na etapie TRM
            loss = diffusion_loss + self.q_loss_weight * q_loss

            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0)
            opt.step()
            opt.zero_grad()

            # BEZPIECZNA AKTUALIZACJA EMA PO KAŻDYM OPT.STEP()
            for cb in self.trainer.callbacks:
                if isinstance(cb, TRMEMACallback) and cb.ema_model is not None:
                    cb.ema_model.step(self.model.parameters())
                    break

            total_diff_loss += diffusion_loss.detach()
            total_q_loss += q_loss.detach()
            total_active += 1

        n_steps_done = max(total_active, 1)
        return total_diff_loss / n_steps_done, total_q_loss / n_steps_done

    def training_step(self, batch, batch_idx):
        x_bits = batch["target"]
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)
        B = x_bits.shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)

        class_labels_train, cond_board_train = self._apply_cfg_dropout(class_labels, cond_board)

        # normalize integer timesteps to [0, 1] before schedule
        num_train_steps = float(self.noise_scheduler.config.num_train_timesteps - 1)
        t_norm = timesteps.float() / num_train_steps
        n_sup_steps = n_sup_schedule(t_norm, self.n_sup_min, self.n_sup_max)
        max_steps = int(n_sup_steps.max().item())

        # Ortogonalna ocena prawdopodobieństwa
        do_self_cond = self.self_cond and (torch.rand(1).item() > 0.5)
        do_recycle = self.use_carry_recycling and (torch.rand(1).item() < self.carry_recycle_prob)

        x_pred = None
        carry_init = None

        if do_self_cond or do_recycle:
            with torch.no_grad():
                model_input_init = self._build_model_input(x_t, None, cond_board_train)
                # Ograniczenie n_steps=1 zabezpiecza wydajność i inicjuje pamięć zgrubnie
                out, (y_sc, z_sc) = self.model(
                    sample=model_input_init, timestep=timesteps, class_labels=class_labels_train,
                    n_steps=1,
                )
                if do_self_cond:
                    x_pred = (out.sample if hasattr(out, "sample") else out).detach()
                    x_pred = x_pred.clamp(-1.0, 1.0)
                if do_recycle:
                    carry_init = (y_sc.detach(), z_sc.detach())

        model_input = self._build_model_input(x_t, x_pred, cond_board_train)
        y, z = carry_init if carry_init is not None else self.model._get_initial_states(B)

        opt = self.optimizers()
        train_loss, q_loss = self._trm_supervised_step(
            model_input, timesteps, class_labels_train, x_bits, noise, y, z, n_sup_steps, opt
        )

        # Manualny krok harmonogramu uczenia
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train/loss", train_loss, prog_bar=True, sync_dist=True)
        self.log("train/q_loss", q_loss, sync_dist=True)
        self.log("train/max_steps", float(max_steps), sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x_bits = batch["target"]
        class_labels = self._extract_class_labels(batch).to(self.device)
        cond_board = batch.get("condition", None)
        B = x_bits.shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)

        x_pred = torch.zeros_like(x_t)
        model_input = self._build_model_input(x_t, x_pred, cond_board)

        num_train_steps = float(self.noise_scheduler.config.num_train_timesteps - 1)
        t_norm = timesteps.float() / num_train_steps
        budget = n_sup_schedule(t_norm, self.n_sup_min, self.n_sup_max)
        max_steps = int(budget.max().item())

        out, _ = self.model(sample=model_input, timestep=timesteps, class_labels=class_labels, n_steps=max_steps)
        x_pred_final = out.sample if hasattr(out, "sample") else out

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "sample":
            loss = F.mse_loss(x_pred_final.float(), x_bits.float())
        elif prediction_type == "epsilon":
            loss = F.mse_loss(x_pred_final.float(), noise.float())
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

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
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        use_cfg = guidance_scale > 1.0 and class_labels is not None and self.conditioning != "unconditional"

        x_t = torch.randn(batch_size, self.num_bits, self.resolution, self.resolution, device=device)
        x_pred = None
        x_pred_step = x_t

        carry = None
        carry_uncond = None

        self.noise_scheduler.set_timesteps(self.inference_steps)
        is_sample_pred = self.noise_scheduler.config.prediction_type == "sample"

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device)

            if self.time_shift_xi > 0.0:
                num_train_steps = self.noise_scheduler.config.num_train_timesteps
                t_model = torch.clamp(t_batch + int(self.time_shift_xi * num_train_steps), max=num_train_steps - 1)
            else:
                t_model = t_batch

            # Bezpieczne pobieranie skalarów
            num_train_steps = float(self.noise_scheduler.config.num_train_timesteps - 1)
            t_scalar_norm = max(0.0, min(1.0, t.item() / num_train_steps))
            t_tensor_norm = torch.tensor([t_scalar_norm], device=device)
            n_steps = int(n_sup_schedule(t_tensor_norm, self.n_sup_min, self.n_sup_max).item())

            t_model_norm = t_model[0].item() / num_train_steps
            threshold = halt_threshold(t_model_norm)

            if use_cfg:
                uncond_labels = torch.full_like(class_labels, self.num_classes)  # type: ignore
                uncond_cond_board = torch.zeros_like(cond_board) if cond_board is not None else None

                model_input_cond = self._build_model_input(x_t, x_pred, cond_board)
                model_input_uncond = self._build_model_input(x_t, x_pred, uncond_cond_board)

                model_input_both = torch.cat([model_input_cond, model_input_uncond], dim=0)
                t_both = torch.cat([t_model, t_model], dim=0)
                labels_both = torch.cat([class_labels, uncond_labels], dim=0)  # type: ignore

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
                carry = (new_carry_both[0][:batch_size], new_carry_both[1][:batch_size])
                carry_uncond = (new_carry_both[0][batch_size:], new_carry_both[1][batch_size:])

                x_pred_step = x_pred_uncond + guidance_scale * (x_pred_cond - x_pred_uncond)
                if is_sample_pred:
                    x_pred_step = x_pred_step.clamp(-1.0, 1.0)
            else:
                model_input = self._build_model_input(x_t, x_pred, cond_board)
                step_carry = carry if self.use_carry_persistence else None

                out, _, new_carry = self.model.forward_with_early_stop(
                    sample=model_input, timestep=t_model, class_labels=class_labels,
                    threshold=threshold, carry=step_carry, n_steps=n_steps,
                )
                x_pred_step = out.sample
                carry = new_carry

                if is_sample_pred:
                    x_pred_step = x_pred_step.clamp(-1.0, 1.0)

            x_t = self.noise_scheduler.step(x_pred_step, t, x_t).prev_sample
            x_pred = x_pred_step if self.self_cond else None

        return x_pred_step

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            betas=self.betas, weight_decay=self.weight_decay
        )
        warmup_steps = self.hparams.get("warmup_steps", 500)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps, # type: ignore[union-attr]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(version_base=None, config_path="config", config_name="trm_diffusion")
def main(cfg: DictConfig):
    L.seed_everything(cfg.get("seed", 42), workers=True)

    num_bits = cfg.num_bits
    # For k_steps the class label indexes the k-value list, so the embedding table must
    # be sized len(k_values) (+1 for the CFG/unconditional token), not cfg.num_classes.
    k_values = cfg.dataset.get("k_values", [3, 8, 10])
    num_classes = len(k_values) if cfg.conditioning == "k_steps" else cfg.num_classes
    use_self_cond = cfg.get("self_cond", True)
    self_cond_mult = 2 if use_self_cond else 1  # noisy + [self_cond]
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * self_cond_mult + num_bits  # + spatial condition board
    else:
        in_channels = num_bits * self_cond_mult

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
        num_embeds_ada_norm=num_classes + 1,
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
    # Build noise scheduler (same defaults as standard training)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    # Lightning module
    lit_model = SokobanTRMBitDiffusion(
        model=model,
        noise_scheduler=noise_scheduler,
        conditioning=cfg.conditioning,
        num_classes=num_classes,
        num_bits=num_bits,
        resolution=cfg.resolution,
        inference_steps=cfg.get("inference_steps", cfg.get("diffusion_steps", 400)),
        self_cond=cfg.self_cond,
        cfg_drop_rate=cfg.cfg_drop_rate,
        guidance_scale=cfg.guidance_scale,
        time_shift_xi=cfg.get("time_shift_xi", 0.0),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_epochs=cfg.num_epochs,
        eval_every_n_epochs=cfg.eval_every_n_epochs,
        num_eval_samples=cfg.num_eval_samples,
        k_values=k_values if cfg.conditioning == "k_steps" else None,
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

    # Parametr EMA z configu
    ema_decay = cfg.get("ema_decay", 0.9999)

    # Callbacks z podpiętą klasą TRMEMACallback
    callbacks = [
        TRMEMACallback(decay=ema_decay, inv_gamma=1.0, power=0.75),
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

    # Trainer
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
