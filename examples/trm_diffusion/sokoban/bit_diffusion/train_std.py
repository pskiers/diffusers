"""
Bit-Diffusion training for Sokoban board generation

Supports three conditioning modes via AdaLN (ada_norm_zero):
  1. Unconditional: no labels, pure generation
  2. num_boxes: class label = number of boxes (1-4), CFG dropout
  3. k_steps: class label = k-step index + spatial board conditioning (concat), CFG dropout

Uses cosine noise schedule from the Bit-Diffusion paper with self-conditioning.
Logs Sokoban-specific metrics (validity, solvability, diversity) via evaluate_sokoban_boards.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler, Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from sokoban.dataset.evaluate_sokoban_boards import generate_metrics
from sokoban.dataset.sokoban_dataset import SokobanBitsDataset


class SokobanBitDiffusion(L.LightningModule):
    """Bit-Diffusion with AdaLN DiT backbone for Sokoban board generation.

    Uses DDPMScheduler with squaredcos_cap_v2 beta schedule and x₀ prediction.
    SNR-weighted MSE loss.

    Conditioning modes:
      - "unconditional": no class labels
      - "num_boxes": class_labels = num_boxes (1..max_boxes), dropout token = num_classes
      - "k_steps": class_labels = k_index, spatial condition board concatenated to input
    """
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        conditioning: str = "unconditional",  # "unconditional", "num_boxes", "k_steps"
        num_classes: int = 4,
        num_bits: int = 3,
        resolution: int = 12,
        inference_steps: int = 400,
        self_cond: bool = False,
        cfg_drop_rate: float = 0.1,
        guidance_scale: float = 6.0,
        time_shift_xi: float = 0.0,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.999),
        weight_decay: float = 1e-6,
        warmup_steps: int = 500,
        num_epochs: int = 300,
        eval_every_n_epochs: int = 50,
        num_eval_samples: int = 100,
        k_values: Optional[List[int]] = None,
        num_parameters: Optional[int] = None,
        num_trainable_parameters: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_scheduler", "num_parameters", "num_trainable_parameters"])

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.inference_steps = inference_steps

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.hparams["num_parameters"] = num_params
        self.hparams["num_trainable_parameters"] = num_trainable_params
        print(f"Model parameters: {num_params:,} (trainable: {num_trainable_params:,})")

        self.conditioning = conditioning
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.resolution = resolution
        self.self_cond = self_cond
        self.cfg_drop_rate = cfg_drop_rate
        self.guidance_scale = guidance_scale
        self.time_shift_xi = time_shift_xi
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_eval_samples = num_eval_samples

        # k_steps: build lookup from k-value → class index
        if conditioning == "k_steps" and k_values:
            k_to_class = torch.zeros(max(k_values) + 1, dtype=torch.long)
            for idx, k in enumerate(k_values):
                k_to_class[k] = idx
            self.register_buffer("k_to_class", k_to_class)
        else:
            self.k_to_class = None

    # Forward helpers
    def _model_forward(self, x_input: torch.Tensor, t: torch.Tensor, class_labels=None):
        out = self.model(hidden_states=x_input, timestep=t, class_labels=class_labels)
        return out.sample if hasattr(out, "sample") else out

    def _apply_cfg_dropout(self, class_labels: torch.Tensor, cond_board: Optional[torch.Tensor] = None):
        """Randomly drop labels (and zero-out spatial condition) for CFG training."""
        if self.cfg_drop_rate <= 0 or class_labels is None:
            return class_labels, cond_board

        drop_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.cfg_drop_rate

        class_labels = torch.where(drop_mask, self.num_classes, class_labels)  # Replace dropped labels with the unconditional token

        if cond_board is not None:  # Zero-out spatial conditioning for dropped samples
            cond_board = torch.where(drop_mask.view(-1, 1, 1, 1), torch.zeros_like(cond_board), cond_board)

        return class_labels, cond_board

    def _build_model_input(self, x_t: torch.Tensor, x_pred: Optional[torch.Tensor] = None, cond_board: Optional[torch.Tensor] = None):
        parts = [x_t]
        if self.self_cond:
            parts.append(x_pred if x_pred is not None else torch.zeros_like(x_t))
        if cond_board is not None:
            parts.append(cond_board)
        return torch.cat(parts, dim=1)

    def _extract_into_tensor(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple):
        """Extract values from 1-D array for given timesteps and broadcast to target shape."""
        res = arr.gather(-1, timesteps)
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    # Training & Validation
    def _compute_loss(self, x_bits, class_labels: Optional[torch.Tensor] = None, cond_board: Optional[torch.Tensor] = None):
        """Bit-diffusion training loss with SNR weighting (x₀ prediction).
        x_bits: (B, num_bits, H, W) with values in {-1, 1}
        """
        device = x_bits.device
        B = x_bits.shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)

        class_labels_train, cond_board_train = self._apply_cfg_dropout(class_labels, cond_board)    #type: ignore

        # Self-conditioning: 50% chance to use model's own previous prediction
        x_pred = None
        if self.self_cond and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                model_input = self._build_model_input(x_t, None, cond_board_train)
                x_pred = self._model_forward(model_input, timesteps, class_labels_train).detach()
                x_pred = x_pred.clamp(-1.0, 1.0)

        model_input = self._build_model_input(x_t, x_pred, cond_board_train)
        model_output = self._model_forward(model_input, timesteps, class_labels_train)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "sample":
            # Min-SNR-γ weighted MSE loss for x₀ prediction (γ=5 prevents fp16 overflow)
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
            alpha_t = self._extract_into_tensor(alphas_cumprod, timesteps, x_bits.shape)
            snr = alpha_t / (1.0 - alpha_t).clamp(min=1e-5)
            snr_weights = snr.clamp(max=5.0)
            loss = snr_weights * F.mse_loss(model_output.float(), x_bits.float(), reduction="none")
            return loss.mean()
        elif prediction_type == "epsilon":
            return F.mse_loss(model_output.float(), noise.float())
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    def _extract_class_labels(self, batch):
        """Extract and convert to 0-indexed for embedding.
        For unconditional: returns the unconditional token (num_classes) for all samples,
        since ada_norm_zero always requires class_labels.
        """
        if self.conditioning == "num_boxes":
            return (batch["num_boxes"] - 1).long()
        elif self.conditioning == "k_steps":
            return self.k_to_class[batch["k"]]  # type: ignore[index] # map k values to class indices

        return torch.full(   # unconditional: model still needs class_labels for ada_norm_zero
            (batch["target"].shape[0],), self.num_classes, dtype=torch.long, device=batch["target"].device
        )

    def training_step(self, batch, batch_idx):
        x_bits = batch["target"]  # (B, num_bits, H, W) in {-1, 1}
        class_labels = self._extract_class_labels(batch)
        cond_board = batch.get("condition", None)

        loss = self._compute_loss(x_bits, class_labels, cond_board)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_bits = batch["target"]
        class_labels = self._extract_class_labels(batch)
        cond_board = batch.get("condition", None)

        # Deterministic validation: fork + fixed-seed the RNG so the timesteps, noise and
        # self-conditioning coin-flip sampled inside _compute_loss are reproducible and
        # comparable across epochs (stabilizes the val/loss used for checkpoint selection).
        fork_devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(42 + batch_idx)
            loss = self._compute_loss(x_bits, class_labels, cond_board)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return
        if not self.trainer.is_global_zero:
            return

        self._run_sokoban_evaluation()

    # Sampling
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
        """Generate boards via iterative denoising with DDPMScheduler and optional CFG.
        Returns bit predictions (B, num_bits, H, W) as floats.
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        use_cfg = guidance_scale > 1.0 and class_labels is not None and self.conditioning != "unconditional"

        x_t = torch.randn(batch_size, self.num_bits, self.resolution, self.resolution, generator=generator, device=device)
        x_pred = None  # for self-conditioning carry-over
        x_pred_step = x_t  # fallback; overwritten in loop

        self.noise_scheduler.set_timesteps(self.inference_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device)

            # Asymmetric time intervals: feed model a shifted timestep to encourage stronger denoising
            if self.time_shift_xi > 0.0:
                num_train_steps = self.noise_scheduler.config.num_train_timesteps
                t_model = torch.clamp(t_batch + int(self.time_shift_xi * num_train_steps), max=num_train_steps - 1)
            else:
                t_model = t_batch

            is_sample_pred = self.noise_scheduler.config.prediction_type == "sample"

            if use_cfg:
                # Batch cond + uncond in one forward pass (2*B)
                uncond_labels = torch.full_like(class_labels, self.num_classes)  # type: ignore
                uncond_cond_board = torch.zeros_like(cond_board) if cond_board is not None else None

                model_input_cond = self._build_model_input(x_t, x_pred, cond_board)
                model_input_uncond = self._build_model_input(x_t, x_pred, uncond_cond_board)

                model_input_both = torch.cat([model_input_cond, model_input_uncond], dim=0)
                t_both = torch.cat([t_model, t_model], dim=0)
                labels_both = torch.cat([class_labels, uncond_labels], dim=0)  # type: ignore

                out_both = self._model_forward(model_input_both, t_both, labels_both)
                x_pred_cond, x_pred_uncond = out_both.chunk(2, dim=0)

                # CFG combination
                x_pred_step = x_pred_uncond + guidance_scale * (x_pred_cond - x_pred_uncond)
                if is_sample_pred:
                    x_pred_step = x_pred_step.clamp(-1.0, 1.0)
            else:
                model_input = self._build_model_input(x_t, x_pred, cond_board)
                x_pred_step = self._model_forward(model_input, t_model, class_labels)
                if is_sample_pred:
                    x_pred_step = x_pred_step.clamp(-1.0, 1.0)

            x_t = self.noise_scheduler.step(x_pred_step, t, x_t, generator=generator).prev_sample

            x_pred = x_pred_step if self.self_cond else None

        return x_pred_step

    # Sokoban Evaluation
    @torch.no_grad()
    def evaluate(self, val_dataloader, num_samples: Optional[int] = None):
        """Generate boards and compute metrics.
        Returns:
            metrics: dict of sokoban metric name -> float
            gen_bits: (N, num_bits, H, W) tensor of raw bit predictions
        """
        device = self.device
        if num_samples is None:
            num_samples = self.num_eval_samples
        batch_size = min(50, num_samples)

        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(42)

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
                batch_size=x_bits.shape[0],
                device=device,
                class_labels=class_labels,
                cond_board=cond_board,
                generator=val_generator
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

        all_gen_np = np.concatenate(all_gen_boards, axis=0)[:num_samples]
        all_gen_np = np.clip(all_gen_np, 0, 6)
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
        val_dl = self.trainer.datamodule.val_dataloader()  # type: ignore[union-attr]
        metrics, gen_bits = self.evaluate(val_dl)

        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)

        if isinstance(self.logger, WandbLogger):
            self._log_board_renders(gen_bits[:16])

    def _log_board_renders(self, gen_bits: torch.Tensor):
        """Render generated bit boards using dataset sprites and log as WandB images."""
        val_ds = self.trainer.datamodule.val_ds  # type: ignore[union-attr]
        rendered = []
        for i in range(gen_bits.shape[0]):
            img = val_ds.render_bit_boards(gen_bits[i])
            rendered.append(wandb.Image(img, caption=f"epoch={self.current_epoch}"))

        self.logger.experiment.log({"sokoban/generated_boards": rendered})  # type: ignore[union-attr]

    # Optimizer
    def configure_optimizers(self):  # type: ignore[override]
        optimizer = AdamW(
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class EMACallback(Callback):
    """EMA via diffusers EMAModel: swaps EMA weights in for validation, bakes them into checkpoints.

    Checkpoints serve both inference and resume: state_dict holds EMA weights (for sample.py), while raw_model_state/ema_optimization_step carry the real training weights and EMA warmup counter so resume continues the true trajectory instead of the smoothed EMA point.
    """
    def __init__(self, decay: float = 0.9999, inv_gamma: float = 1.0, power: float = 0.75):
        super().__init__()
        self.decay = decay
        self.inv_gamma = inv_gamma
        self.power = power
        self.ema_model: Optional[EMAModel] = None
        self._pending_raw_state: Optional[dict] = None
        self._pending_ema_step: Optional[int] = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema_model is None:
            self.ema_model = EMAModel(
                pl_module.model.parameters(),
                decay=self.decay,
                use_ema_warmup=True,
                inv_gamma=self.inv_gamma,
                power=self.power,
            )
        if self._pending_ema_step is not None:
            self.ema_model.optimization_step = self._pending_ema_step
            self._pending_ema_step = None
        if self._pending_raw_state is not None:
            with torch.no_grad():
                for name, p in pl_module.model.named_parameters():
                    if name in self._pending_raw_state:
                        p.data.copy_(self._pending_raw_state[name].to(p.device))
            self._pending_raw_state = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert self.ema_model is not None
        # Step EMA once per OPTIMIZER update, not once per batch.
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return
        self.ema_model.step(pl_module.model.parameters())

    def on_validation_epoch_start(self, trainer, pl_module):
        assert self.ema_model is not None
        self.ema_model.store(pl_module.model.parameters())
        self.ema_model.copy_to(pl_module.model.parameters())

    def on_validation_epoch_end(self, trainer, pl_module):
        assert self.ema_model is not None
        self.ema_model.restore(pl_module.model.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is None:
            return
        checkpoint["raw_model_state"] = {
            name: p.detach().cpu().clone() for name, p in pl_module.model.named_parameters()
        }
        checkpoint["ema_optimization_step"] = self.ema_model.optimization_step
        for i, (name, _) in enumerate(pl_module.model.named_parameters()):
            key = f"model.{name}"
            if key in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.ema_model.shadow_params[i].clone()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self._pending_raw_state = checkpoint.get("raw_model_state", None)
        self._pending_ema_step = checkpoint.get("ema_optimization_step", None)


class SokobanBitDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        val_data_path: Optional[str] = None,
        conditioning: str = "unconditional",
        total_train_size: int = 64000,
        total_eval_size: int = 6400,
        batch_size: int = 64,
        num_workers: int = 4,
        num_bits: int = 3,
        k_values: Optional[List[int]] = None,
        use_dihedral_aug: bool = False,
        bot_removal_prob: float = 0.75,
    ):
        super().__init__()
        self.data_path = data_path
        self.val_data_path = val_data_path or data_path
        self.conditioning = conditioning
        self.total_train_size = total_train_size
        self.total_eval_size = total_eval_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.k_values = k_values or [1, 3, 5, 8, 10]
        self.use_dihedral_aug = use_dihedral_aug
        self.bot_removal_prob = bot_removal_prob

    def setup(self, stage=None):
        if self.conditioning == "num_boxes":
            self.train_ds = SokobanBitsDataset.for_conditioning_num_boxes_generation(
                self.data_path, self.total_train_size
            )
            self.val_ds = SokobanBitsDataset.for_conditioning_num_boxes_generation(
                self.val_data_path, self.total_eval_size
            )
        elif self.conditioning == "k_steps":
            self.train_ds = SokobanBitsDataset.for_new_k_steps_conditioning_generation(
                self.data_path, self.total_train_size, k_values=self.k_values,
                bot_removal_prob=self.bot_removal_prob,
            )
            self.val_ds = SokobanBitsDataset.for_new_k_steps_conditioning_generation(
                self.val_data_path, self.total_eval_size, k_values=self.k_values,
                bot_removal_prob=self.bot_removal_prob,
            )
        elif self.conditioning == "k_steps_old":
            self.train_ds = SokobanBitsDataset.for_conditioning_k_steps_generation(
                self.data_path, self.total_train_size, k_values=self.k_values,
                bot_removal_prob=self.bot_removal_prob,
            )
            self.val_ds = SokobanBitsDataset.for_conditioning_k_steps_generation(
                self.val_data_path, self.total_eval_size, k_values=self.k_values,
                bot_removal_prob=self.bot_removal_prob,
            )
        else:  # unconditional
            self.train_ds = SokobanBitsDataset.for_unconditional(
                self.data_path, self.total_train_size,
                bot_removal_prob=self.bot_removal_prob,
            )
            self.val_ds = SokobanBitsDataset.for_unconditional(
                self.val_data_path, self.total_eval_size,
                bot_removal_prob=self.bot_removal_prob,
            )

        self.train_ds.num_bits = self.num_bits
        self.val_ds.num_bits = self.num_bits
        self.train_ds.use_dihedral_aug = self.use_dihedral_aug
        self.val_ds.use_dihedral_aug = False

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


@hydra.main(version_base=None, config_path="config", config_name="standard_diffusion")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    num_bits = cfg.num_bits
    k_values = cfg.dataset.get("k_values", [1, 3, 5, 8, 10])
    num_classes = len(k_values) if cfg.conditioning == "k_steps" else cfg.num_classes

    use_self_cond = cfg.get("self_cond", False)
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * (2 if use_self_cond else 1) + num_bits  # noisy + [self_cond] + spatial_cond
    else:
        in_channels = num_bits * (2 if use_self_cond else 1)  # noisy + [self_cond]

    cross_attention_dim = cfg.model.get("cross_attention_dim", None)

    model = Transformer2DModel(
        sample_size=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_layers=cfg.model.num_layers,
        patch_size=cfg.model.patch_size,
        attention_head_dim=cfg.model.attention_head_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        cross_attention_dim=cross_attention_dim if cross_attention_dim else None,
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        num_embeds_ada_norm=num_classes + 1,
        norm_type="ada_norm_zero",
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    lit_model = SokobanBitDiffusion(
        model=model,
        noise_scheduler=noise_scheduler,
        conditioning=cfg.conditioning,
        num_classes=num_classes,
        num_bits=num_bits,
        resolution=cfg.resolution,
        inference_steps=cfg.get("inference_steps", 400),
        self_cond=use_self_cond,
        cfg_drop_rate=cfg.cfg_drop_rate,
        guidance_scale=cfg.guidance_scale,
        time_shift_xi=cfg.get("time_shift_xi", 0.0),
        lr=cfg.lr,
        betas=tuple(cfg.get("adam_betas", [0.95, 0.999])),
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.get("warmup_steps", 500),
        num_epochs=cfg.num_epochs,
        eval_every_n_epochs=cfg.eval_every_n_epochs,
        num_eval_samples=cfg.num_eval_samples,
        k_values=k_values if cfg.conditioning == "k_steps" else None,
    )

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
        use_dihedral_aug=cfg.dataset.get("use_dihedral_aug", False),
        bot_removal_prob=cfg.dataset.get("bot_removal_prob", 0.75),
    )

    run_name = cfg.get("run_name", None) or f"standard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "Sokoban-BitDiffusion"),
        group=cfg.get("wandb_group", None),
        name=run_name,
        save_dir=cfg.output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    ema_decay = cfg.get("ema_decay", 0.9999)
    ckpt_dir = Path(cfg.output_dir) / run_name / "std_checkpoints"
    callbacks = [
        EMACallback(decay=ema_decay, inv_gamma=1.0, power=0.75),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch}-{step}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,  # last.ckpt for resume
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="periodic-{epoch}",
            every_n_epochs=cfg.save_every_n_epochs,
            save_top_k=-1,
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
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get("gradient_accumulation_steps", 1),
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # Persist the W&B run id next to the checkpoints so sample.py can resume this same experiment and log test metrics onto these training charts.
    if rank_zero_only.rank == 0:
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "wandb_run_id.txt").write_text(str(wandb_logger.experiment.id))
        except Exception as e:
            print(f"Could not save W&B run id: {e}")

    ckpt_path = cfg.get("resume_from_checkpoint", None)
    trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
