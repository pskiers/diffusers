"""
Training script for Discrete TRM Diffusion on Sokoban boards.

Architecture: Option III hybrid — absorbing-state masked diffusion with a TRM
reasoning loop at each denoising step. Carry persists across diffusion timesteps
at inference; at training each batch samples a single random timestep t.

Uses Hydra config, HF Accelerate for mixed-precision, and WandB/TensorBoard logging.
Reuses existing SokobanDataset, SokobanEvaluator, and SokobanSampler infrastructure.

Comprehensive W&B logging covers:
  - TRM metrics: halt steps distribution, q-head accuracy, exact/token accuracy
  - Diffusion metrics: loss per mask-ratio bucket, prediction quality vs timestep
  - Training metrics: lr, grad norm, epoch/step counters
  - Validation metrics: all of the above in val/ namespace
  - Sampling metrics: board validity, solvability, diversity (from SokobanEvaluator)
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import timedelta
from typing import Dict

import hydra
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sokoban.diffusion.discrete_trm import (
    ACTMaskedDiffusionWrapper,
    DiscreteTRMDiffusion,
)
from sokoban.diffusion.mask_schedule import AbsorbingMaskSchedule
from sokoban.diffusion.sokoban_dataset_diffusion import (
    GroupBatchSampler,
    SokobanDataset,
)
from sokoban.diffusion.sokoban_token_dataset import SokobanTokenDataset
from sokoban.sokoban_utils import SokobanEvaluator

logger = get_logger(__name__, log_level="INFO")


# ─── Metric helpers ───────────────────────────────────────────────────────────


def _safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _bucket_name(mask_ratio: float) -> str:
    """Map mask ratio to a human-readable bucket name."""
    if mask_ratio < 0.25:
        return "low_mask_0_25"
    elif mask_ratio < 0.50:
        return "mid_mask_25_50"
    elif mask_ratio < 0.75:
        return "high_mask_50_75"
    else:
        return "very_high_mask_75_100"


class MetricAccumulator:
    """Accumulates per-step metrics and computes epoch-level aggregates.

    Supports bucketed tracking by mask ratio for diffusion-specific analysis.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.values: Dict[str, list] = defaultdict(list)
        self.bucket_values: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.halt_step_counts: Dict[int, int] = defaultdict(int)

    def update(self, metrics: dict, mask_ratio: float = 0.0, halt_steps: float = 0.0):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.values[k].append(v)
        # Bucket by mask ratio
        bucket = _bucket_name(mask_ratio)
        for k in ("lm_loss", "token_accuracy", "exact_accuracy"):
            if k in metrics:
                self.bucket_values[bucket][k].append(metrics[k])
        # Halt step histogram
        self.halt_step_counts[int(round(halt_steps))] += 1

    def aggregate(self, prefix: str = "train") -> dict:
        result = {}
        for k, vals in self.values.items():
            result[f"{prefix}/{k}"] = _safe_mean(vals)
        # Bucketed metrics
        for bucket, bvals in self.bucket_values.items():
            for k, vals in bvals.items():
                result[f"{prefix}/{bucket}/{k}"] = _safe_mean(vals)
        return result

    def halt_histogram(self) -> dict:
        """Return halt step distribution as {step: fraction}."""
        total = sum(self.halt_step_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(self.halt_step_counts.items())}


# ─── Sampling & evaluation ────────────────────────────────────────────────────


@torch.no_grad()
def run_sampling_evaluation(
    model: ACTMaskedDiffusionWrapper,
    schedule: AbsorbingMaskSchedule,
    args: DictConfig,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> dict:
    """Generate boards, evaluate with SokobanEvaluator, log to WandB."""
    base_model = accelerator.unwrap_model(model)
    inner = base_model.inner
    inner.eval()

    evaluator = SokobanEvaluator(args.dataset.num_boxes)
    num_samples = args.num_samples
    batch_size = min(args.sample_batch_size, num_samples)
    device = accelerator.device

    all_boards = []

    for start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - start)
        tokens, _ = schedule.sample(
            inner,
            batch_size=bsz,
            seq_len=144,
            device=device,
            generator=None,  # multinomial samples on CPU internally
            temperature=args.get("sample_temperature", 1.0),
        )
        # tokens [B, 144] ∈ {0,...,7}. Evaluator expects FieldStates IDs 1-7.
        # MASK=0 residue will naturally fail validity checks.
        boards_np = tokens.cpu().numpy().reshape(-1, 12, 12)
        all_boards.append(boards_np)

    all_boards_np = np.concatenate(all_boards, axis=0)

    sokoban_metrics = evaluator.generate_metrics(
        generated_boards=all_boards_np,
        conditioning_boards=None,
        target_boards=None,
        k_values=None,
        n_images_per_conditioning=1,
    )

    logger.info(f"Epoch {epoch} sokoban metrics: {sokoban_metrics}")
    accelerator.log(sokoban_metrics, step=global_step)

    # Count residual MASK tokens in generated boards
    mask_residue = (all_boards_np == 0).sum() / max(all_boards_np.size, 1)
    accelerator.log({"sample/mask_residue_ratio": mask_residue}, step=global_step)

    # Rendered board grid for WandB
    if args.logger == "wandb" and accelerator.is_main_process:
        try:
            import wandb
            from torchvision.utils import make_grid
            from sokoban.sokoban_utils import SokobanSampler

            sampler = SokobanSampler(args)
            n_show = min(64, len(all_boards_np))
            # Renderer uses board value as index into 7-element surface array;
            # our tokens are 1-7 (FieldStates IDs) but renderer expects 0-6.
            boards_for_render = np.clip(all_boards_np[:n_show] - 1, 0, 6)
            sampler.all_gen_boards_list = [boards_for_render]
            rendered = sampler.render_boards()
            n_cols = min(8, int(math.ceil(math.sqrt(n_show) * 1.5)))
            grid = make_grid(rendered, nrow=n_cols, padding=2)
            accelerator.get_tracker("wandb").log(
                {"sokoban_samples": wandb.Image(grid), "epoch": epoch},
                step=global_step,
            )
        except Exception as e:
            logger.warning(f"Failed to render WandB board grid: {e}")

    return sokoban_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # ── 1. Accelerator ──
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir
        ),
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── 2. Dataset ──
    train_base = SokobanDataset(
        data_path=args.dataset.train_data_dir,
        k=0,
        max_trajectories=args.dataset.get("max_trajectories", None),
        max_boards=args.dataset.get("max_training_boards", None),
    )
    eval_base = SokobanDataset(
        data_path=args.dataset.eval_data_dir,
        k=0,
        max_boards=args.dataset.get("max_validation_boards", None),
    )

    train_ds = SokobanTokenDataset(train_base)
    eval_ds = SokobanTokenDataset(eval_base)

    logger.info(f"Train dataset: {len(train_ds)} samples, {train_base.num_trajectories} trajectories")
    logger.info(f"Eval dataset: {len(eval_ds)} samples, {eval_base.num_trajectories} trajectories")

    def collate_fn(examples):
        return {"tokens": torch.stack([ex["tokens"] for ex in examples])}

    train_sampler = GroupBatchSampler(
        train_base.group_boundaries, batch_size=args.train_batch_size, drop_last=True,
    )
    eval_sampler = GroupBatchSampler(
        eval_base.group_boundaries, batch_size=args.eval_batch_size, drop_last=True,
    )

    n_workers = args.dataloader_num_workers
    train_dl = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=n_workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=(n_workers > 0),
    )
    eval_dl = DataLoader(
        eval_ds, batch_sampler=eval_sampler, num_workers=n_workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=(n_workers > 0),
    )

    # ── 3. Model ──
    inner_model = DiscreteTRMDiffusion(
        vocab_size=args.model.vocab_size,
        seq_len=args.model.seq_len,
        hidden_size=args.model.hidden_size,
        num_heads=args.model.num_heads,
        L_layers=args.model.L_layers,
        L_cycles=args.model.L_cycles,
        H_cycles=args.model.H_cycles,
        expansion=args.model.expansion,
        forward_dtype=args.model.forward_dtype,
        pos_encodings=args.model.pos_encodings,
    )

    model = ACTMaskedDiffusionWrapper(
        inner=inner_model,
        halt_max_steps=args.model.halt_max_steps,
        halt_exploration_prob=args.model.halt_exploration_prob,
        halt_loss_weight=args.model.get("halt_loss_weight", 0.5),
        use_carry_recycling=args.model.get("use_carry_recycling", False),
        carry_recycle_prob=args.model.get("carry_recycle_prob", 0.5),
        halt_warmup_steps=args.model.get("halt_warmup_steps", 0),
    )

    # Optional: compile for throughput
    if args.get("compile", False):
        model.inner = torch.compile(model.inner, mode="reduce-overhead")  # type: ignore[assignment]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── 4. Mask schedule ──
    schedule = AbsorbingMaskSchedule(num_steps=args.mask_diffusion_steps)

    # ── 5. Optimizer & LR scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optimizer.lr,
        betas=tuple(args.optimizer.betas),
        weight_decay=args.optimizer.weight_decay,
        eps=args.optimizer.eps,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    total_training_steps = num_update_steps_per_epoch * args.num_epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ── 6. Accelerate prepare ──
    model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, lr_scheduler
    )

    # ── 7. Resume from checkpoint ──
    global_step = 0
    first_epoch = 0
    resume_step = 0
    grad_norm = 0.0

    if args.resume_from_checkpoint:
        ckpt_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")] if os.path.exists(args.output_dir) else []
        if ckpt_dirs:
            if args.resume_from_checkpoint == "latest":
                ckpt_path = os.path.join(
                    args.output_dir, sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]
                )
            else:
                ckpt_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
            if os.path.exists(ckpt_path):
                accelerator.load_state(ckpt_path)
                global_step = int(ckpt_path.split("-")[-1])
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
                logger.info(f"Resumed from {ckpt_path} at step {global_step}")

    # ── 8. WandB / tracker setup ──
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        if args.logger == "wandb":
            accelerator.init_trackers(
                args.get("project_name", "discrete-trm-sokoban"),
                config=tracker_config,
                init_kwargs={"wandb": {"name": args.get("run_name", None)}},
            )
        elif args.logger == "tensorboard":
            accelerator.init_trackers("discrete-trm-sokoban", config=tracker_config)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_training_steps}")
    logger.info(f"  Mask diffusion steps T = {args.mask_diffusion_steps}")
    logger.info(f"  TRM: H={args.model.H_cycles} L={args.model.L_cycles} D={args.model.hidden_size} halt_max={args.model.halt_max_steps}")
    logger.info(f"  Carry recycling: {args.model.get('use_carry_recycling', False)}")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        train_metrics = MetricAccumulator()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dl):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            x_0 = batch["tokens"].to(accelerator.device)  # [B, 144]
            B = x_0.shape[0]

            # Sample t ∈ {1, ..., T}
            t = torch.randint(1, schedule.num_steps + 1, (B,), device=accelerator.device)

            # Forward mask: x_0 → x_t
            x_t = schedule.forward_mask(x_0, t)

            with accelerator.accumulate(model):
                loss, metrics, _ = model(x_t, x_0, t, diffusion_carry=None)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Track metrics
            train_metrics.update(
                metrics,
                mask_ratio=metrics.get("mask_ratio", 0.0),
                halt_steps=metrics.get("avg_halt_steps", 0.0),
            )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Per-step logging (every step for W&B)
                base_model = accelerator.unwrap_model(model)
                step_logs = {
                    "train/lm_loss": metrics["lm_loss"],
                    "train/halt_loss": metrics["halt_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "train/token_accuracy": metrics["token_accuracy"],
                    "train/exact_accuracy": metrics["exact_accuracy"],
                    "train/avg_halt_steps": metrics["avg_halt_steps"],
                    "train/current_max_halt": base_model.get_current_max_halt(),
                    "train/q_halt_mean": metrics["q_halt_mean"],
                    "train/q_halt_pos_frac": metrics["q_halt_pos_frac"],
                    "train/mask_ratio": metrics["mask_ratio"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                if isinstance(grad_norm, torch.Tensor):
                    step_logs["train/grad_norm"] = grad_norm.item()
                elif isinstance(grad_norm, (int, float)):
                    step_logs["train/grad_norm"] = float(grad_norm)

                accelerator.log(step_logs, step=global_step)

                progress_bar.set_postfix(
                    loss=f"{metrics['total_loss']:.4f}",
                    acc=f"{metrics['token_accuracy']:.3f}",
                    halt=f"{metrics['avg_halt_steps']:.1f}",
                )

                # Checkpoint
                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(ckpts) >= args.checkpoints_total_limit:
                            for rm in ckpts[: len(ckpts) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, rm))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

        progress_bar.close()

        # ── Epoch-level aggregate metrics ──
        epoch_agg = train_metrics.aggregate("train")
        halt_hist = train_metrics.halt_histogram()
        if halt_hist:
            for step_k, frac in halt_hist.items():
                epoch_agg[f"train/halt_dist/step_{step_k}"] = frac
        accelerator.log(epoch_agg, step=global_step)

        accelerator.wait_for_everyone()

        # ── Validation ──
        model.eval()
        val_metrics = MetricAccumulator()
        val_pbar = tqdm(total=len(eval_dl), disable=not accelerator.is_local_main_process)
        val_pbar.set_description(f"Val {epoch}")

        for batch in eval_dl:
            x_0 = batch["tokens"].to(accelerator.device)
            B = x_0.shape[0]
            t = torch.randint(1, schedule.num_steps + 1, (B,), device=accelerator.device)
            x_t = schedule.forward_mask(x_0, t)

            with torch.no_grad():
                loss, metrics, _ = model(x_t, x_0, t, diffusion_carry=None)

            val_metrics.update(
                metrics,
                mask_ratio=metrics.get("mask_ratio", 0.0),
                halt_steps=metrics.get("avg_halt_steps", 0.0),
            )
            val_pbar.update(1)

        val_pbar.close()

        val_agg = val_metrics.aggregate("val")
        val_halt_hist = val_metrics.halt_histogram()
        if val_halt_hist:
            for step_k, frac in val_halt_hist.items():
                val_agg[f"val/halt_dist/step_{step_k}"] = frac
        accelerator.log(val_agg, step=global_step)

        if val_agg:
            logger.info(
                f"Epoch {epoch} val: "
                f"loss={val_agg.get('val/lm_loss', 0):.4f} "
                f"acc={val_agg.get('val/token_accuracy', 0):.4f} "
                f"exact={val_agg.get('val/exact_accuracy', 0):.4f} "
                f"halt={val_agg.get('val/avg_halt_steps', 0):.1f}"
            )

        # ── Sampling & Board Evaluation ──
        if (
            accelerator.is_main_process
            and epoch > 0
            and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1)
        ):
            run_sampling_evaluation(model, schedule, args, accelerator, epoch, global_step)

        # ── Save model ──
        if (
            accelerator.is_main_process
            and epoch > 0
            and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1)
        ):
            save_dir = os.path.join(args.output_dir, "model")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))
            OmegaConf.save(args, os.path.join(save_dir, "config.yaml"))
            logger.info(f"Saved model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
