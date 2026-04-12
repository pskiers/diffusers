"""
train_mnist_sudoku.py – Train a Ratatouille MNIST-Sudoku model.

Usage:
    python train_mnist_sudoku.py experiment=v0
    python train_mnist_sudoku.py experiment=v1 train.sudoku_loss_weight=0.1
    accelerate launch --num_processes=2 train_mnist_sudoku.py experiment=v2

The experiment config selects the model variant and any overrides.
"""

import inspect
import os
import math
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mnist_eval import evaluate_grids, load_or_train_classifier, sample_grids
from mnist_sudoku_dataset import MNISTSudokuDataset
from mnist_sudoku_models import (
    MNISTRatatouilleV0,
    MNISTRatatouilleV1,
    MNISTRatatouilleV2,
    MNISTRatatouilleV3,
    MNISTRatatouilleV4,
)


logger = get_logger(__name__, log_level="INFO")

MODEL_REGISTRY = {
    "v0": MNISTRatatouilleV0,
    "v1": MNISTRatatouilleV1,
    "v2": MNISTRatatouilleV2,
    "v3": MNISTRatatouilleV3,
    "v4": MNISTRatatouilleV4,
}

IGNORE_LABEL_ID = -100


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ── Noise scheduler helper ─────────────────────────────────────────────────────

def add_noise(
    scheduler: DDPMScheduler,
    images: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    return scheduler.add_noise(images, noise, timesteps)


# ── Training step ──────────────────────────────────────────────────────────────

def train_step(
    model,
    batch: dict,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    sudoku_loss_weight: float = 0.0,
) -> dict:
    images     = batch["images"].to(accelerator.device)      # (B,1,H,W)
    conditions = batch["conditions"].to(accelerator.device)  # (B,1,H,W)
    solution   = batch["solution"].to(accelerator.device)    # (B,81) long

    B = images.shape[0]
    noise     = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,),
                              device=accelerator.device, dtype=torch.long)

    noisy = add_noise(scheduler, images, noise, timesteps)

    noise_pred, sudoku_logits = model(noisy, timesteps, conditions)

    # Ratatouille (diffusion) loss – simple MSE on predicted noise
    diff_loss = F.mse_loss(noise_pred, noise)

    # Optional sudoku CE loss (only for V0 and V1)
    sudoku_loss = torch.tensor(0.0, device=accelerator.device)
    if sudoku_logits is not None and sudoku_loss_weight > 0:
        # sudoku_logits: (B, N, num_classes) where N = 81 for V0/V1
        B_, N, C = sudoku_logits.shape
        # solution may have more/fewer positions than logits (V0/V1 always 81)
        sudoku_loss = F.cross_entropy(
            sudoku_logits.reshape(B_ * N, C),
            solution[:, :N].reshape(B_ * N),
            ignore_index=IGNORE_LABEL_ID,
        )

    total_loss = diff_loss + sudoku_loss_weight * sudoku_loss
    return {
        "loss":         total_loss,
        "diff_loss":    diff_loss,
        "sudoku_loss":  sudoku_loss,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loop(
    model,
    dataloader: DataLoader,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    sudoku_loss_weight: float = 0.0,
) -> dict:
    model.eval()
    metrics = {"loss": [], "diff_loss": [], "sudoku_loss": []}

    for batch in dataloader:
        m = train_step(model, batch, scheduler, accelerator, sudoku_loss_weight)
        for k in metrics:
            metrics[k].append(m[k].item())

    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save(accelerator, model, optimizer, step, output_dir, tag):
    ckpt = {
        "step":            step,
        "model_state":     accelerator.unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs/mnist_sudoku", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision=args.get("mixed_precision", "no"))
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(args))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dir = os.path.join(args.data.sudoku_dir, "train")
    test_dir  = os.path.join(args.data.sudoku_dir, "test")

    cell_size = args.data.get("cell_size", 32)

    train_ds = MNISTSudokuDataset(
        sudoku_dir=train_dir,
        mnist_root=args.data.get("mnist_root", "data/mnist"),
        cell_size=cell_size,
        mnist_split="train",
        mask_given=True,
    )
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=test_dir if os.path.isdir(test_dir) else train_dir,
        mnist_root=args.data.get("mnist_root", "data/mnist"),
        cell_size=cell_size,
        mnist_split="test",
        mask_given=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.get("num_workers", 4),
        drop_last=True,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.train.batch_size * 2,
        shuffle=False,
        num_workers=args.get("num_workers", 4),
    )

    # ── MNIST cell classifier (for digit-level eval) ──────────────────────────
    classifier = None
    classifier_path = args.get("eval_classifier_path", None)
    if classifier_path and accelerator.is_main_process:
        classifier = load_or_train_classifier(
            classifier_path,
            args.data.get("mnist_root", "data/mnist"),
            cell_size,
            accelerator.device,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    variant   = args.model.variant   # "v0", "v1", ..., "v4"
    ModelCls  = MODEL_REGISTRY[variant]
    painter_size = cell_size * 9     # e.g. 32*9=288

    model_kwargs = dict(OmegaConf.to_container(args.model.get("kwargs", {}), resolve=True))
    model_kwargs.setdefault("painter_size", painter_size)
    model_kwargs.setdefault("cell_size", cell_size)
    valid_params = set(inspect.signature(ModelCls.__init__).parameters) - {"self"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    model = ModelCls(**model_kwargs)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model {variant}: {n_params:,} parameters")

    # ── Noise scheduler ───────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        num_train_timesteps=args.get("num_timesteps", 1000),
        beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type="epsilon",
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.train.lr,
        betas=(0.9, 0.95),
        weight_decay=args.train.get("weight_decay", 0.01),
    )

    model, optimizer, train_dl, eval_dl = accelerator.prepare(
        model, optimizer, train_dl, eval_dl
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    resume_path = args.get("resume_from_checkpoint", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = int(ckpt["step"])
        logger.info(f"Resumed from {resume_path} at step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    num_steps    = args.train.num_steps
    warmup_steps = args.train.get("warmup_steps", 500)
    eval_every   = args.train.get("eval_every", 1000)
    save_every   = args.train.get("save_every", 5000)
    log_every    = args.train.get("log_every", 100)
    sudoku_w     = args.train.get("sudoku_loss_weight", 0.0)

    best_loss   = float("inf")
    train_iter  = iter(train_dl)

    progress_bar = tqdm(
        total=num_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    while global_step < num_steps:
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # LR update
        lr = get_lr(global_step, warmup_steps, num_steps, args.train.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with accelerator.accumulate(model):
            m = train_step(
                accelerator.unwrap_model(model),
                batch,
                scheduler,
                accelerator,
                sudoku_loss_weight=sudoku_w,
            )
            accelerator.backward(m["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        if global_step % log_every == 0 and accelerator.is_main_process:
            logger.info(
                f"step={global_step}  loss={m['loss'].item():.4f}  "
                f"diff={m['diff_loss'].item():.4f}  "
                f"sudoku={m['sudoku_loss'].item():.4f}  lr={lr:.2e}"
            )

        if global_step % eval_every == 0:
            metrics = eval_loop(
                accelerator.unwrap_model(model),
                eval_dl, scheduler, accelerator,
                sudoku_loss_weight=sudoku_w,
            )
            if accelerator.is_main_process:
                logger.info(
                    f"[eval] step={global_step}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"diff={metrics['diff_loss']:.4f}  "
                    f"sudoku={metrics['sudoku_loss']:.4f}"
                )
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    _save(accelerator, model, optimizer, global_step, args.output_dir, "best")

                # Digit-level eval via DDIM sampling + classifier
                if classifier is not None:
                    n_eval    = args.get("eval_num_samples", 16)
                    n_steps   = args.get("eval_num_ddim_steps", 20)
                    eval_batch = next(iter(DataLoader(eval_ds, batch_size=n_eval, shuffle=False)))
                    conditions = eval_batch["conditions"]
                    solutions  = eval_batch["solution"]
                    generated  = sample_grids(
                        accelerator.unwrap_model(model),
                        conditions,
                        num_train_timesteps=args.get("num_timesteps", 1000),
                        beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
                        num_steps=n_steps,
                        device=accelerator.device,
                    )
                    acc = evaluate_grids(generated, solutions, classifier, cell_size)
                    logger.info(
                        f"[eval] cell_acc={acc['cell_acc']:.4f}  "
                        f"puzzle_acc={acc['puzzle_acc']:.4f}"
                    )

        if global_step % save_every == 0 and accelerator.is_main_process:
            _save(accelerator, model, optimizer, global_step, args.output_dir, f"step-{global_step}")

    if accelerator.is_main_process:
        _save(accelerator, model, optimizer, global_step, args.output_dir, "final")
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
