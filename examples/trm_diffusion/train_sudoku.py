"""
train_sudoku.py – Train a SudokuTRM model on a pre-processed Sudoku dataset.

Usage (single GPU):
    python train_sudoku.py data_dir=data/sudoku-extreme-1k

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes=2 train_sudoku.py data_dir=data/sudoku-extreme-1k

Hydra config overrides (examples):
    python train_sudoku.py \
        data_dir=data/sudoku-extreme-1k \
        output_dir=runs/sudoku_trm \
        model.d_model=256 model.n_heads=4 model.n_layers=2 \
        model.L_cycles=6 model.H_cycles=3 model.n_sup=4 \
        train.lr=1e-4 train.batch_size=256 train.num_steps=50000
"""

import os
import math
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from sudoku_dataset import SudokuDataset, IGNORE_LABEL_ID
from sudoku_models import SudokuTRM


logger = get_logger(__name__, log_level="INFO")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor):
    """
    Args:
        logits : (B, 81, vocab_size)
        labels : (B, 81)   – IGNORE_LABEL_ID on given cells
        inputs : (B, 81)   – original puzzle (1 = blank, 2-10 = given digit)

    Returns dict with:
        cell_acc   – accuracy on blank cells only
        puzzle_acc – fraction of fully solved puzzles
        loss       – cross-entropy on blank cells
    """
    preds = logits.argmax(-1)                              # (B, 81)
    blank_mask = labels != IGNORE_LABEL_ID                 # only blank cells

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1).clamp(min=0),                      # clamp so -100 → 0 doesn't crash
        ignore_index=IGNORE_LABEL_ID,
        reduction="mean",
    )

    correct_cells = (preds == labels) & blank_mask         # (B, 81)
    cell_acc = correct_cells.float().sum() / blank_mask.float().sum().clamp(min=1)

    # A puzzle is solved only if every blank cell is correct
    solved = correct_cells.sum(-1) == blank_mask.sum(-1)   # (B,)
    puzzle_acc = solved.float().mean()

    return {"loss": loss, "cell_acc": cell_acc, "puzzle_acc": puzzle_acc}


# ── Training ───────────────────────────────────────────────────────────────────

def train_step(
    model: SudokuTRM,
    batch,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Run n_sup supervision steps, each with its own backward + optimizer update.

    Mirrors the ACT training loop: the reference model calls backward() and
    optimizer.step() once per reasoning step.  Here we do the same — n_sup
    steps, each independently back-propagated and applied.

    Memory is O(1 step) regardless of n_sup because z_H/z_L are detached
    between steps so each step's graph is independent; calling backward()
    immediately after each step frees its activations before the next step.

    Returns the mean loss value (float) for logging.
    """
    inputs  = batch["inputs"].to(accelerator.device)   # (B, 81)
    labels  = batch["labels"].to(accelerator.device)   # (B, 81)
    puzzle_ids = batch.get("puzzle_id")
    if puzzle_ids is not None:
        puzzle_ids = puzzle_ids.to(accelerator.device)

    bsz = inputs.shape[0]
    input_emb = model.embed(inputs, puzzle_ids=puzzle_ids)
    z_H, z_L  = model.get_initial_states(bsz)
    z_H = z_H.to(accelerator.device)
    z_L = z_L.to(accelerator.device)

    total_loss_val = 0.0
    for _ in range(model.n_sup):
        logits, z_H, z_L = model.reasoning_step(input_emb, z_H, z_L)
        step_loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
        accelerator.backward(step_loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss_val += step_loss.item()

    return total_loss_val / model.n_sup


@torch.no_grad()
def eval_loop(model: SudokuTRM, dataloader: DataLoader, accelerator: Accelerator):
    model.eval()
    all_metrics = {"loss": [], "cell_acc": [], "puzzle_acc": []}

    for batch in dataloader:
        inputs  = batch["inputs"].to(accelerator.device)
        labels  = batch["labels"].to(accelerator.device)
        puzzle_ids = batch.get("puzzle_id")
        if puzzle_ids is not None:
            puzzle_ids = puzzle_ids.to(accelerator.device)

        logits = model.predict(inputs, puzzle_ids=puzzle_ids)
        m = compute_metrics(logits, labels, inputs)
        for k, v in m.items():
            all_metrics[k].append(v.item() if torch.is_tensor(v) else v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs/sudoku", config_name="config")
def main(args: DictConfig):
    wandb_project = args.get("wandb_project", None)
    log_with = ["wandb"] if wandb_project else []
    accelerator = Accelerator(
        mixed_precision=args.get("mixed_precision", "no"),
        log_with=log_with,
    )
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(args))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if wandb_project:
        run_name = args.get("wandb_run_name", None)
        init_kwargs = {"wandb": {"name": run_name}} if run_name else {}
        accelerator.init_trackers(
            project_name=wandb_project,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs=init_kwargs,
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dir = os.path.join(args.data_dir, "train")
    test_dir  = os.path.join(args.data_dir, "test")

    train_ds = SudokuDataset(train_dir, mask_given=True)
    eval_ds  = SudokuDataset(test_dir,  mask_given=True) if os.path.isdir(test_dir) else None

    # Fall back: split off 10% of train if no separate test set
    if eval_ds is None:
        n_val  = max(1, int(0.1 * len(train_ds)))
        n_tr   = len(train_ds) - n_val
        train_ds, eval_ds = random_split(train_ds, [n_tr, n_val])

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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SudokuTRM(
        vocab_size     = args.model.get("vocab_size", 11),
        seq_len        = args.model.get("seq_len", 81),
        d_model        = args.model.d_model,
        n_heads        = args.model.n_heads,
        n_layers       = args.model.n_layers,
        L_cycles       = args.model.L_cycles,
        H_cycles       = args.model.H_cycles,
        n_sup          = args.model.n_sup,
        dropout        = args.model.get("dropout", 0.0),
        num_puzzle_ids = args.model.get("num_puzzle_ids", None),
    )

    if accelerator.is_main_process:
        logger.info(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.train.lr,
        betas=(0.9, 0.95),
        weight_decay=args.train.get("weight_decay", 0.1),
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
    num_steps     = args.train.num_steps
    warmup_steps  = args.train.get("warmup_steps", 2000)
    eval_every    = args.train.get("eval_every", 1000)
    save_every    = args.train.get("save_every", 5000)
    log_every     = args.train.get("log_every", 100)

    best_puzzle_acc = 0.0
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

        loss_val = train_step(accelerator.unwrap_model(model), batch, accelerator, optimizer)

        global_step += 1
        progress_bar.update(1)

        if global_step % log_every == 0 and accelerator.is_main_process:
            logger.info(f"step={global_step}  loss={loss_val:.4f}  lr={lr:.2e}")
            if wandb_project:
                accelerator.log({"train/loss": loss_val, "train/lr": lr}, step=global_step)

        # ── Evaluation ────────────────────────────────────────────────────────
        if global_step % eval_every == 0:
            metrics = eval_loop(accelerator.unwrap_model(model), eval_dl, accelerator)
            if accelerator.is_main_process:
                logger.info(
                    f"[eval] step={global_step} "
                    f"loss={metrics['loss']:.4f}  "
                    f"cell_acc={metrics['cell_acc']*100:.2f}%  "
                    f"puzzle_acc={metrics['puzzle_acc']*100:.2f}%"
                )
                if wandb_project:
                    accelerator.log({
                        "eval/loss":       metrics["loss"],
                        "eval/cell_acc":   metrics["cell_acc"],
                        "eval/puzzle_acc": metrics["puzzle_acc"],
                    }, step=global_step)
                if metrics["puzzle_acc"] > best_puzzle_acc:
                    best_puzzle_acc = metrics["puzzle_acc"]
                    _save(accelerator, model, optimizer, global_step, args.output_dir, "best")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % save_every == 0 and accelerator.is_main_process:
            _save(accelerator, model, optimizer, global_step, args.output_dir, f"step-{global_step}")

    # Final save
    if accelerator.is_main_process:
        _save(accelerator, model, optimizer, global_step, args.output_dir, "final")
        logger.info(f"Training complete. Best puzzle_acc: {best_puzzle_acc*100:.2f}%")

    if wandb_project:
        accelerator.end_training()


def _save(accelerator, model, optimizer, step, output_dir, tag):
    ckpt = {
        "step":            step,
        "model_state":     accelerator.unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


if __name__ == "__main__":
    main()
