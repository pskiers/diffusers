"""
train_trm.py — Unified training script for OriginalTRMSudoku and OriginalTRMRatatouilleV0Tok.

Usage:
    # Standalone sudoku training (reproduces original TRM paper):
    python train_trm.py experiment=sudoku

    # Painter-thinker (V0Tok) training:
    python train_trm.py experiment=v0tok

    # Multi-GPU:
    accelerate launch --num_processes=2 train_trm.py experiment=sudoku

Config: configs/trm/config.yaml  (select mode via experiment= override)

Design notes:
  - LR schedule and EMA copied verbatim from TinyRecursiveModels/pretrain.py.
  - Optimizer: AdamATan2 (same as original) + optional SignSGD for puzzle embeddings.
  - Loss scaled by 1/global_batch_size to match original pretrain.py.
  - No accelerator.autocast() — TRM handles its own dtype via forward_dtype/CastedEmbedding.
  - Painter eval: ported from train_mnist_sudoku.py (compute_losses + eval_loop).
"""

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
from adam_atan2 import AdamATan2
from diffusers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from sudoku_dataset import SudokuDataset, IGNORE_LABEL_ID
from mnist_sudoku_dataset import MNISTSudokuDataset
from mnist_eval import (
    evaluate_grids, load_or_train_classifier, sample_grids,
    make_panel_image, plot_thinker_ts_curve,
)
from trm_wrappers import (
    OriginalTRMSudoku,
    OriginalTRMRatatouilleV0Tok,
    OriginalTRMRatatouilleV0,
    OriginalTRMRatatouilleV1,
    OriginalTRMRatatouilleV2,
    OriginalTRMRatatouilleV3,
    OriginalTRMRatatouilleV4,
    build_puzzle_emb_optimizer,
    get_non_puzzle_emb_params,
)
from models.ema import EMAHelper

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


logger = get_logger(__name__, log_level="INFO")


# ── LR schedule — copied verbatim from TinyRecursiveModels/pretrain.py ─────────

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(step: int, base_lr: float, warmup_steps: int, num_steps: int, min_ratio: float) -> float:
    return cosine_schedule_with_warmup_lr_lambda(
        step, base_lr=base_lr, num_warmup_steps=warmup_steps,
        num_training_steps=num_steps, min_ratio=min_ratio,
    )


# ── Optimizer building — adapted from TinyRecursiveModels/pretrain.py ──────────

def build_optimizers(model, cfg, world_size: int):
    """
    Returns (optimizers, base_lrs).

    Thinker uses AdamATan2 (same as original pretrain.py).
    Painter+bridge (if present) use a separate AdamW — different optimizer
    family since the UNet benefits from adaptive moments but not from
    AdamATan2's implicit normalisation.

    Thinker side mirrors pretrain.py create_model():
      - puzzle_emb_ndim=0: one AdamATan2 for thinker params.
      - freeze_weights: SignSGD for puzzle emb only.
      - Both enabled: SignSGD for puzzle emb + AdamATan2 for thinker rest.
    """
    t = cfg.thinker
    tr = cfg.train
    p_cfg = cfg.get("painter", None)

    is_painter_model = isinstance(model, OriginalTRMRatatouilleV0Tok)
    thinker_params = model.get_thinker_params() if is_painter_model else list(model.parameters())

    adamatan2_kwargs = dict(
        lr=0,  # set per-step by scheduler
        weight_decay=tr.weight_decay,
        betas=(tr.beta1, tr.beta2),
    )

    if t.puzzle_emb_ndim == 0:
        optimizers = [AdamATan2(thinker_params, **adamatan2_kwargs)]
        base_lrs = [tr.lr]
    elif t.freeze_weights:
        puzzle_opt = build_puzzle_emb_optimizer(
            model, world_size=world_size, lr=0,
            weight_decay=tr.puzzle_emb_weight_decay,
        )
        optimizers = [puzzle_opt]
        base_lrs = [tr.puzzle_emb_lr]
    else:
        puzzle_opt = build_puzzle_emb_optimizer(
            model, world_size=world_size, lr=0,
            weight_decay=tr.puzzle_emb_weight_decay,
        )
        # Exclude puzzle emb buffers from AdamATan2
        puzzle_ids = {id(p) for p in get_non_puzzle_emb_params(model)} ^ {id(p) for p in model.parameters()}
        adamatan2 = AdamATan2(
            [p for p in thinker_params if id(p) not in puzzle_ids],
            **adamatan2_kwargs,
        )
        optimizers = [puzzle_opt, adamatan2]
        base_lrs = [tr.puzzle_emb_lr, tr.lr]

    if is_painter_model:
        raw_lr = getattr(p_cfg, "lr", None) if p_cfg is not None else None
        raw_wd = getattr(p_cfg, "weight_decay", None) if p_cfg is not None else None
        painter_lr = tr.lr if raw_lr is None else raw_lr
        painter_wd = tr.weight_decay if raw_wd is None else raw_wd
        painter_opt = torch.optim.AdamW(model.get_painter_params(), lr=0, weight_decay=painter_wd)
        optimizers.append(painter_opt)
        base_lrs.append(painter_lr)

    return optimizers, base_lrs


def _apply_lr_and_step(optimizers, base_lrs, global_step, cfg):
    """Update LR for all optimizers, step them. Returns lr of last optimizer."""
    tr = cfg.train
    lr_now = None
    for opt, base_lr in zip(optimizers, base_lrs):
        lr_now = compute_lr(global_step, base_lr, tr.warmup_steps, tr.num_steps, tr.lr_min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.step()
    return lr_now


def _zero_grads(optimizers):
    for opt in optimizers:
        opt.zero_grad()


# ── Sudoku mode ────────────────────────────────────────────────────────────────

def sudoku_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """logits: (B,81,V), labels: (B,81) with IGNORE_LABEL_ID on given cells."""
    preds = logits.argmax(-1)
    blank = labels != IGNORE_LABEL_ID
    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        labels.view(-1).clamp(min=0),
        ignore_index=IGNORE_LABEL_ID,
        reduction="mean",
    )
    cell_acc = (preds == labels)[blank].float().mean()
    correct = (preds == labels) & blank
    puzzle_acc = (correct.sum(-1) == blank.sum(-1)).float().mean()
    return {"sudoku_loss": loss, "cell_acc": cell_acc, "puzzle_acc": puzzle_acc}


def train_step_sudoku(model, micro_batches, accelerator, optimizers, base_lrs, global_step, cfg, ema_helper, global_batch_size):
    K = len(micro_batches)
    device = accelerator.device

    # Each micro-batch has its own independent carry state
    mb_data = []
    for mb in micro_batches:
        bsz = mb["inputs"].shape[0]
        z_H, z_L = model.get_initial_states(bsz)
        puzzle_ids = mb.get("puzzle_id")
        mb_data.append({
            "inputs":     mb["inputs"].to(device),
            "labels":     mb["labels"].to(device),
            "puzzle_ids": puzzle_ids.to(device) if puzzle_ids is not None else None,
            "z_H": z_H.to(device),
            "z_L": z_L.to(device),
        })

    total_loss = 0.0
    lr = None
    for _ in range(model.n_sup):
        for d in mb_data:
            logits, d["z_H"], d["z_L"] = model.reasoning_step(
                d["inputs"], d["z_H"], d["z_L"], d["puzzle_ids"]
            )
            step_loss = F.cross_entropy(
                logits.float().view(-1, model.vocab_size),
                d["labels"].view(-1).clamp(min=0),
                ignore_index=IGNORE_LABEL_ID,
            )
            accelerator.backward(step_loss / (global_batch_size * K))
            total_loss += step_loss.item()

        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        lr = _apply_lr_and_step(optimizers, base_lrs, global_step, cfg)
        _zero_grads(optimizers)
        if ema_helper is not None:
            ema_helper.update(model)
        global_step += 1

    return total_loss / (model.n_sup * K), lr, global_step


@torch.no_grad()
def eval_sudoku(model, dataloader, accelerator, max_batches=10):
    model.eval()
    accum = {"sudoku_loss": [], "cell_acc": [], "puzzle_acc": []}
    for i, batch in tqdm(enumerate(dataloader), desc="Evaluating", total=max_batches):
        if i >= max_batches:
            break
        inputs = batch["inputs"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)
        puzzle_ids = batch.get("puzzle_id")
        if puzzle_ids is not None:
            puzzle_ids = puzzle_ids.to(accelerator.device)
        logits = model.predict(inputs, puzzle_ids=puzzle_ids)
        m = sudoku_metrics(logits.float(), labels)
        for k, v in m.items():
            accum[k].append(v.item())
    return {k: float(np.mean(v)) for k, v in accum.items()}


# ── Painter mode ───────────────────────────────────────────────────────────────
# compute_losses and eval_loop ported from train_mnist_sudoku.py.
# Key difference: thinker vocab=11, so labels are solution+2 (not raw 0-8).

def _make_tok_labels(solution: torch.Tensor) -> torch.Tensor:
    """
    solution: (B,81) int64 — 0-8 for blank cells, IGNORE_LABEL_ID (-100) for given cells.
    Returns token-format labels: 2-10 for blank cells, IGNORE_LABEL_ID for given cells.
    """
    labels = solution.clone()
    valid = labels != IGNORE_LABEL_ID
    labels[valid] = labels[valid] + 2
    return labels


def compute_losses_painter(model, batch, scheduler, accelerator, sudoku_loss_weight) -> dict:
    """Single forward pass (eval). Ported from train_mnist_sudoku.py compute_losses."""
    device = accelerator.device
    images     = batch["images"].to(device)
    condition  = _get_condition(batch, model, device)
    solution   = batch["solution"].to(device)    # (B,81) 0-8 or IGNORE for given
    given_mask = batch.get("given_mask")
    puzzle_ids = batch["puzzle_id"].to(device) if "puzzle_id" in batch else None

    B = images.shape[0]
    noise     = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,),
                              device=device, dtype=torch.long)
    noisy = scheduler.add_noise(images, noise, timesteps)

    noise_pred, sudoku_logits = model(noisy, timesteps, condition, puzzle_ids=puzzle_ids)

    target    = noise if scheduler.config.prediction_type == "epsilon" else images
    diff_loss = F.mse_loss(noise_pred.float(), target)

    sudoku_loss = torch.tensor(0.0, device=device)
    ce_labels   = _make_ce_labels(solution, model)
    if sudoku_logits is not None and sudoku_loss_weight > 0:
        B_, N, C = sudoku_logits.shape
        sudoku_loss = F.cross_entropy(
            sudoku_logits.float().reshape(B_ * N, C),
            ce_labels[:, :N].reshape(B_ * N).clamp(min=0),
            ignore_index=IGNORE_LABEL_ID,
        )

    total_loss = diff_loss + sudoku_loss_weight * sudoku_loss

    thinker_cell_acc   = None
    thinker_puzzle_acc = None
    if sudoku_logits is not None:
        B_, N, C = sudoku_logits.shape
        preds   = sudoku_logits.argmax(dim=-1)
        targets = ce_labels[:B_, :N]
        correct = preds == targets               # (B_, N)

        thinker_puzzle_acc = correct.all(dim=1).float().mean()

        if given_mask is not None:
            blank = ~given_mask.to(device)[:B_, :N]
            n_blank = blank.sum()
            thinker_cell_acc = (correct[blank].float().mean()
                                if n_blank > 0 else correct.float().mean())
        else:
            thinker_cell_acc = correct.float().mean()

    return {
        "loss":               total_loss,
        "diff_loss":          diff_loss,
        "sudoku_loss":        sudoku_loss,
        "thinker_cell_acc":   thinker_cell_acc,
        "thinker_puzzle_acc": thinker_puzzle_acc,
    }


@torch.no_grad()
def eval_painter(model, dataloader, scheduler, accelerator, sudoku_loss_weight, max_batches=10):
    model.eval()
    metrics: dict[str, list] = {
        "loss": [], "diff_loss": [], "sudoku_loss": [],
        "thinker_cell_acc": [], "thinker_puzzle_acc": [],
    }
    for i, batch in tqdm(enumerate(dataloader), "Evaluating", total=max_batches):
        if i >= max_batches:
            break
        m = compute_losses_painter(model, batch, scheduler, accelerator, sudoku_loss_weight)
        for k in metrics:
            val = m.get(k)
            if val is not None:
                metrics[k].append(val.item() if torch.is_tensor(val) else float(val))
    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items() if v}


def train_step_painter(model, micro_batches, scheduler, accelerator, optimizers, base_lrs, global_step, cfg, ema_helper, global_batch_size):
    K = len(micro_batches)
    device = accelerator.device
    sudoku_w = cfg.train.sudoku_loss_weight

    # Pre-process: sample noise, build carry state for each micro-batch
    mb_data = []
    for mb in micro_batches:
        images     = mb["images"].to(device)
        condition  = _get_condition(mb, model, device)
        solution   = mb["solution"].to(device)
        puzzle_ids = mb["puzzle_id"].to(device) if "puzzle_id" in mb else None

        bsz   = images.shape[0]
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,),
                                  device=device, dtype=torch.long)
        noisy  = scheduler.add_noise(images, noise, timesteps)
        target = noise if scheduler.config.prediction_type == "epsilon" else images
        z_H, z_L = model.get_initial_states(bsz)

        mb_data.append({
            "condition":  condition,
            "ce_labels":  _make_ce_labels(solution, model),
            "puzzle_ids": puzzle_ids,
            "noisy":      noisy,
            "timesteps":  timesteps,
            "target":     target,
            "z_H": z_H.to(device),
            "z_L": z_L.to(device),
        })

    total_diff_loss   = 0.0
    total_sudoku_loss = 0.0
    lr = None

    for _ in range(model.n_sup):
        for d in mb_data:
            noise_pred, logits, d["z_H"], d["z_L"] = model.reasoning_step(
                d["condition"], d["noisy"], d["z_H"], d["z_L"],
                d["timesteps"], d["puzzle_ids"],
            )
            diff_loss = F.mse_loss(noise_pred.float(), d["target"])

            sudoku_loss = torch.tensor(0.0, device=device)
            if logits is not None and sudoku_w > 0:
                B_, N, C = logits.shape
                sudoku_loss = F.cross_entropy(
                    logits.float().reshape(B_ * N, C),
                    d["ce_labels"][:, :N].reshape(B_ * N).clamp(min=0),
                    ignore_index=IGNORE_LABEL_ID,
                )

            step_loss = diff_loss + sudoku_w * sudoku_loss
            accelerator.backward(step_loss / (global_batch_size * K))
            total_diff_loss   += diff_loss.item()
            total_sudoku_loss += sudoku_loss.item()

        # Clip thinker and painter separately, matching train_mnist_sudoku.py
        accelerator.clip_grad_norm_(model.get_thinker_params(), 1.0)
        accelerator.clip_grad_norm_(model.get_painter_params(), 1.0)
        lr = _apply_lr_and_step(optimizers, base_lrs, global_step, cfg)
        _zero_grads(optimizers)
        if ema_helper is not None:
            ema_helper.update(model)
        global_step += 1

    n = model.n_sup * K
    return {"diff_loss": total_diff_loss / n, "sudoku_loss": total_sudoku_loss / n}, lr, global_step


def _get_condition(mb: dict, model, device) -> torch.Tensor:
    """Return the condition tensor for this batch based on model.token_input."""
    if model.token_input:
        return mb["puzzle_tokens"].to(device)
    return mb["conditions"].to(device)


def _get_teacher_condition(mb: dict, model, device) -> torch.Tensor:
    """Teacher-forcing condition: full solution tokens for token models,
    complete rendered MNIST image for image-conditioned models."""
    if model.token_input:
        return _solution_tokens(mb["solution"].to(device))
    return mb["images"].to(device)


def _make_ce_labels(solution: torch.Tensor, model) -> torch.Tensor:
    """Return CE supervision labels compatible with model output space.
    Token models: 2-10 (tok_labels).  Image models: 0-8 (raw solution).
    In both cases IGNORE_LABEL_ID marks given/masked cells."""
    if model.token_input:
        return _make_tok_labels(solution)
    return solution   # already 0-8 with IGNORE_LABEL_ID for given cells


def _solution_tokens(solution: torch.Tensor) -> torch.Tensor:
    """Raw solution (0-8) → full token grid (2-10, no blanks) for painter stage."""
    return solution.clamp(min=0) + 2


def train_step_painter_two_stage(
    model, micro_batches, scheduler, accelerator, optimizers, base_lrs,
    global_step, cfg, ema_helper, global_batch_size,
):
    """
    Interleaved two-stage painter training.

    Each tick of the outer n_sup loop does two sequential sub-steps:

      1. Painter sub-step (first ps_n_sup ticks only):
         Thinker receives the full solution tokens (teacher forcing) so it has
         a trivial task and the painter learns to reliably follow it.
         Reduced H_cycles / L_cycles are optional.  Updates all params.

      2. Thinker sub-step (every tick):
         Thinker receives the real puzzle tokens.  Painter is frozen; only CE
         loss is computed so diffusion cannot interfere with sudoku reasoning.
         Updates thinker params only.

    Both stages keep their own carry state (z_H / z_L) that persists across
    ticks, since they process different inputs.
    """
    K       = len(micro_batches)
    device  = accelerator.device
    sudoku_w = cfg.train.sudoku_loss_weight

    ps = cfg.get("painter_stage", {})
    ts = cfg.get("thinker_stage", {})
    ps_n_sup  = int(ps.get("n_sup",    model.n_sup))
    ps_H      = ps.get("H_cycles", None)
    ps_L      = ps.get("L_cycles", None)
    ts_n_sup  = int(ts.get("n_sup",    model.n_sup))

    # Painter AdamW is always the last optimizer (built by build_optimizers).
    thinker_opts     = optimizers[:-1]
    thinker_base_lrs = base_lrs[:-1]

    # Pre-process micro-batches — reused across both stages each tick.
    mb_data = []
    for mb in micro_batches:
        images     = mb["images"].to(device)
        condition  = _get_condition(mb, model, device)
        teacher    = _get_teacher_condition(mb, model, device)
        solution   = mb["solution"].to(device)
        puzzle_ids = mb["puzzle_id"].to(device) if "puzzle_id" in mb else None

        bsz   = images.shape[0]
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,),
                                  device=device, dtype=torch.long)
        noisy  = scheduler.add_noise(images, noise, timesteps)
        target = noise if scheduler.config.prediction_type == "epsilon" else images

        z_H_p, z_L_p = model.get_initial_states(bsz)   # painter-stage carry
        z_H_t, z_L_t = model.get_initial_states(bsz)   # thinker-stage carry

        mb_data.append({
            "condition":  condition,
            "teacher":    teacher,
            "ce_labels":  _make_ce_labels(solution, model),
            "puzzle_ids": puzzle_ids,
            "noisy":      noisy,
            "timesteps":  timesteps,
            "target":     target,
            "z_H_p": z_H_p.to(device), "z_L_p": z_L_p.to(device),
            "z_H_t": z_H_t.to(device), "z_L_t": z_L_t.to(device),
        })

    total_diff_loss   = 0.0
    total_sudoku_loss = 0.0
    lr = None
    n_ticks = max(ps_n_sup, ts_n_sup)

    painter_params = model.get_painter_params()

    for tick in range(n_ticks):
        # ── Painter sub-step (teacher forcing with solution tokens) ──────────
        if tick < ps_n_sup:
            for d in mb_data:
                noise_pred, logits, d["z_H_p"], d["z_L_p"] = model.reasoning_step(
                    d["teacher"], d["noisy"], d["z_H_p"], d["z_L_p"], d["timesteps"],
                    d["puzzle_ids"], H_cycles=ps_H, L_cycles=ps_L,
                )
                diff_loss = F.mse_loss(noise_pred.float(), d["target"])
                sudoku_loss = torch.tensor(0.0, device=device)
                if logits is not None and sudoku_w > 0:
                    B_, N, C = logits.shape
                    sudoku_loss = F.cross_entropy(
                        logits.float().reshape(B_ * N, C),
                        d["ce_labels"][:, :N].reshape(B_ * N).clamp(min=0),
                        ignore_index=IGNORE_LABEL_ID,
                    )
                step_loss = diff_loss + sudoku_w * sudoku_loss
                accelerator.backward(step_loss / (global_batch_size * K))
                total_diff_loss   += diff_loss.item()
                total_sudoku_loss += sudoku_loss.item()

            accelerator.clip_grad_norm_(model.get_thinker_params(), 1.0)
            accelerator.clip_grad_norm_(painter_params, 1.0)
            lr = _apply_lr_and_step(optimizers, base_lrs, global_step, cfg)
            _zero_grads(optimizers)

        # ── Thinker sub-step (real condition, painter frozen) ───────────────
        # Full forward pass with painter frozen: diff gradient flows back
        # through the bridge into the thinker; painter weights don't update.
        if tick < ts_n_sup:
            for p in painter_params:
                p.requires_grad_(False)

            for d in mb_data:
                noise_pred, logits, d["z_H_t"], d["z_L_t"] = model.reasoning_step(
                    d["condition"], d["noisy"], d["z_H_t"], d["z_L_t"], d["timesteps"],
                    d["puzzle_ids"],
                )
                diff_loss = F.mse_loss(noise_pred.float(), d["target"])
                sudoku_loss = torch.tensor(0.0, device=device)
                if logits is not None and sudoku_w > 0:
                    B_, N, C = logits.shape
                    sudoku_loss = F.cross_entropy(
                        logits.float().reshape(B_ * N, C),
                        d["ce_labels"][:, :N].reshape(B_ * N).clamp(min=0),
                        ignore_index=IGNORE_LABEL_ID,
                    )
                step_loss = diff_loss + sudoku_w * sudoku_loss
                accelerator.backward(step_loss / (global_batch_size * K))
                total_sudoku_loss += sudoku_loss.item()

            accelerator.clip_grad_norm_(model.get_thinker_params(), 1.0)
            lr_t = _apply_lr_and_step(thinker_opts, thinker_base_lrs, global_step, cfg)
            lr = lr_t if lr is None else lr
            for opt in thinker_opts:
                opt.zero_grad()

            for p in painter_params:
                p.requires_grad_(True)

        if ema_helper is not None:
            ema_helper.update(model)
        global_step += 1

    n_p = ps_n_sup * K or 1
    n_t = ts_n_sup * K or 1
    return {
        "diff_loss":   total_diff_loss   / n_p,
        "sudoku_loss": total_sudoku_loss / (n_p + n_t),
    }, lr, global_step


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizers, step, output_dir, tag, ema_helper=None):
    ckpt = {
        "step":             step,
        "model_state":      accelerator.unwrap_model(model).state_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "ema_state":        ema_helper.state_dict() if ema_helper is not None else None,
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs/trm", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.mode   # "sudoku" | "painter"

    wandb_project = cfg.get("wandb_project", None)
    log_with = ["wandb"] if wandb_project else []

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "no"),
        log_with=log_with,
    )
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if wandb_project and accelerator.is_main_process:
        run_name = Path(cfg.output_dir).name
        init_kwargs = {"wandb": {"name": run_name}}
        accelerator.init_trackers(
            project_name=wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs,
        )

    torch.manual_seed(cfg.seed + accelerator.process_index)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if mode == "sudoku":
        train_dir = os.path.join(cfg.data.sudoku_dir, "train")
        test_dir  = os.path.join(cfg.data.sudoku_dir, "test")
        train_ds = SudokuDataset(train_dir, mask_given=True)
        eval_ds  = SudokuDataset(test_dir,  mask_given=True) if os.path.isdir(test_dir) else None
        if eval_ds is None:
            n_val = max(1, int(0.1 * len(train_ds)))
            train_ds, eval_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])
        scheduler = None
    else:
        cell_size    = cfg.data.cell_size
        painter_size = 9 * cell_size
        train_ds = MNISTSudokuDataset(
            sudoku_dir=os.path.join(cfg.data.sudoku_dir, "train"),
            mnist_root=cfg.data.mnist_root,
            cell_size=cell_size,
            mnist_split="train",
            mask_given=True,
        )
        test_dir = os.path.join(cfg.data.sudoku_dir, "test")
        eval_ds = MNISTSudokuDataset(
            sudoku_dir=test_dir if os.path.isdir(test_dir) else os.path.join(cfg.data.sudoku_dir, "train"),
            mnist_root=cfg.data.mnist_root,
            cell_size=cell_size,
            mnist_split="test",
            mask_given=True,
        )
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
        )

    n_workers = cfg.get("num_workers", 4)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )

    # ── MNIST cell classifier (painter mode, opt-in) ──────────────────────────
    classifier = None
    if mode == "painter" and accelerator.is_main_process:
        classifier_path = cfg.train.get("eval_classifier_path", None)
        if classifier_path:
            classifier = load_or_train_classifier(
                classifier_path,
                cfg.data.mnist_root,
                cell_size,
                accelerator.device,
            )

    # ── Model ─────────────────────────────────────────────────────────────────
    t = cfg.thinker
    thinker_kwargs = dict(
        vocab_size=t.vocab_size,
        seq_len=t.seq_len,
        hidden_size=t.hidden_size,
        n_heads=t.n_heads,
        L_layers=t.L_layers,
        L_cycles=t.L_cycles,
        H_cycles=t.H_cycles,
        n_sup=t.n_sup,
        expansion=t.expansion,
        forward_dtype=t.forward_dtype,
        mlp_t=t.mlp_t,
        pos_encodings=t.pos_encodings,
        puzzle_emb_ndim=t.puzzle_emb_ndim,
        puzzle_emb_len=t.puzzle_emb_len,
        num_puzzle_identifiers=t.num_puzzle_identifiers,
        halt_exploration_prob=t.halt_exploration_prob,
        batch_size=cfg.train.batch_size,
        freeze_weights=t.freeze_weights,
    )

    painter_variant = cfg.get("painter_variant", "v0tok")

    if mode == "sudoku":
        model = OriginalTRMSudoku(**thinker_kwargs)
    elif painter_variant == "v0tok":
        p = cfg.painter
        model = OriginalTRMRatatouilleV0Tok(
            painter_size=painter_size,
            cell_size=cell_size,
            bridge_channels=p.bridge_channels,
            painter_channels=tuple(p.painter_channels),
            painter_layers_per_block=p.painter_layers_per_block,
            diff_thinker_weight=p.diff_thinker_weight,
            painter_dtype=p.get("dtype", None),
            **thinker_kwargs,
        )
    else:
        # Image-conditioned variants: V0, V1, V2
        # These share a common subset of thinker kwargs (no puzzle embeddings).
        p = cfg.painter
        img_thinker_kwargs = dict(
            seq_len=t.seq_len,
            hidden_size=t.hidden_size,
            n_heads=t.n_heads,
            L_layers=t.L_layers,
            L_cycles=t.L_cycles,
            H_cycles=t.H_cycles,
            n_sup=t.n_sup,
            expansion=t.expansion,
            forward_dtype=t.forward_dtype,
            mlp_t=t.mlp_t,
            pos_encodings=t.pos_encodings,
            halt_exploration_prob=t.halt_exploration_prob,
            batch_size=cfg.train.batch_size,
            freeze_weights=t.freeze_weights,
        )
        img_painter_kwargs = dict(
            painter_size=painter_size,
            cell_size=cell_size,
            enc_channels=p.get("enc_channels", 32),
            bridge_channels=p.bridge_channels,
            painter_channels=tuple(p.painter_channels),
            painter_layers_per_block=p.painter_layers_per_block,
            diff_thinker_weight=p.diff_thinker_weight,
            painter_dtype=p.get("dtype", None),
        )
        _VARIANT_CLS = {
            "v0": OriginalTRMRatatouilleV0,
            "v1": OriginalTRMRatatouilleV1,
            "v2": OriginalTRMRatatouilleV2,
            "v3": OriginalTRMRatatouilleV3,
            "v4": OriginalTRMRatatouilleV4,
        }
        cls = _VARIANT_CLS.get(painter_variant)
        if cls is None:
            raise ValueError(f"Unknown painter_variant: {painter_variant!r}. "
                             f"Choose from: v0tok, v0, v1, v2, v3, v4")
        if painter_variant in ("v2", "v3", "v4"):
            extra = dict(thinker_out_channels=p.get("thinker_out_channels", 16))
            thinker_kw = img_thinker_kwargs
            if painter_variant == "v4":
                extra["compression_factor"] = p.get("compression_factor", cell_size)
                extra["bridge_num_heads"]   = p.get("bridge_num_heads", 4)
                # V4 computes seq_len internally from compression_factor
                thinker_kw = {k: v for k, v in img_thinker_kwargs.items() if k != "seq_len"}
            model = cls(**extra, **img_painter_kwargs, **thinker_kw)
        else:
            model = cls(
                num_classes=t.get("num_classes", 9),
                **img_painter_kwargs, **img_thinker_kwargs,
            )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    # ── torch.compile (opt-in) ────────────────────────────────────────────────
    # Compile submodules individually to avoid tracing the Python-level n_sup /
    # H_cycles / L_cycles loops (dynamic control flow defeats fullgraph compile).
    # L_level is the hot path: called H_cycles*L_cycles times per sup step.
    if cfg.get("compile", False):
        if mode == "sudoku":
            model.inner.L_level = torch.compile(model.inner.L_level, fullgraph=False)
        else:
            model.thinker.inner.L_level = torch.compile(model.thinker.inner.L_level, fullgraph=False)
            model.painter = torch.compile(model.painter)
            if model.bridge is not None:
                model.bridge = torch.compile(model.bridge)
        if accelerator.is_main_process:
            logger.info("torch.compile applied")

    # ── Optimizers (same structure as pretrain.py create_model) ───────────────
    world_size = accelerator.num_processes
    global_batch_size = cfg.train.batch_size * world_size
    optimizers, base_lrs = build_optimizers(model, cfg, world_size)

    # Don't prepare optimizers — accelerate wraps them in AcceleratedOptimizer
    # which passes closure to step(), but AdamATan2/SignSGD don't accept it.
    # We step all optimizers manually so there's no need for the wrapper.
    model, train_dl, eval_dl = accelerator.prepare(model, train_dl, eval_dl)

    # ── EMA — using EMAHelper from models/ema.py, same as pretrain.py ─────────
    ema_helper = None
    if cfg.use_ema:
        ema_helper = EMAHelper(mu=cfg.ema_rate)
        ema_helper.register(accelerator.unwrap_model(model))
        logger.info(f"EMA enabled (mu={cfg.ema_rate})")

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    resume_path = cfg.get("resume_from_checkpoint", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        for opt, sd in zip(optimizers, ckpt["optimizer_states"]):
            opt.load_state_dict(sd)
        global_step = int(ckpt["step"])
        if ema_helper is not None and ckpt.get("ema_state") is not None:
            ema_helper.load_state_dict(ckpt["ema_state"])
        logger.info(f"Resumed from {resume_path} at step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    num_steps        = cfg.train.num_steps
    eval_every       = cfg.train.get("eval_every", 1000)
    save_every       = cfg.train.get("save_every", 5000)
    log_every        = cfg.train.get("log_every", 100)
    sudoku_w         = cfg.train.get("sudoku_loss_weight", 1.0)
    grad_accum_steps = cfg.train.get("gradient_accumulation_steps", 1)

    next_log  = log_every
    next_eval = eval_every
    next_save = save_every

    train_iter   = iter(train_dl)
    unwrapped    = accelerator.unwrap_model(model)
    progress_bar = tqdm(
        total=num_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            return next(train_iter)

    while global_step < num_steps:
        unwrapped.train()
        micro_batches = [_next_batch() for _ in range(grad_accum_steps)]

        if mode == "sudoku":
            loss_val, lr, global_step = train_step_sudoku(
                unwrapped, micro_batches, accelerator, optimizers, base_lrs,
                global_step, cfg, ema_helper, global_batch_size,
            )
            log_dict = {"train/loss": loss_val, "train/lr": lr}
        else:
            _painter_step_fn = (
                train_step_painter_two_stage
                if cfg.get("painter_stage", None) is not None
                else train_step_painter
            )
            losses, lr, global_step = _painter_step_fn(
                unwrapped, micro_batches, scheduler, accelerator, optimizers, base_lrs,
                global_step, cfg, ema_helper, global_batch_size,
            )
            log_dict = {f"train/{k}": v for k, v in losses.items()}
            log_dict["train/lr"] = lr

        progress_bar.update(unwrapped.n_sup)

        if global_step >= next_log and accelerator.is_main_process:
            logger.info(f"step={global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
            if wandb_project:
                accelerator.log(log_dict, step=global_step)
            next_log = global_step + log_every

        if global_step >= next_eval:
            # Swap in EMA weights for eval, then restore live weights.
            # Avoids deepcopy (which fails on non-leaf tensors like local_weights).
            if ema_helper is not None:
                live_params = [p.data.clone() for p in unwrapped.parameters() if p.requires_grad]
                ema_helper.ema(unwrapped)
            unwrapped.eval()

            if mode == "sudoku":
                metrics = eval_sudoku(unwrapped, eval_dl, accelerator, max_batches=100)
                val_log = {f"val/{k}": v for k, v in metrics.items()}
            else:
                metrics = eval_painter(unwrapped, eval_dl, scheduler, accelerator, sudoku_w, max_batches=100)
                val_log = {
                    "val/loss":        metrics.get("loss", 0),
                    "val/diff_loss":   metrics.get("diff_loss", 0),
                    "val/sudoku_loss": metrics.get("sudoku_loss", 0),
                }
                if "thinker_cell_acc" in metrics:
                    val_log["val/thinker_cell_acc"]   = metrics["thinker_cell_acc"]
                    val_log["val/thinker_puzzle_acc"] = metrics["thinker_puzzle_acc"]

            if ema_helper is not None:
                for p, live in zip((p for p in unwrapped.parameters() if p.requires_grad), live_params):
                    p.data.copy_(live)
            unwrapped.train()

            if accelerator.is_main_process:
                logger.info(f"[val] step={global_step}  " +
                            "  ".join(f"{k}={v:.4f}" for k, v in val_log.items()))
                if wandb_project:
                    accelerator.log(val_log, step=global_step)

            # ── Digit-level sampling eval (painter mode, main process only) ──
            if mode == "painter" and accelerator.is_main_process and classifier is not None:
                n_total = cfg.train.get("eval_num_samples",   128)
                n_batch = cfg.train.get("eval_batch_size",     32)
                n_ddim  = cfg.train.get("eval_num_ddim_steps", 20)
                n_log   = cfg.train.get("eval_num_log_images", 10)

                all_cell_acc:   list[float] = []
                all_puzzle_acc: list[float] = []
                ts_cell_accs:   dict[int, list[float]] = {}
                ts_puzzle_accs: dict[int, list[float]] = {}
                panels_list:    list = []

                all_thinker_cell_best:   list[float] = []
                all_thinker_cell_mean:   list[float] = []
                all_thinker_puzzle_best: list[float] = []
                all_thinker_puzzle_mean: list[float] = []
                all_thinker_deviation:   list[float] = []
                all_painter_dev_best:    list[float] = []
                all_painter_dev_mean:    list[float] = []

                n_done       = 0
                n_panels_done = 0

                # Swap in EMA for sampling
                if ema_helper is not None:
                    live_params = [p.data.clone() for p in unwrapped.parameters() if p.requires_grad]
                    ema_helper.ema(unwrapped)
                unwrapped.eval()

                sample_loader = DataLoader(eval_ds, batch_size=n_batch, shuffle=False)
                for eb in tqdm(sample_loader, "Sampling for eval",
                               total=(n_total + n_batch - 1) // n_batch):
                    if n_done >= n_total:
                        break
                    sols        = eb["solution"]
                    pids        = eb.get("puzzle_id", None)
                    given_masks = eb.get("given_mask", None)
                    B_cur       = sols.shape[0]
                    conds_sample = _get_condition(eb, unwrapped, "cpu")

                    sr = sample_grids(
                        unwrapped,
                        conds_sample,
                        num_train_timesteps=cfg.num_timesteps,
                        beta_schedule=cfg.beta_schedule,
                        prediction_type=cfg.prediction_type,
                        num_steps=n_ddim,
                        device=accelerator.device,
                        puzzle_ids=pids,
                        solutions=sols,
                        painter_size=painter_size,
                        given_masks=given_masks,
                    )
                    generated = sr["generated"]

                    acc = evaluate_grids(generated, sols, classifier, cell_size,
                                        given_masks=given_masks)
                    all_cell_acc.append(acc["cell_acc"])
                    all_puzzle_acc.append(acc["puzzle_acc"])

                    for t_step, a in sr.get("ts_cell_acc", []):
                        ts_cell_accs.setdefault(t_step, []).append(a)
                    for t_step, a in sr.get("ts_puzzle_acc", []):
                        ts_puzzle_accs.setdefault(t_step, []).append(a)

                    for key, lst in [
                        ("thinker_cell_acc_best",    all_thinker_cell_best),
                        ("thinker_cell_acc_mean",    all_thinker_cell_mean),
                        ("thinker_puzzle_acc_best",  all_thinker_puzzle_best),
                        ("thinker_puzzle_acc_mean",  all_thinker_puzzle_mean),
                        ("thinker_deviation_from_best", all_thinker_deviation),
                    ]:
                        if key in sr:
                            lst.append(sr[key])

                    painter_preds = acc["preds"]
                    best_tp = sr.get("best_thinker_preds")
                    mean_tp = sr.get("mean_thinker_preds")
                    _gm = given_masks[:B_cur] if given_masks is not None else None
                    for tp, dev_lst in [(best_tp, all_painter_dev_best), (mean_tp, all_painter_dev_mean)]:
                        if tp is not None:
                            N   = tp.shape[1]
                            diff = painter_preds[:, :N] != tp
                            if _gm is not None:
                                blank = ~_gm[:, :N]
                                n_b   = blank.sum()
                                dev   = diff[blank].float().mean().item() if n_b > 0 else diff.float().mean().item()
                            else:
                                dev = diff.float().mean().item()
                            dev_lst.append(dev)

                    if wandb_project and _wandb is not None and n_panels_done < n_log:
                        n_new  = min(n_log - n_panels_done, B_cur)
                        tp_all = sr.get("best_thinker_preds")
                        tt_all = sr.get("best_thinker_ts")
                        sols_np = sols.cpu().numpy()
                        conds_np = conds_sample.cpu()
                        for i in range(n_new):
                            tp = tp_all[i].numpy() if tp_all is not None else None
                            tt = tt_all[i]         if tt_all is not None else None
                            panel = make_panel_image(
                                conds_np[i], generated[i], sols_np[i],
                                thinker_preds=tp, thinker_t=tt,
                                img_size=324,
                            )
                            panels_list.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))
                        n_panels_done += n_new

                    n_done += B_cur

                # Restore live weights
                if ema_helper is not None:
                    for p, live in zip((p for p in unwrapped.parameters() if p.requires_grad), live_params):
                        p.data.copy_(live)
                unwrapped.train()

                mean_cell   = float(np.mean(all_cell_acc))
                mean_puzzle = float(np.mean(all_puzzle_acc))
                logger.info(
                    f"[eval] step={global_step}  cell_acc={mean_cell:.4f}  "
                    f"puzzle_acc={mean_puzzle:.4f}  (over {n_done} samples)"
                )
                if all_thinker_cell_best:
                    logger.info(
                        f"[eval] thinker_cell_best={np.mean(all_thinker_cell_best):.4f}  "
                        f"thinker_cell_mean={np.mean(all_thinker_cell_mean):.4f}  "
                        f"thinker_puzzle_best={np.mean(all_thinker_puzzle_best):.4f}  "
                        f"thinker_puzzle_mean={np.mean(all_thinker_puzzle_mean):.4f}  "
                        f"thinker_dev={np.mean(all_thinker_deviation):.4f}"
                    )
                if all_painter_dev_best:
                    logger.info(
                        f"[eval] painter_dev_best={np.mean(all_painter_dev_best):.4f}  "
                        f"painter_dev_mean={np.mean(all_painter_dev_mean):.4f}"
                    )
                if wandb_project:
                    wandb_acc: dict = {
                        "eval/cell_acc":   mean_cell,
                        "eval/puzzle_acc": mean_puzzle,
                    }
                    if all_thinker_cell_best:
                        wandb_acc["eval/thinker_cell_acc_best"]       = float(np.mean(all_thinker_cell_best))
                        wandb_acc["eval/thinker_cell_acc_mean"]       = float(np.mean(all_thinker_cell_mean))
                        wandb_acc["eval/thinker_puzzle_acc_best"]     = float(np.mean(all_thinker_puzzle_best))
                        wandb_acc["eval/thinker_puzzle_acc_mean"]     = float(np.mean(all_thinker_puzzle_mean))
                        wandb_acc["eval/thinker_deviation_from_best"] = float(np.mean(all_thinker_deviation))
                    if all_painter_dev_best:
                        wandb_acc["eval/painter_dev_from_best_thinker"] = float(np.mean(all_painter_dev_best))
                        wandb_acc["eval/painter_dev_from_mean_thinker"] = float(np.mean(all_painter_dev_mean))
                    if panels_list:
                        wandb_acc["eval/samples"] = panels_list
                    if ts_cell_accs:
                        curve = plot_thinker_ts_curve(ts_cell_accs, ts_puzzle_accs)
                        wandb_acc["eval/thinker_vs_timestep"] = _wandb.Image(curve)
                    accelerator.log(wandb_acc, step=global_step)

            next_eval = global_step + eval_every

        if global_step >= next_save and accelerator.is_main_process:
            save_checkpoint(accelerator, model, optimizers, global_step,
                            cfg.output_dir, f"step-{global_step}", ema_helper)
            next_save = global_step + save_every

    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizers, global_step,
                        cfg.output_dir, "final", ema_helper)
        logger.info("Training complete.")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
