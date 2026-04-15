"""
train_mnist_sudoku.py – Train a TRM-DiT Ratatouille MNIST-Sudoku model.

Usage:
    python train_mnist_sudoku.py experiment=v0
    python train_mnist_sudoku.py experiment=v1 train.sudoku_loss_weight=0.1
    accelerate launch --num_processes=2 train_mnist_sudoku.py experiment=v2

The experiment config selects the model variant and any overrides.

Training loop (TRM-aware):
  Each batch runs model.n_sup backward passes — one per supervision step,
  matching the SudokuTRM training convention (1 backprop = 1 step).
  LR is updated before every backprop so each sub-step gets its correct LR.
"""

import inspect
import os
import math
import logging
from pathlib import Path

import wandb
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

from mnist_eval import (
    evaluate_grids, load_or_train_classifier, sample_grids,
    make_panel_image, plot_thinker_ts_curve,
)
from mnist_sudoku_dataset import MNISTSudokuDataset
from mnist_sudoku_models import (
    MNISTRatatouilleV0,
    MNISTRatatouilleV1,
    MNISTRatatouilleV2,
    MNISTRatatouilleV3,
    MNISTRatatouilleV4,
)


logger = get_logger(__name__, log_level="INFO")


# ── EMA ────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.

    Shadow copies are stored in float32 to avoid precision loss during
    accumulation, regardless of the model's training dtype.
    """

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow_params = [p.clone().float().detach() for p in parameters]

    def to(self, device):
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        return self

    @torch.no_grad()
    def update(self, parameters):
        for shadow, param in zip(self.shadow_params, parameters):
            shadow.mul_(self.decay).add_(param.float().data, alpha=1 - self.decay)

    def copy_to(self, parameters):
        """Copy EMA shadow weights into model parameters (for eval / saving)."""
        for shadow, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow.to(param.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow_params": [p.cpu() for p in self.shadow_params]}

    def load_state_dict(self, state: dict, device=None):
        self.decay = state["decay"]
        self.shadow_params = [
            p.to(device) if device is not None else p
            for p in state["shadow_params"]
        ]


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


# ── Forward pass (eval only) ──────────────────────────────────────────────────

def compute_losses(
    model,
    batch: dict,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    sudoku_loss_weight: float = 0.0,
) -> dict:
    """
    Single forward pass using model.forward() (full inference).
    Used for eval — no backward.
    """
    images     = batch["images"].to(accelerator.device)      # (B,1,H,W)
    conditions = batch["conditions"].to(accelerator.device)  # (B,1,H,W)
    solution   = batch["solution"].to(accelerator.device)    # (B,81) long
    puzzle_ids = batch["puzzle_id"].to(accelerator.device) if "puzzle_id" in batch else None

    B = images.shape[0]
    noise     = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,),
                              device=accelerator.device, dtype=torch.long)
    noisy = scheduler.add_noise(images, noise, timesteps)

    noise_pred, sudoku_logits = model(noisy, timesteps, conditions, puzzle_ids=puzzle_ids)

    diff_loss = F.mse_loss(noise_pred, noise)

    sudoku_loss = torch.tensor(0.0, device=accelerator.device)
    if sudoku_logits is not None and sudoku_loss_weight > 0:
        B_, N, C = sudoku_logits.shape
        sudoku_loss = F.cross_entropy(
            sudoku_logits.reshape(B_ * N, C),
            solution[:, :N].reshape(B_ * N),
            ignore_index=IGNORE_LABEL_ID,
        )

    total_loss = diff_loss + sudoku_loss_weight * sudoku_loss

    thinker_cell_acc   = None
    thinker_puzzle_acc = None
    if sudoku_logits is not None:
        B_, N, C = sudoku_logits.shape
        preds   = sudoku_logits.argmax(dim=-1)   # (B_, N)
        targets = solution[:B_, :N]              # (B_, N)
        correct = preds == targets
        thinker_cell_acc   = correct.float().mean()
        thinker_puzzle_acc = correct.all(dim=1).float().mean()

    return {
        "loss":               total_loss,
        "diff_loss":          diff_loss,
        "sudoku_loss":        sudoku_loss,
        "thinker_cell_acc":   thinker_cell_acc,
        "thinker_puzzle_acc": thinker_puzzle_acc,
    }


# ── Training step (n_sup backward passes) ────────────────────────────────────

def train_step(
    model,
    batch: dict,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    warmup_steps: int,
    num_steps: int,
    base_lr: float,
    sudoku_loss_weight: float = 0.0,
    ema: EMA | None = None,
) -> tuple[dict, int]:
    """
    Run model.n_sup supervision steps, each with its own LR update, backward,
    and optimizer step.  global_step is incremented once per backprop.

    Returns (mean loss dict, new global_step).
    """
    images     = batch["images"].to(accelerator.device)
    conditions = batch["conditions"].to(accelerator.device)
    solution   = batch["solution"].to(accelerator.device)
    puzzle_ids = batch["puzzle_id"].to(accelerator.device) if "puzzle_id" in batch else None

    B = images.shape[0]
    noise     = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,),
                              device=accelerator.device, dtype=torch.long)
    noisy = scheduler.add_noise(images, noise, timesteps)

    z_H, z_L = model.get_initial_states(B)
    z_H = z_H.to(accelerator.device)
    z_L = z_L.to(accelerator.device)

    total_loss_val   = 0.0
    last_diff_loss   = 0.0
    last_sudoku_loss = 0.0

    for _ in range(model.n_sup):
        # Update LR for this specific backprop step (per-step, not per-batch)
        lr = get_lr(global_step, warmup_steps, num_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # reasoning_step re-runs encoder + embed internally for a fresh graph
        noise_pred, sudoku_logits, z_H, z_L = model.reasoning_step(
            conditions, noisy, z_H, z_L, timesteps, puzzle_ids=puzzle_ids
        )

        diff_loss = F.mse_loss(noise_pred, noise)

        sudoku_loss = torch.tensor(0.0, device=accelerator.device)
        if sudoku_logits is not None and sudoku_loss_weight > 0:
            B_, N, C = sudoku_logits.shape
            sudoku_loss = F.cross_entropy(
                sudoku_logits.reshape(B_ * N, C),
                solution[:, :N].reshape(B_ * N),
                ignore_index=IGNORE_LABEL_ID,
            )

        loss = diff_loss + sudoku_loss_weight * sudoku_loss
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model.parameters())

        total_loss_val   += loss.item()
        last_diff_loss    = diff_loss.item()
        last_sudoku_loss  = sudoku_loss.item()
        global_step      += 1

    return {
        "loss":        total_loss_val / model.n_sup,
        "diff_loss":   last_diff_loss,
        "sudoku_loss": last_sudoku_loss,
    }, global_step


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
    metrics: dict[str, list] = {
        "loss": [], "diff_loss": [], "sudoku_loss": [],
        "thinker_cell_acc": [], "thinker_puzzle_acc": [],
    }

    max_eval_batches = 10
    for i, batch in tqdm(enumerate(dataloader), "Evaluating", total=max_eval_batches):
        if i >= max_eval_batches:
            break
        m = compute_losses(model, batch, scheduler, accelerator, sudoku_loss_weight)
        for k in metrics:
            val = m.get(k)
            if val is not None:
                metrics[k].append(val.item() if torch.is_tensor(val) else float(val))

    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items() if v}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save(accelerator, model, optimizer, step, output_dir, tag, ema=None):
    ckpt = {
        "step":            step,
        "model_state":     accelerator.unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ema_state":       ema.state_dict() if ema is not None else None,
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs/mnist_sudoku", config_name="config")
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
        # Derive run name from the output_dir leaf (e.g. "runs/mnist_sudoku_v0" → "mnist_sudoku_v0")
        run_name = Path(args.output_dir).name

        init_kwargs = {"wandb": {"name": run_name, "settings": wandb.Settings(init_timeout=300)}}
        accelerator.init_trackers(
            project_name=wandb_project,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs=init_kwargs,
        )

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
    classifier_path = args.train.get("eval_classifier_path", None)
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
        weight_decay=args.train.get("weight_decay", 1.0),
    )

    model, optimizer, train_dl, eval_dl = accelerator.prepare(
        model, optimizer, train_dl, eval_dl
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = None
    if args.get("use_ema", True):
        ema = EMA(
            accelerator.unwrap_model(model).parameters(),
            decay=args.get("ema_decay", 0.9999),
        )
        ema.to(accelerator.device)
        if accelerator.is_main_process:
            logger.info(f"EMA enabled (decay={ema.decay})")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    resume_path = args.get("resume_from_checkpoint", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = int(ckpt["step"])
        if ema is not None and "ema_state" in ckpt and ckpt["ema_state"] is not None:
            ema.load_state_dict(ckpt["ema_state"], device=accelerator.device)
        logger.info(f"Resumed from {resume_path} at step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    num_steps    = args.train.num_steps
    warmup_steps = args.train.get("warmup_steps", 1000)
    eval_every   = args.train.get("eval_every", 2000)
    save_every   = args.train.get("save_every", 10000)
    log_every    = args.train.get("log_every", 100)
    sudoku_w     = args.train.get("sudoku_loss_weight", 0.0)

    n_sup       = accelerator.unwrap_model(model).n_sup
    best_loss   = float("inf")
    train_iter  = iter(train_dl)

    # Threshold-based triggers so intervals fire correctly even when n_sup
    # doesn't divide them evenly (same pattern as train_sudoku.py).
    next_log  = log_every
    next_eval = eval_every
    next_save = save_every

    progress_bar = tqdm(
        total=num_steps,
        initial=global_step,
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

        m, global_step = train_step(
            accelerator.unwrap_model(model),
            batch, scheduler, accelerator, optimizer,
            global_step, warmup_steps, num_steps, args.train.lr,
            sudoku_loss_weight=sudoku_w,
            ema=ema,
        )
        progress_bar.update(n_sup)

        if global_step >= next_log and accelerator.is_main_process:
            lr = get_lr(global_step, warmup_steps, num_steps, args.train.lr)
            logger.info(
                f"step={global_step}  loss={m['loss']:.4f}  "
                f"diff={m['diff_loss']:.4f}  "
                f"sudoku={m['sudoku_loss']:.4f}  lr={lr:.2e}"
            )
            if wandb_project:
                accelerator.log({
                    "train/loss":        m["loss"],
                    "train/diff_loss":   m["diff_loss"],
                    "train/sudoku_loss": m["sudoku_loss"],
                    "train/lr":          lr,
                }, step=global_step)
            next_log = global_step + log_every

        if global_step >= next_eval:
            unwrapped = accelerator.unwrap_model(model)
            if ema is not None:
                live_params = [p.clone() for p in unwrapped.parameters()]
                ema.copy_to(unwrapped.parameters())
            metrics = eval_loop(
                unwrapped,
                eval_dl, scheduler, accelerator,
                sudoku_loss_weight=sudoku_w,
            )
            if ema is not None:
                for p, live in zip(unwrapped.parameters(), live_params):
                    p.data.copy_(live)
            if accelerator.is_main_process:
                thinker_suffix = ""
                if "thinker_cell_acc" in metrics:
                    thinker_suffix = (
                        f"  thinker_cell={metrics['thinker_cell_acc']:.4f}"
                        f"  thinker_puzzle={metrics['thinker_puzzle_acc']:.4f}"
                    )
                logger.info(
                    f"[eval] step={global_step}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"diff={metrics['diff_loss']:.4f}  "
                    f"sudoku={metrics['sudoku_loss']:.4f}"
                    + thinker_suffix
                )
                if wandb_project:
                    wandb_eval = {
                        "eval/loss":        metrics["loss"],
                        "eval/diff_loss":   metrics["diff_loss"],
                        "eval/sudoku_loss": metrics["sudoku_loss"],
                    }
                    if "thinker_cell_acc" in metrics:
                        wandb_eval["eval/thinker_cell_acc"]   = metrics["thinker_cell_acc"]
                        wandb_eval["eval/thinker_puzzle_acc"] = metrics["thinker_puzzle_acc"]
                    accelerator.log(wandb_eval, step=global_step)
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    _save(accelerator, model, optimizer, global_step, args.output_dir, "best", ema=ema)

                # Digit-level eval via multi-batch DDIM sampling + classifier
                if classifier is not None:
                    n_total = args.train.get("eval_num_samples",    128)
                    n_batch = args.train.get("eval_batch_size",      32)
                    n_ddim  = args.train.get("eval_num_ddim_steps",  20)
                    n_log   = args.train.get("eval_num_log_images",  10)

                    all_cell_acc    = []
                    all_puzzle_acc  = []
                    ts_cell_accs:   dict[int, list[float]] = {}
                    ts_puzzle_accs: dict[int, list[float]] = {}
                    panels_list:    list = []
                    n_panels_done   = 0
                    n_done          = 0

                    for eb in DataLoader(eval_ds, batch_size=n_batch, shuffle=False):
                        if n_done >= n_total:
                            break
                        conds = eb["conditions"]
                        sols  = eb["solution"]
                        pids  = eb.get("puzzle_id", None)
                        B_cur = conds.shape[0]

                        sr = sample_grids(
                            accelerator.unwrap_model(model),
                            conds,
                            num_train_timesteps=args.get("num_timesteps", 1000),
                            beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
                            num_steps=n_ddim,
                            device=accelerator.device,
                            puzzle_ids=pids,
                            solutions=sols,
                        )
                        generated = sr["generated"]

                        acc = evaluate_grids(generated, sols, classifier, cell_size)
                        all_cell_acc.append(acc["cell_acc"])
                        all_puzzle_acc.append(acc["puzzle_acc"])

                        for t, a in sr.get("ts_cell_acc", []):
                            ts_cell_accs.setdefault(t, []).append(a)
                        for t, a in sr.get("ts_puzzle_acc", []):
                            ts_puzzle_accs.setdefault(t, []).append(a)

                        # Build panel images for the first n_log samples
                        if wandb_project and n_panels_done < n_log:
                            n_new     = min(n_log - n_panels_done, B_cur)
                            tp_all    = sr.get("best_thinker_preds")   # (B, N) cpu int64 or None
                            tt_all    = sr.get("best_thinker_ts")      # list[int] or None
                            sols_np   = sols.cpu().numpy()
                            for i in range(n_new):
                                tp = tp_all[i].numpy() if tp_all is not None else None
                                tt = tt_all[i]         if tt_all is not None else None
                                panel = make_panel_image(
                                    conds[i], generated[i], sols_np[i],
                                    thinker_preds=tp, thinker_t=tt,
                                    img_size=cell_size * 9,
                                )
                                panels_list.append(
                                    wandb.Image(panel, caption=f"sample[{n_done + i}]")
                                )
                            n_panels_done += n_new

                        n_done += B_cur

                    mean_cell   = float(np.mean(all_cell_acc))
                    mean_puzzle = float(np.mean(all_puzzle_acc))
                    logger.info(
                        f"[eval] cell_acc={mean_cell:.4f}  "
                        f"puzzle_acc={mean_puzzle:.4f}  "
                        f"(over {n_done} samples)"
                    )
                    if wandb_project:
                        wandb_acc: dict = {
                            "eval/cell_acc":   mean_cell,
                            "eval/puzzle_acc": mean_puzzle,
                        }
                        if panels_list:
                            wandb_acc["eval/samples"] = panels_list
                        if ts_cell_accs:
                            curve = plot_thinker_ts_curve(ts_cell_accs, ts_puzzle_accs)
                            wandb_acc["eval/thinker_vs_timestep"] = wandb.Image(curve)
                        accelerator.log(wandb_acc, step=global_step)
            next_eval = global_step + eval_every

        if global_step >= next_save and accelerator.is_main_process:
            _save(accelerator, model, optimizer, global_step, args.output_dir, f"step-{global_step}", ema=ema)
            next_save = global_step + save_every

    if accelerator.is_main_process:
        _save(accelerator, model, optimizer, global_step, args.output_dir, "final", ema=ema)
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
