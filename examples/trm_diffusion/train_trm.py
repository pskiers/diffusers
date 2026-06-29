"""
train_trm.py — Unified training script for TRM models.

Usage:
    # Standalone sudoku training (reproduces original TRM paper):
    python train_trm.py experiment=sudoku

    # Painter-thinker (V0Tok) training:
    python train_trm.py experiment=v0tok

    # Multi-GPU:
    accelerate launch --num_processes=2 train_trm.py experiment=sudoku

Config: configs/trm/config.yaml  (select mode via experiment= override)

Design notes:
  - Training logic lives in model classes (train_step / eval_step).
  - LR schedule and EMA copied verbatim from TinyRecursiveModels/pretrain.py.
  - Optimizer: AdamATan2 (same as original) + optional SignSGD for puzzle embeddings.
  - Loss scaled by 1/global_batch_size to match original pretrain.py.
  - No accelerator.autocast() — TRM handles its own dtype via forward_dtype/CastedEmbedding.
"""

import os
import logging
from pathlib import Path

import hydra
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # fall back to eager on inductor failures
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from factory import build_model, build_datasets, build_scheduler
from models.trm.ema import EMAHelper
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


# ── Checkpoint ─────────────────────────────────────────────────────────────────


def save_checkpoint(accelerator, model, optimizers, step, output_dir, tag, ema_helper=None):
    ckpt = {
        "step": step,
        "model_state": accelerator.unwrap_model(model).state_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "ema_state": ema_helper.state_dict() if ema_helper is not None else None,
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    wandb_project = cfg.run.get("wandb_project", None)
    log_with = ["wandb"] if wandb_project else []

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        mixed_precision=cfg.precision.mixed_precision,
        log_with=log_with,
    )
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)

    if wandb_project and accelerator.is_main_process:
        run_name = Path(cfg.run.output_dir).name
        accelerator.init_trackers(
            project_name=wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": run_name}},
        )

    torch.manual_seed(cfg.train.seed + accelerator.process_index)

    # ── Dataset & scheduler ───────────────────────────────────────────────────
    scheduler = build_scheduler(cfg)
    train_ds, eval_ds = build_datasets(cfg)

    n_workers = cfg.data.num_workers
    train_collate_fn = getattr(type(train_ds), "collate_fn", None)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
        collate_fn=train_collate_fn,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.eval.get("batch_size", cfg.train.batch_size),
        shuffle=False,
        num_workers=0,  # eval iterates the dataloader twice; forking after CUDA init causes worker segfaults
        pin_memory=False,
        collate_fn=eval_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, scheduler)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    # ── torch.compile (opt-in) ────────────────────────────────────────────────
    # Compile submodules individually to avoid tracing the Python-level n_sup /
    # H_cycles / L_cycles loops (dynamic control flow defeats fullgraph compile).
    if cfg.train.compile:
        model.compile_submodules()
        if accelerator.is_main_process:
            logger.info("torch.compile applied")

    # ── Optimizers ────────────────────────────────────────────────────────────
    world_size = accelerator.num_processes
    global_batch_size = cfg.train.batch_size * world_size
    num_steps = cfg.train.num_steps
    optimizers = model.build_optimizers(world_size, num_steps)

    # Don't prepare optimizers — accelerate wraps them in AcceleratedOptimizer
    # which passes closure to step(), but AdamATan2/SignSGD don't accept it.
    # eval_dl is intentionally NOT prepared: accelerate wraps it in a
    # DataLoaderDispatcher which races with persistent_workers when iterated
    # twice per eval call (loss loop then sampling callback).
    model, train_dl = accelerator.prepare(model, train_dl)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_helper = None
    if cfg.ema.enabled:
        ema_helper = EMAHelper(mu=cfg.ema.rate)
        ema_helper.register(accelerator.unwrap_model(model))
        logger.info(f"EMA enabled (mu={cfg.ema.rate})")

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    resume_path = cfg.run.get("resume_from_checkpoint", None)
    load_opt = cfg.get("load_optimizer_state", True)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=False)
        if load_opt and ckpt.get("optimizer_states"):
            for opt, sd in zip(optimizers, ckpt["optimizer_states"]):
                opt.load_state_dict(sd)
        global_step = int(ckpt["step"])
        if ema_helper is not None:
            if load_opt and ckpt.get("ema_state") is not None:
                ema_helper.load_state_dict(ckpt["ema_state"])
                for k in ema_helper.shadow:
                    ema_helper.shadow[k] = ema_helper.shadow[k].to(accelerator.device)
            else:
                unwrapped = accelerator.unwrap_model(model)
                ema_helper.shadow = {
                    name: param.data.clone() for name, param in unwrapped.named_parameters() if param.requires_grad
                }
        logger.info(
            f"Resumed from {resume_path} at step {global_step}" + ("" if load_opt else " (optimizer state NOT loaded)")
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    eval_every = cfg.eval.eval_every
    save_every = cfg.eval.save_every
    log_every = cfg.eval.log_every
    grad_accum_steps = cfg.train.get("gradient_accumulation_steps", 1)

    next_log = log_every
    next_eval = eval_every
    next_save = save_every

    train_iter = iter(train_dl)
    unwrapped = accelerator.unwrap_model(model)
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

        metrics, lr, global_step = unwrapped.train_step(
            micro_batches,
            accelerator,
            optimizers,
            ema_helper,
            global_batch_size,
            global_step,
        )

        log_dict = {f"train/{k}": v for k, v in metrics.items()}
        log_dict["train/lr"] = lr

        step_size = getattr(unwrapped, "n_sup", 1)
        progress_bar.update(step_size)

        if global_step >= next_log and accelerator.is_main_process:
            logger.info(f"step={global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
            if wandb_project:
                accelerator.log(log_dict, step=global_step)
            next_log = global_step + log_every

        if global_step >= next_eval:
            if ema_helper is not None:
                live_params = [p.data.clone() for p in unwrapped.parameters() if p.requires_grad]
                ema_helper.ema(unwrapped)

            val_metrics = unwrapped.eval_step(eval_dl, accelerator, max_batches=100, step=global_step)
            val_log = {f"val/{k}": v for k, v in val_metrics.items()}

            if ema_helper is not None:
                for p, live in zip((p for p in unwrapped.parameters() if p.requires_grad), live_params):
                    p.data.copy_(live)
            unwrapped.train()

            if accelerator.is_main_process:
                logger.info(
                    f"[val] step={global_step}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in val_log.items() if isinstance(v, (int, float)))
                )
                if wandb_project:
                    accelerator.log(val_log, step=global_step)

            next_eval = global_step + eval_every

        if global_step >= next_save and accelerator.is_main_process:
            save_checkpoint(
                accelerator, model, optimizers, global_step, cfg.run.output_dir, f"step-{global_step}", ema_helper
            )
            next_save = global_step + save_every

    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizers, global_step, cfg.run.output_dir, "final", ema_helper)
        logger.info("Training complete.")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
