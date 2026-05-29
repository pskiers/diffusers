"""
train_spatial.py — Training script for spatially-varying latent diffusion.

Usage:
    python train_spatial.py vae.checkpoint=runs/vae/checkpoint_final.pt
    accelerate launch --num_processes=2 train_spatial.py vae.checkpoint=...
"""

import logging
import os
from pathlib import Path

import hydra
import hydra.utils
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.mnist_eval import load_or_train_classifier
from models.spatial_latent_unet import SpatialLatentConfig, SpatialLatentOptimConfig, SpatialLatentUNet
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


# ── Builders ───────────────────────────────────────────────────────────────────


def _build_vae(cfg: DictConfig) -> AutoencoderKL:
    v = cfg.vae
    return AutoencoderKL(
        in_channels=int(v.in_channels),
        out_channels=int(v.out_channels),
        latent_channels=int(v.latent_channels),
        down_block_types=list(v.down_block_types),
        up_block_types=list(v.up_block_types),
        block_out_channels=list(v.block_out_channels),
        layers_per_block=int(v.layers_per_block),
        norm_num_groups=int(v.norm_num_groups),
        act_fn=str(v.act_fn),
    )


def _build_model_cfg(cfg: DictConfig) -> SpatialLatentConfig:
    m = cfg.model
    cell_size = int(cfg.data.cell_size)
    return SpatialLatentConfig(
        vae_checkpoint=str(cfg.vae.checkpoint),
        latent_channels=int(m.latent_channels),
        latent_size=int(m.latent_size),
        down_block_types=tuple(m.down_block_types),
        up_block_types=tuple(m.up_block_types),
        block_out_channels=tuple(m.block_out_channels),
        layers_per_block=int(m.layers_per_block),
        attention_head_dim=int(m.attention_head_dim),
        norm_num_groups=int(m.norm_num_groups),
        vocab_size=int(m.vocab_size),
        cond_embed_dim=int(m.cond_embed_dim),
        f_spatial=float(m.f_spatial),
        tau_init=float(m.tau_init),
        tau_student=float(m.tau_student),
        n_octaves=int(m.n_octaves),
        p_refine_max=float(m.p_refine_max),
        p_refine_warmup_steps=int(m.p_refine_warmup_steps),
        teacher_ema_rate=float(m.teacher_ema_rate),
        noise_mode=str(m.get("noise_mode", "perlin")),
        stage1_steps=int(m.get("stage1_steps", 5000)),
        aug_prob_vanilla=float(m.get("aug_prob_vanilla", 0.20)),
        aug_prob_power=float(m.get("aug_prob_power", 0.30)),
        aug_prob_threshold=float(m.get("aug_prob_threshold", 0.30)),
        aug_prob_perlin=float(m.get("aug_prob_perlin", 0.20)),
        progress_min=float(m.get("progress_min", 0.1)),
        progress_max=float(m.get("progress_max", 0.9)),
        teacher_t_min_frac=float(m.get("teacher_t_min_frac", 0.6)),
        threshold_n_min=int(m.get("threshold_n_min", 1)),
        threshold_n_max=int(m.get("threshold_n_max", 3)),
        continuous_time=bool(m.get("continuous_time", False)),
        model_type=str(m.get("model_type", "unet")),
        patch_size=int(m.get("patch_size", 4)),
        n_heads=int(m.get("n_heads", 8)),
        n_layers=int(m.get("n_layers", 6)),
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        t_freq_dim=int(m.get("t_freq_dim", 256)),
        cell_size=cell_size,
        painter_size=cell_size * 9,
    )


def _build_optim_cfg(cfg: DictConfig) -> SpatialLatentOptimConfig:
    o = cfg.optim
    return SpatialLatentOptimConfig(
        lr=float(o.lr),
        weight_decay=float(o.weight_decay),
        warmup_steps=int(o.warmup_steps),
        lr_min_ratio=float(o.lr_min_ratio),
    )


def _build_datasets(cfg: DictConfig):
    base = hydra.utils.to_absolute_path(cfg.data.sudoku_dir)
    mnist_root = hydra.utils.to_absolute_path(cfg.data.mnist_root)
    cell_size = int(cfg.data.cell_size)
    train_ds = MNISTSudokuDataset(
        sudoku_dir=os.path.join(base, "train"),
        mnist_root=mnist_root,
        cell_size=cell_size,
        mnist_split="train",
        mask_given=True,
    )
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=os.path.join(base, "test"),
        mnist_root=mnist_root,
        cell_size=cell_size,
        mnist_split="test",
        mask_given=True,
    )
    return train_ds, eval_ds


def _save_checkpoint(accelerator, model, optimizers, step, output_dir, tag):
    ckpt = {
        "step": step,
        "model_state": accelerator.unwrap_model(model).state_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="configs", config_name="spatial_config")
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

    output_dir = hydra.utils.to_absolute_path(cfg.run.output_dir)
    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if wandb_project and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": Path(output_dir).name}},
        )

    torch.manual_seed(int(cfg.train.seed) + accelerator.process_index)

    # ── VAE ───────────────────────────────────────────────────────────────────
    vae_ckpt_path = hydra.utils.to_absolute_path(cfg.vae.checkpoint)
    vae = _build_vae(cfg)
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    sf_path = os.path.join(os.path.dirname(vae_ckpt_path), "scaling_factor.pt")
    scaling_factor = (
        torch.load(sf_path, map_location="cpu", weights_only=True)["scaling_factor"] if os.path.exists(sf_path) else 1.0
    )
    logger.info(f"VAE scaling factor: {scaling_factor:.4f}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

    # ── Eval classifier ───────────────────────────────────────────────────────
    clf_path = cfg.eval.get("classifier_path", None)
    eval_clf = None
    if clf_path is not None:
        clf_path = hydra.utils.to_absolute_path(clf_path)
        if os.path.exists(clf_path):
            eval_clf = load_or_train_classifier(clf_path, None, int(cfg.data.cell_size), "cuda")
            for p in eval_clf.parameters():
                p.requires_grad_(False)
        else:
            logger.warning(f"Classifier not found at {clf_path}, skipping accuracy eval")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = _build_model_cfg(cfg)
    model_kwargs = dict(
        model_cfg=model_cfg,
        optim_cfg=_build_optim_cfg(cfg),
        scheduler=scheduler,
        vae=vae,
        scaling_factor=scaling_factor,
        eval_clf=eval_clf,
    )
    if model_cfg.model_type == "dit":
        from models.spatial_dit import LatentSpatialDiT

        model = LatentSpatialDiT(**model_kwargs)
    else:
        model = SpatialLatentUNet(**model_kwargs)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable:,}")

    if cfg.train.get("compile", False):
        model.compile_submodules()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, eval_ds = _build_datasets(cfg)
    n_workers = int(cfg.data.num_workers)
    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=int(cfg.train.batch_size) * 2,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )

    # ── Optimizers ────────────────────────────────────────────────────────────
    world_size = accelerator.num_processes
    global_batch_size = int(cfg.train.batch_size) * world_size
    num_steps = int(cfg.train.num_steps)
    optimizers = model.build_optimizers(world_size, num_steps)

    model, train_dl, eval_dl = accelerator.prepare(model, train_dl, eval_dl)

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    resume_path = cfg.run.get("resume_from_checkpoint", None)
    if resume_path:
        resume_path = hydra.utils.to_absolute_path(resume_path)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=False)
        for opt, sd in zip(optimizers, ckpt.get("optimizer_states", [])):
            opt.load_state_dict(sd)
        global_step = int(ckpt["step"])
        logger.info(f"Resumed from {resume_path} at step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    eval_every = int(cfg.eval.eval_every)
    save_every = int(cfg.eval.save_every)
    log_every = int(cfg.eval.log_every)
    grad_accum = int(cfg.train.get("gradient_accumulation_steps", 1))
    cfg_prob = float(cfg.train.get("cfg_prob", 0.0))

    next_log, next_eval, next_save = log_every, eval_every, save_every

    train_iter = iter(train_dl)
    unwrapped = accelerator.unwrap_model(model)
    progress_bar = tqdm(
        total=num_steps, initial=global_step, disable=not accelerator.is_local_main_process, desc="Training"
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
        micro_batches = [_next_batch() for _ in range(grad_accum)]

        metrics, lr, global_step = unwrapped.train_step(
            micro_batches,
            accelerator,
            optimizers,
            global_batch_size,
            global_step,
            cfg_prob=cfg_prob,
        )
        progress_bar.update(1)

        log_dict = {f"train/{k}": v for k, v in metrics.items()}
        log_dict["train/lr"] = lr

        if global_step >= next_log and accelerator.is_main_process:
            logger.info(f"step={global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
            if wandb_project:
                accelerator.log(log_dict, step=global_step)
            next_log = global_step + log_every

        if global_step >= next_eval:
            val_metrics = unwrapped.eval_step(
                eval_dl,
                accelerator,
                step=global_step,
                max_batches=50,
                num_ddim_steps=int(cfg.eval.num_ddim_steps),
                num_samples=int(cfg.eval.num_samples),
                cfg_scale=float(cfg.eval.cfg_scale),
                num_log_images=int(cfg.eval.num_log_images),
            )
            val_log = {f"val/{k}": v for k, v in val_metrics.items()}
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
            _save_checkpoint(accelerator, model, optimizers, global_step, output_dir, f"step-{global_step}")
            next_save = global_step + save_every

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, model, optimizers, global_step, output_dir, "final")
        logger.info("Training complete.")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
