"""
train_vae.py — KL-VAE training for MNIST Sudoku images (144×144 → 36×36×4).

Usage:
    python train_vae.py
    python train_vae.py train.num_steps=50000 train.kl_weight=1e-6
    accelerate launch --num_processes=2 train_vae.py
"""

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset


logger = get_logger(__name__, log_level="INFO")


def build_vae(cfg) -> AutoencoderKL:
    m = cfg.model
    return AutoencoderKL(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        down_block_types=list(m.down_block_types),
        up_block_types=list(m.up_block_types),
        block_out_channels=list(m.block_out_channels),
        layers_per_block=m.layers_per_block,
        latent_channels=m.latent_channels,
        norm_num_groups=m.norm_num_groups,
        act_fn=m.act_fn,
    )


def build_datasets(cfg):
    d = cfg.data
    train_ds = MNISTSudokuDataset(
        sudoku_dir=d.sudoku_dir,
        mnist_root=d.mnist_root,
        cell_size=d.cell_size,
        mnist_split="train",
    )
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=d.sudoku_dir,
        mnist_root=d.mnist_root,
        cell_size=d.cell_size,
        mnist_split="test",
    )
    return train_ds, eval_ds


def save_checkpoint(accelerator, vae, optimizer, scheduler, step, output_dir, tag):
    ckpt = {
        "step": step,
        "model_state": accelerator.unwrap_model(vae).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


@torch.no_grad()
def log_reconstructions(vae, batch, output_dir, step, num_images, device):
    vae.eval()
    images = batch["images"][:num_images].to(device)
    posterior = vae.encode(images).latent_dist
    recon = vae.decode(posterior.mode()).sample.clamp(0, 1)
    grid = torch.cat([images, recon], dim=0)
    path = os.path.join(output_dir, f"recon_step{step:06d}.png")
    save_image(grid, path, nrow=num_images)
    vae.train()


@hydra.main(version_base=None, config_path="configs", config_name="vae_config")
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

    train_ds, eval_ds = build_datasets(cfg)
    n_workers = cfg.data.num_workers
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
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )

    vae = build_vae(cfg)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        logger.info(f"VAE parameters: {n_params:,}")

    num_steps = cfg.train.num_steps
    optimizer = AdamW(vae.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.train.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=num_steps - cfg.train.warmup_steps, eta_min=cfg.train.lr * cfg.train.lr_min_ratio)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.warmup_steps])

    vae, optimizer, train_dl, eval_dl = accelerator.prepare(vae, optimizer, train_dl, eval_dl)

    global_step = 0
    resume_path = cfg.run.get("resume_from_checkpoint", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(vae).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        global_step = int(ckpt["step"])
        logger.info(f"Resumed from {resume_path} at step {global_step}")

    eval_every = cfg.eval.eval_every
    save_every = cfg.eval.save_every
    log_every = cfg.eval.log_every
    kl_weight = cfg.train.kl_weight
    num_log_images = cfg.eval.num_log_images

    next_log = log_every
    next_eval = eval_every
    next_save = save_every

    train_iter = iter(train_dl)
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
        vae.train()
        batch = _next_batch()
        images = batch["images"].to(accelerator.device)

        posterior = accelerator.unwrap_model(vae).encode(images).latent_dist
        z = posterior.sample()
        recon = accelerator.unwrap_model(vae).decode(z).sample

        recon_loss = F.mse_loss(recon, images)
        kl_loss = posterior.kl().mean()
        loss = recon_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        accelerator.backward(loss)
        if cfg.train.get("grad_clip", 0.0) > 0:
            accelerator.clip_grad_norm_(vae.parameters(), cfg.train.grad_clip)
        optimizer.step()
        lr_scheduler.step()

        global_step += 1
        progress_bar.update(1)

        if global_step >= next_log and accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            log_dict = {
                "train/recon_loss": recon_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/loss": loss.item(),
                "train/lr": lr,
            }
            logger.info(f"step={global_step}  " + "  ".join(f"{k}={v:.5f}" for k, v in log_dict.items()))
            if wandb_project:
                accelerator.log(log_dict, step=global_step)
            next_log = global_step + log_every

        if global_step >= next_eval and accelerator.is_main_process:
            unwrapped_vae = accelerator.unwrap_model(vae)
            unwrapped_vae.eval()
            val_recon = val_kl = 0.0
            n = 0
            with torch.no_grad():
                for val_batch in eval_dl:
                    val_images = val_batch["images"].to(accelerator.device)
                    posterior = unwrapped_vae.encode(val_images).latent_dist
                    z = posterior.mode()
                    recon = unwrapped_vae.decode(z).sample
                    val_recon += F.mse_loss(recon, val_images).item()
                    val_kl += posterior.kl().mean().item()
                    n += 1
                    if n >= 20:
                        break
            val_log = {"val/recon_loss": val_recon / n, "val/kl_loss": val_kl / n}
            logger.info(f"[val] step={global_step}  " + "  ".join(f"{k}={v:.5f}" for k, v in val_log.items()))
            if wandb_project:
                accelerator.log(val_log, step=global_step)

            # Save sample reconstructions
            val_batch = next(iter(eval_dl))
            log_reconstructions(
                unwrapped_vae, val_batch, cfg.run.output_dir,
                global_step, num_log_images, accelerator.device,
            )
            vae.train()
            next_eval = global_step + eval_every

        if global_step >= next_save and accelerator.is_main_process:
            save_checkpoint(
                accelerator, vae, optimizer, lr_scheduler,
                global_step, cfg.run.output_dir, f"step-{global_step}",
            )
            next_save = global_step + save_every

    if accelerator.is_main_process:
        save_checkpoint(
            accelerator, vae, optimizer, lr_scheduler,
            global_step, cfg.run.output_dir, "final",
        )
        # Compute and save latent scaling factor (std of latents on training set)
        logger.info("Computing latent scaling factor...")
        unwrapped_vae = accelerator.unwrap_model(vae)
        unwrapped_vae.eval()
        latent_samples = []
        with torch.no_grad():
            for i, b in enumerate(train_dl):
                z = unwrapped_vae.encode(b["images"].to(accelerator.device)).latent_dist.mode()
                latent_samples.append(z.cpu())
                if i >= 50:
                    break
        all_latents = torch.cat(latent_samples, dim=0)
        scaling_factor = 1.0 / all_latents.std().item()
        logger.info(f"Latent std={1/scaling_factor:.4f}  scaling_factor={scaling_factor:.4f}")
        torch.save({"scaling_factor": scaling_factor}, os.path.join(cfg.run.output_dir, "scaling_factor.pt"))
        logger.info("Training complete.")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
