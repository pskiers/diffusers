"""
experiments/visualize_x0_hint.py — Diagnostic: visualize x0-pred quality at
several diffusion timesteps.

Sanity-checks the assumption behind models.condition_encoders.
X0PredHintConditionEncoder — that a frozen painter's one-step x0 estimate at
a given noise level still carries the correct spatial layout (scale/offset)
of the ground-truth target, even when it can't recover exact identity.
Renders, per sample: the puzzle condition (if the dataset has one), the
ground-truth target, and x0_pred at each requested timestep, side by side.

Works for both pixel-space painters (MNIST: 1-channel, no VAE) and
latent-space painters (CLEVR: 3-channel RGB via VAE encode/decode).

Only the frozen painter + scheduler + val dataset are used — no thinker
checkpoint is required. Works with either a bare painter experiment
(mode=painter_base) or a thinker experiment (mode=thinker_base /
action_thinker_base, uses model.painter and ignores the untrained thinker).

Usage (MNIST):
    python experiments/visualize_x0_hint.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_scaled_painter/checkpoint_final.pt \\
      data=mnist_sudoku_scaled \\
      +timesteps=[10,30,50,70,90] \\
      +num_samples=6 \\
      +out_path=x0_hint_check.png

Usage (CLEVR, frozen painter forced unconditional — the "treat the DM as
unconditional" scenario, matching train.force_unconditional_painter):
    python experiments/visualize_x0_hint.py experiment=clevr_thinker_v0_controlnet \\
      painter.checkpoint=runs/clevr_unet_big_cfg/checkpoint_final.pt \\
      backbone.block_out_channels=[128,256,512] \\
      data.clevr_root=$SCRATCH/clevr \\
      +force_unconditional=true \\
      +timesteps=[10,30,50,70,90] \\
      +num_samples=6 \\
      +out_path=x0_hint_clevr.png
"""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets.data_sample import collate_data_samples
from factory import build_datasets, build_model
from models.diffusion_utils import x0_from_noise_pred


def _show(ax, img_chw: torch.Tensor) -> None:
    """imshow a (C, H, W) tensor already in [0, 1] — 1-channel gray or 3-channel RGB."""
    c = img_chw.shape[0]
    if c == 1:
        ax.imshow(img_chw[0].cpu(), cmap="gray", vmin=0, vmax=1)
    elif c == 3:
        ax.imshow(img_chw.permute(1, 2, 0).cpu().clamp(0, 1))
    else:
        ax.text(0.5, 0.5, f"{c}ch\n(not viewable)", ha="center", va="center", fontsize=8)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [10, 30, 50, 70, 90])]
    num_samples = int(cfg.get("num_samples", 6))
    out_path = str(cfg.get("out_path", "x0_hint_check.png"))
    # Force the frozen painter's own conditioning to null for the hint pass —
    # matches train.force_unconditional_painter, for checking what a painter
    # that's about to be trained/used that way can actually recover.
    force_unconditional = bool(cfg.get("force_unconditional", False))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = instantiate(cfg.diffusion)
    _, val_ds = build_datasets(cfg)
    model = build_model(cfg, scheduler).to(device)
    model.eval()

    # thinker_base/action_thinker_base experiments expose the frozen painter
    # as .painter; painter_base experiments ARE the painter.
    painter = getattr(model, "painter", model)
    painter.eval()
    has_vae = getattr(painter, "vae", None) is not None

    batch = collate_data_samples([val_ds[i] for i in range(num_samples)]).to(device)
    condition = batch.spatial_conditions  # (B, C, H, W) or None, depending on dataset/mode
    images = batch.images  # (B, C, H, W) ground-truth target, dataset's own pixel range
    images_disp = painter.images_to_log(images)  # -> [0, 1] regardless of VAE/tanh convention

    # Diffusion happens in VAE latent space for latent-space painters (CLEVR) —
    # noise must be added to the encoded latent, not the raw pixel image, or the
    # frozen UNet's conv_in gets the wrong channel count entirely.
    target = painter.encode(images) if has_vae else images

    show_condition = condition is not None and condition.shape[1] in (1, 3)
    ncols = (1 if show_condition else 0) + 1 + len(timesteps)
    fig, axes = plt.subplots(num_samples, ncols, figsize=(2.2 * ncols, 2.2 * num_samples), squeeze=False)
    col_titles = (["puzzle"] if show_condition else []) + ["ground truth"] + [f"x0_pred @ t={t}" for t in timesteps]

    with torch.no_grad():
        col0 = 0
        if show_condition:
            for row in range(num_samples):
                _show(axes[row][0], condition[row])
            col0 = 1
        for row in range(num_samples):
            _show(axes[row][col0], images_disp[row])

        for col, t in enumerate(timesteps, start=col0 + 1):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise = torch.randn_like(target)
            x_noisy = scheduler.add_noise(target, noise, t_batch)

            # Real sample's own conditioning fields carried through (so a
            # conditional frozen painter, e.g. CLEVR's, still has real
            # embedding_conditions/embedding_mask etc. to work with) unless
            # force_unconditional asks to null them out instead.
            hint_sample = dataclasses.replace(batch, x_noisy=x_noisy, timesteps=t_batch)
            if force_unconditional:
                hint_sample = painter.null_condition_sample(hint_sample)

            eps_pred = painter(hint_sample, steering=None).pred
            x0_pred = x0_from_noise_pred(eps_pred, x_noisy, t_batch, scheduler, clamp=not has_vae)
            x0_disp = painter.decode_for_eval(x0_pred) if has_vae else x0_pred.clamp(0.0, 1.0)
            for row in range(num_samples):
                _show(axes[row][col], x0_disp[row])

    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=9)
    for row in axes:
        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()