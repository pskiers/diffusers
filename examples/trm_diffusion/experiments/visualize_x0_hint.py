"""
experiments/visualize_x0_hint.py — Diagnostic: visualize x0-pred quality at
several diffusion timesteps.

Sanity-checks the assumption behind models.condition_encoders.
X0PredHintConditionEncoder — that a frozen, unsteered painter's one-step x0
estimate at a given noise level still carries the correct spatial layout
(scale/offset) of the ground-truth target, even when it can't recover exact
digit identity. Renders, per sample: the puzzle condition, the ground-truth
target, and x0_pred at each requested timestep, side by side.

Only the frozen painter + scheduler + val dataset are used — no thinker
checkpoint is required. Works with either a bare painter experiment
(mode=painter_base) or a thinker experiment (mode=thinker_base, uses
model.painter and ignores the untrained thinker).

Usage:
    python experiments/visualize_x0_hint.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_scaled_painter/checkpoint_final.pt \\
      data=mnist_sudoku_scaled \\
      +timesteps=[10,30,50,70,90] \\
      +num_samples=6 \\
      +out_path=x0_hint_check.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets.data_sample import DataSample, collate_data_samples
from factory import build_datasets, build_model
from models.diffusion_utils import x0_from_noise_pred


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [10, 30, 50, 70, 90])]
    num_samples = int(cfg.get("num_samples", 6))
    out_path = str(cfg.get("out_path", "x0_hint_check.png"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = instantiate(cfg.diffusion)
    _, val_ds = build_datasets(cfg)
    model = build_model(cfg, scheduler).to(device)
    model.eval()

    # thinker_base experiments expose the frozen painter as .painter;
    # painter_base experiments ARE the painter.
    painter = getattr(model, "painter", model)
    painter.eval()

    batch = collate_data_samples([val_ds[i] for i in range(num_samples)]).to(device)
    condition = batch.spatial_conditions  # (B, 1, H, W)
    images = batch.images                  # (B, 1, H, W) ground-truth target

    ncols = 2 + len(timesteps)
    fig, axes = plt.subplots(num_samples, ncols, figsize=(2.2 * ncols, 2.2 * num_samples), squeeze=False)
    col_titles = ["puzzle", "ground truth"] + [f"x0_pred @ t={t}" for t in timesteps]

    with torch.no_grad():
        for row in range(num_samples):
            axes[row][0].imshow(condition[row, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row][1].imshow(images[row, 0].cpu(), cmap="gray", vmin=0, vmax=1)

        for col, t in enumerate(timesteps, start=2):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise = torch.randn_like(images)
            x_noisy = scheduler.add_noise(images, noise, t_batch)
            hint_sample = DataSample(x_noisy=x_noisy, timesteps=t_batch)
            eps_pred = painter(hint_sample, steering=None).pred
            x0_pred = x0_from_noise_pred(eps_pred, x_noisy, t_batch, scheduler)
            for row in range(num_samples):
                axes[row][col].imshow(x0_pred[row, 0].cpu(), cmap="gray", vmin=0, vmax=1)

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
