"""
experiments/probe_painter_representations.py — Linear-probe the frozen
painter's mid-block activations for digit-category information.

Question: does the frozen diffusion model's internal feature space, at the
exact resolution/layer ConditioningPyramid injects its mid_block residual
into, linearly encode which digit belongs at each sudoku cell, at a given
noise level? Standard linear-probing methodology (Alain & Bengio 2016):
freeze the model, train ONLY a plain logistic-regression head on its frozen
activations, and read decodability off the probe's HELD-OUT accuracy —
success is attributable to the representation, not the probe (kept
deliberately weak/linear on purpose).

Sweeps a list of timesteps, so you can see at which noise level digit
identity becomes linearly decodable — directly informs the x0_hint
threshold question elsewhere in this project. Run once per checkpoint
(plain vs scaled painter) to compare representation quality between the
working and broken settings — that comparison is the point, not either
number in isolation.

Only the frozen painter + scheduler + train dataset are used; no thinker
checkpoint is required. mid_block's spatial resolution must be evenly
divisible by 9 (true for the current mnist_unet backbone: 36x36) — the
script asserts this rather than silently mispooling.

Usage:
    python experiments/probe_painter_representations.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      data=mnist_sudoku \\
      +timesteps=[0,10,30,50,70,90] \\
      +num_batches=8

    python experiments/probe_painter_representations.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_scaled_painter/checkpoint_final.pt \\
      data=mnist_sudoku_scaled \\
      +timesteps=[0,10,30,50,70,90] \\
      +num_batches=8
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from datasets.data_sample import DataSample, collate_data_samples
from factory import build_datasets, build_model

GRID = 9  # sudoku grid size


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [0, 10, 30, 50, 70, 90])]
    num_batches = int(cfg.get("num_batches", 8))
    batch_size = int(cfg.get("probe_batch_size", 64))
    seed = int(cfg.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = instantiate(cfg.diffusion)
    train_ds, _ = build_datasets(cfg)
    model = build_model(cfg, scheduler).to(device)
    model.eval()
    painter = getattr(model, "painter", model)
    painter.eval()

    unet = painter.unet

    activations: dict = {}

    def _hook(_module, _inputs, out):
        activations["mid"] = out

    handle = unet.mid_block.register_forward_hook(_hook)

    rng = np.random.default_rng(seed)

    print(f"chance level (1/9) = {1/9:.3f}")
    for t in timesteps:
        feats_all, labels_all = [], []
        with torch.no_grad():
            for _ in range(num_batches):
                idx = rng.integers(0, len(train_ds), size=batch_size)
                batch = collate_data_samples([train_ds[int(i)] for i in idx]).to(device)
                images = batch.images
                solution = batch.solution  # (B, 81) in {-100} U {0..8}

                t_batch = torch.full((images.shape[0],), t, device=device, dtype=torch.long)
                noise = torch.randn_like(images)
                x_noisy = scheduler.add_noise(images, noise, t_batch)

                painter(DataSample(x_noisy=x_noisy, timesteps=t_batch), steering=None)
                feat = activations["mid"]  # (B, C, H, W)

                assert feat.shape[-1] % GRID == 0, (
                    f"mid_block resolution {feat.shape[-1]} not divisible by {GRID}; "
                    "adjust pooling for this backbone."
                )
                pool = feat.shape[-1] // GRID
                pooled = F.avg_pool2d(feat, kernel_size=pool)  # (B, C, 9, 9)
                B, C = pooled.shape[:2]
                pooled = pooled.permute(0, 2, 3, 1).reshape(B * GRID * GRID, C)  # (B*81, C)
                labs = solution.reshape(-1)

                valid = labs >= 0
                feats_all.append(pooled[valid].cpu().numpy())
                labels_all.append(labs[valid].cpu().numpy())

        X = np.concatenate(feats_all, axis=0)
        y = np.concatenate(labels_all, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"t={t:>4d}  n={len(y):>6d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    handle.remove()


if __name__ == "__main__":
    main()
