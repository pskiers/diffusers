"""
experiments/causal_intervention_painter.py — Causal-intervention test of the
frozen painter's mid-block feature space.

Complements probe_painter_representations.py's question ("is digit identity
present in the frozen model's activations") with a different one: "is there
a controllable, class-correlated causal channel at the exact point
ConditioningPyramid injects into" — i.e. is the pathway gradient descent
would actually have to discover, during real TRM+ControlNet training, one
that exists at all. Rather than inspecting raw backprop gradients (noisy,
hard to interpret for a single example), this uses activation steering /
concept-vector editing (difference-of-class-means, a la ActAdd-style
representation editing):

  1. Run the frozen, UNSTEERED painter over many real images at a fixed
     noise level, pool mid-block activations per sudoku cell (same pooling
     as probe_painter_representations.py), and compute the mean activation
     vector per digit class — a "concept vector" per digit.
  2. For a held-out image with a known digit at some target cell, add
     k * (concept_vector[target_digit] - overall_mean) directly into the
     mid-block output at that cell's spatial patch (via a forward hook —
     bypasses the (currently zero-init-trivial, if untrained)
     ControlNetTranslator/ConditioningPyramid pathway entirely and injects
     straight into the UNet's own internals), run the rest of the forward
     pass unchanged, decode x0_pred, crop the target cell, and classify it.
  3. Compare against a control: the same-magnitude but RANDOM (non-class)
     direction, at the same strengths — rules out "any big perturbation
     changes the digit" as an explanation for a positive result.

If accuracy-toward-target-digit rises with strength for the concept
direction but not the random control, there's a real, class-correlated,
controllable channel at this layer — evidence the causal pathway ControlNet
training would need to find is actually there. If neither direction moves
classification, or both do about equally, that's evidence against it.

Usage:
    python experiments/causal_intervention_painter.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      data=mnist_sudoku \\
      +timesteps=[10,50] \\
      +strengths=[0,1,2,4,8] \\
      +target_row=4 +target_col=4 \\
      +num_concept_batches=8 \\
      +num_test_images=32
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

from datasets.data_sample import DataSample, collate_data_samples
from eval.mnist_eval import load_or_train_classifier
from factory import build_datasets, build_model
from models.diffusion_utils import x0_from_noise_pred

GRID = 9  # sudoku grid size
N_CLASSES = 9  # digits 1-9 -> class 0-8


def _pooled_features(painter, activations, images, timesteps, scheduler):
    """Unsteered forward pass -> mid-block features pooled to (B, 9, 9, C)."""
    noise = torch.randn_like(images)
    x_noisy = scheduler.add_noise(images, noise, timesteps)
    painter(DataSample(x_noisy=x_noisy, timesteps=timesteps), steering=None)
    feat = activations["mid"]  # (B, C, H, W)
    pool = feat.shape[-1] // GRID
    pooled = F.avg_pool2d(feat, kernel_size=pool)  # (B, C, 9, 9)
    return pooled.permute(0, 2, 3, 1), pool  # (B, 9, 9, C)


def _build_concept_vectors(painter, activations, train_ds, scheduler, t, num_batches, batch_size, device, rng):
    """Per-class mean pooled activation at cell positions, plus the overall mean."""
    feats_by_class: list[list[np.ndarray]] = [[] for _ in range(N_CLASSES)]
    feats_all: list[np.ndarray] = []
    t_batch_template = None
    with torch.no_grad():
        for _ in range(num_batches):
            idx = rng.integers(0, len(train_ds), size=batch_size)
            batch = collate_data_samples([train_ds[int(i)] for i in idx]).to(device)
            t_batch = torch.full((batch.images.shape[0],), t, device=device, dtype=torch.long)
            pooled, pool = _pooled_features(painter, activations, batch.images, t_batch, scheduler)
            t_batch_template = pool
            labs = batch.solution  # (B, 81) in {-100} U {0..8}
            flat_feat = pooled.reshape(-1, pooled.shape[-1]).cpu().numpy()
            flat_lab = labs.reshape(-1).cpu().numpy()
            valid = flat_lab >= 0
            feats_all.append(flat_feat[valid])
            for c in range(N_CLASSES):
                sel = flat_lab == c
                if sel.any():
                    feats_by_class[c].append(flat_feat[sel])
    overall_mean = np.concatenate(feats_all, axis=0).mean(axis=0)
    class_means = np.stack(
        [np.concatenate(feats_by_class[c], axis=0).mean(axis=0) for c in range(N_CLASSES)]
    )
    return class_means, overall_mean, t_batch_template


def _injection_hook_factory(delta_vec, row, col, pool):
    def _hook(_module, _inputs, out):
        out = out.clone()
        r0, c0 = row * pool, col * pool
        out[:, :, r0 : r0 + pool, c0 : c0 + pool] += delta_vec.view(1, -1, 1, 1)
        return out

    return _hook


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [10, 50])]
    strengths = [float(k) for k in cfg.get("strengths", [0, 1, 2, 4, 8])]
    target_row = int(cfg.get("target_row", 4))
    target_col = int(cfg.get("target_col", 4))
    num_concept_batches = int(cfg.get("num_concept_batches", 8))
    concept_batch_size = int(cfg.get("concept_batch_size", 64))
    num_test_images = int(cfg.get("num_test_images", 32))
    cell_size = int(cfg.data.cell_size)
    mnist_root = str(cfg.data.mnist_root)
    classifier_path = str(cfg.get("classifier_path", "runs/mnist_classifier_cell16.pt"))
    seed = int(cfg.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = instantiate(cfg.diffusion)
    train_ds, val_ds = build_datasets(cfg)
    model = build_model(cfg, scheduler).to(device)
    model.eval()
    painter = getattr(model, "painter", model)
    painter.eval()
    unet = painter.unet

    eval_clf = load_or_train_classifier(classifier_path, mnist_root, cell_size, device)
    for p in eval_clf.parameters():
        p.requires_grad_(False)

    activations: dict = {}

    def _capture_hook(_m, _i, out):
        activations["mid"] = out

    rng = np.random.default_rng(seed)
    test_idx = rng.integers(0, len(val_ds), size=num_test_images)
    test_batch = collate_data_samples([val_ds[int(i)] for i in test_idx]).to(device)

    print(f"target cell = (row={target_row}, col={target_col})   chance = {1/N_CLASSES:.3f}")

    for t in timesteps:
        cap_handle = unet.mid_block.register_forward_hook(_capture_hook)
        class_means, overall_mean, pool = _build_concept_vectors(
            painter, activations, train_ds, scheduler, t, num_concept_batches, concept_batch_size, device, rng
        )
        cap_handle.remove()

        class_means_t = torch.from_numpy(class_means).float().to(device)
        overall_mean_t = torch.from_numpy(overall_mean).float().to(device)

        t_batch = torch.full((test_batch.images.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(test_batch.images)
        x_noisy = scheduler.add_noise(test_batch.images, noise, t_batch)

        r0, c0 = target_row * cell_size, target_col * cell_size

        print(f"\n--- t={t} ---")
        print(f"{'target_digit':>12} {'strength':>9} {'concept_acc':>12} {'random_acc':>11}")
        for target_digit in range(N_CLASSES):
            concept_delta_dir = class_means_t[target_digit] - overall_mean_t
            random_dir = torch.randn_like(concept_delta_dir)
            random_dir = random_dir * (concept_delta_dir.norm() / random_dir.norm().clamp_min(1e-8))

            for k in strengths:
                results = {}
                for name, direction in (("concept", concept_delta_dir), ("random", random_dir)):
                    delta = k * direction
                    inj_handle = unet.mid_block.register_forward_hook(
                        _injection_hook_factory(delta, target_row, target_col, pool)
                    )
                    with torch.no_grad():
                        sample = DataSample(x_noisy=x_noisy, timesteps=t_batch)
                        eps_pred = painter(sample, steering=None).pred
                        x0_pred = x0_from_noise_pred(eps_pred, x_noisy, t_batch, scheduler)
                        patch = x0_pred[:, :, r0 : r0 + cell_size, c0 : c0 + cell_size]
                        preds = eval_clf(patch).argmax(dim=1)
                        results[name] = (preds == target_digit).float().mean().item()
                    inj_handle.remove()
                print(f"{target_digit:>12} {k:>9.1f} {results['concept']:>12.3f} {results['random']:>11.3f}")


if __name__ == "__main__":
    main()