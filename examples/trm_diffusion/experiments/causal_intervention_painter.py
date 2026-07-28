"""
experiments/causal_intervention_painter.py — Causal-intervention test of the
frozen painter's feature space.

Complements probe_painter_representations.py's question ("is digit identity
present in the frozen model's activations") with a different one: "is there
a controllable, class-correlated causal channel at the exact points
ConditioningPyramid injects into" — i.e. is the pathway gradient descent
would actually have to discover, during real TRM+ControlNet training, one
that exists at all. Rather than inspecting raw backprop gradients (noisy,
hard to interpret for a single example), this uses activation steering /
concept-vector editing (difference-of-class-means, a la ActAdd-style
representation editing):

  1. Run the frozen, UNSTEERED painter over many real images at a fixed
     noise level, pool activations per sudoku cell at each tested layer,
     and compute the mean activation vector per digit class — a "concept
     vector" per digit. Feature maps are crop-aligned to each sample's
     actual content bounding box (recovered from the clean ground-truth
     image) before pooling, so canonical grid position pairs with the
     correct `solution` label even for mnist_sudoku_scaled — mirrors
     eval.mnist_eval.extract_and_resize_sudoku, applied to activations
     instead of pixels. This step is the only one that needs correct
     labels; the intervention/readout step below is self-referential
     (it only asks "does this cell's OWN output move toward the injected
     class", never comparing against ground truth) so it does not need
     the alignment fix.
  2. For a held-out image, add k * typical_activation_norm *
     unit(concept_vector[target_digit] - overall_mean) directly into a
     layer's activation at a target cell's spatial patch (via a forward
     hook — bypasses the ControlNetTranslator/ConditioningPyramid pathway
     entirely and injects straight into the UNet's own internals), run
     the rest of the forward pass unchanged, decode x0_pred, crop the
     target cell, and classify it. Strength k is relative to that layer's
     own typical activation norm (measured from the same batches used to
     build concept vectors), not an arbitrary absolute scale — a raw,
     uncalibrated strength can look like "no causal channel" simply
     because it was too small to matter, independent of whether one
     exists.
  3. Compare against a control: the same-magnitude but RANDOM (non-class)
     direction, at the same strengths — rules out "any big enough
     perturbation changes the digit" as an explanation for a positive
     result.

Tests two layers: mid_block (the coarse bottleneck ConditioningPyramid's
mid-block residual targets) and down_blocks[0]'s pre-downsample output (full
144x144 pixel resolution — one of the down_block_additional_residuals
injection points, and structurally closer to the skip-connection pathway
that carries most fine spatial detail in a standard UNet2DModel). A null
result at mid_block alone doesn't tell you much about the mechanism ControlNet
actually uses, since it also injects at every down-block resolution.

If accuracy-toward-target-digit rises with strength for the concept
direction but not the random control, there's a real, class-correlated,
controllable channel at that layer — evidence the causal pathway ControlNet
training would need to find is actually there. If neither direction moves
classification, or both do about equally, that's evidence against it.

Usage:
    python experiments/causal_intervention_painter.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      data=mnist_sudoku \\
      +timesteps=[10,50] \\
      +strengths=[0,0.25,0.5,1,2,4] \\
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


def align_feature_map(images: torch.Tensor, feat: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Per-sample crop-to-content-bbox + resize, applied to a feature map
    using the bounding box recovered from the clean ground-truth image —
    see module docstring. A no-op in practice for mnist_sudoku."""
    B, C, Hf, Wf = feat.shape
    Himg = images.shape[-1]
    ratio = Hf / Himg
    out = torch.empty_like(feat)
    for i in range(B):
        mask = images[i].amax(dim=0) > threshold
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = mask.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            out[i] = feat[i]
            continue
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        fr0, fr1 = max(0, round(r0 * ratio)), min(Hf, round(r1 * ratio))
        fc0, fc1 = max(0, round(c0 * ratio)), min(Wf, round(c1 * ratio))
        if fr1 <= fr0 or fc1 <= fc0:
            out[i] = feat[i]
            continue
        crop = feat[i : i + 1, :, fr0:fr1, fc0:fc1]
        out[i] = F.interpolate(crop, size=(Hf, Wf), mode="bilinear", align_corners=False)[0]
    return out


# ── Per-layer plumbing ────────────────────────────────────────────────────────
# mid_block's forward hook fires with a plain tensor as `out`. down_blocks[i]'s
# fires with (hidden_states, res_samples_tuple) — res_samples[1] is the last
# per-resnet output computed BEFORE that block's own downsample, i.e. still at
# the block's input resolution (see experiments/ dry-run notes: down_blocks[0]
# res_samples = (144x144, 144x144, 72x72) for this backbone — index 1 is the
# last full-resolution one, and one of the actual down_block_additional_
# residuals injection points).


def _capture_mid(_module, _inputs, out):
    return out


def _capture_down0(_module, _inputs, out):
    return out[1][1]


def _inject_mid(delta_vec, row, col, pool):
    def _hook(_module, _inputs, out):
        out = out.clone()
        r0, c0 = row * pool, col * pool
        out[:, :, r0 : r0 + pool, c0 : c0 + pool] += delta_vec.view(1, -1, 1, 1)
        return out

    return _hook


def _inject_down0(delta_vec, row, col, pool):
    def _hook(_module, _inputs, out):
        hidden, res_samples = out
        res_samples = list(res_samples)
        modified = res_samples[1].clone()
        r0, c0 = row * pool, col * pool
        modified[:, :, r0 : r0 + pool, c0 : c0 + pool] += delta_vec.view(1, -1, 1, 1)
        res_samples[1] = modified
        return (hidden, tuple(res_samples))

    return _hook


def _layer_specs(unet):
    return {
        "mid_block": dict(module=unet.mid_block, capture=_capture_mid, inject=_inject_mid),
        "down_block0": dict(module=unet.down_blocks[0], capture=_capture_down0, inject=_inject_down0),
    }


def _pooled_features(spec, activations, layer_name, painter, images, timesteps, scheduler):
    """Unsteered forward pass -> layer features, alignment-corrected, pooled
    to (B, 9, 9, C). Returns (pooled, pool_factor)."""
    noise = torch.randn_like(images)
    x_noisy = scheduler.add_noise(images, noise, timesteps)
    painter(DataSample(x_noisy=x_noisy, timesteps=timesteps), steering=None)
    feat = activations[layer_name]  # (B, C, H, W)
    feat = align_feature_map(images, feat, threshold=0.05)
    assert feat.shape[-1] % GRID == 0, f"{layer_name} resolution {feat.shape[-1]} not divisible by {GRID}"
    pool = feat.shape[-1] // GRID
    pooled = F.avg_pool2d(feat, kernel_size=pool)  # (B, C, 9, 9)
    return pooled.permute(0, 2, 3, 1), pool  # (B, 9, 9, C)


def _build_concept_vectors(spec, activations, layer_name, painter, train_ds, scheduler, t, num_batches, batch_size, device, rng):
    """Per-class mean pooled activation, overall mean, and typical (mean)
    per-cell activation norm — all alignment-corrected."""
    feats_by_class: list[list[np.ndarray]] = [[] for _ in range(N_CLASSES)]
    feats_all: list[np.ndarray] = []
    pool = None
    with torch.no_grad():
        for _ in range(num_batches):
            idx = rng.integers(0, len(train_ds), size=batch_size)
            batch = collate_data_samples([train_ds[int(i)] for i in idx]).to(device)
            t_batch = torch.full((batch.images.shape[0],), t, device=device, dtype=torch.long)
            pooled, pool = _pooled_features(spec, activations, layer_name, painter, batch.images, t_batch, scheduler)
            labs = batch.solution  # (B, 81) in {-100} U {0..8}
            flat_feat = pooled.reshape(-1, pooled.shape[-1]).cpu().numpy()
            flat_lab = labs.reshape(-1).cpu().numpy()
            valid = flat_lab >= 0
            feats_all.append(flat_feat[valid])
            for c in range(N_CLASSES):
                sel = flat_lab == c
                if sel.any():
                    feats_by_class[c].append(flat_feat[sel])
    all_feats = np.concatenate(feats_all, axis=0)
    overall_mean = all_feats.mean(axis=0)
    typical_norm = float(np.linalg.norm(all_feats, axis=1).mean())
    class_means = np.stack(
        [np.concatenate(feats_by_class[c], axis=0).mean(axis=0) for c in range(N_CLASSES)]
    )
    return class_means, overall_mean, typical_norm, pool


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [10, 50])]
    strengths = [float(k) for k in cfg.get("strengths", [0, 0.25, 0.5, 1, 2, 4])]
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

    rng = np.random.default_rng(seed)
    test_idx = rng.integers(0, len(val_ds), size=num_test_images)
    test_batch = collate_data_samples([val_ds[int(i)] for i in test_idx]).to(device)

    print(f"target cell = (row={target_row}, col={target_col})   chance = {1/N_CLASSES:.3f}")

    for layer_name, spec in _layer_specs(unet).items():
        activations: dict = {}

        def _capture(_m, _i, out, _spec=spec, _name=layer_name):
            activations[_name] = _spec["capture"](_m, _i, out)

        for t in timesteps:
            cap_handle = spec["module"].register_forward_hook(_capture)
            class_means, overall_mean, typical_norm, pool = _build_concept_vectors(
                spec, activations, layer_name, painter, train_ds, scheduler, t,
                num_concept_batches, concept_batch_size, device, rng,
            )
            cap_handle.remove()

            class_means_t = torch.from_numpy(class_means).float().to(device)
            overall_mean_t = torch.from_numpy(overall_mean).float().to(device)

            t_batch = torch.full((test_batch.images.shape[0],), t, device=device, dtype=torch.long)
            noise = torch.randn_like(test_batch.images)
            x_noisy = scheduler.add_noise(test_batch.images, noise, t_batch)

            r0, c0 = target_row * cell_size, target_col * cell_size

            print(f"\n=== layer={layer_name}  t={t}  typical_activation_norm={typical_norm:.3f} ===")
            print(f"{'target_digit':>12} {'strength':>9} {'concept_acc':>12} {'random_acc':>11} {'concept_dir_norm/typical':>24}")
            for target_digit in range(N_CLASSES):
                concept_delta_dir = class_means_t[target_digit] - overall_mean_t
                dir_norm = concept_delta_dir.norm().clamp_min(1e-8)
                unit_concept = concept_delta_dir / dir_norm
                random_dir = torch.randn_like(concept_delta_dir)
                unit_random = random_dir / random_dir.norm().clamp_min(1e-8)

                for k in strengths:
                    results = {}
                    for name, unit_dir in (("concept", unit_concept), ("random", unit_random)):
                        delta = k * typical_norm * unit_dir
                        inj_handle = spec["module"].register_forward_hook(
                            spec["inject"](delta, target_row, target_col, pool)
                        )
                        with torch.no_grad():
                            sample = DataSample(x_noisy=x_noisy, timesteps=t_batch)
                            eps_pred = painter(sample, steering=None).pred
                            x0_pred = x0_from_noise_pred(eps_pred, x_noisy, t_batch, scheduler)
                            patch = x0_pred[:, :, r0 : r0 + cell_size, c0 : c0 + cell_size]
                            preds = eval_clf(patch).argmax(dim=1)
                            results[name] = (preds == target_digit).float().mean().item()
                        inj_handle.remove()
                    print(
                        f"{target_digit:>12} {k:>9.2f} {results['concept']:>12.3f} {results['random']:>11.3f} "
                        f"{(dir_norm / typical_norm).item():>24.3f}"
                    )


if __name__ == "__main__":
    main()
