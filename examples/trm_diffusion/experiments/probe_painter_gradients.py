"""
experiments/probe_painter_gradients.py — Linear-probe the GRADIENT of the
frozen painter's own reconstruction loss w.r.t. its mid-block activations,
for digit-category information.

This targets the question the investigation actually started from, which
turned out to be distinct from what the other two probing scripts answer:

  - probe_painter_representations.py: does the forward ACTIVATION encode
    digit identity? (yes, strongly for plain, present-but-weaker for scaled)
  - causal_intervention_painter.py: can a hand-picked direction in
    activation space STEER the output toward a digit? (inconclusive — this
    turned out to be its own hard, mechanistic-interpretability-flavored
    question — choice of direction and injection magnitude both confound
    the answer — and isn't actually the same question as the one below)
  - this script: does the GRADIENT a real ControlNet would receive at this
    point, via ordinary backprop of the diffusion loss through the frozen
    painter, carry digit-category information at all?

Method: identical linear-probing methodology to probe_painter_representations
(freeze everything, fit only a plain logistic-regression head, read
decodability off its held-out accuracy) but the thing being probed is
d(diffusion_loss)/d(activation) instead of the activation's own value.
diffusion_loss is the ordinary training objective (MSE against the true
noise/x0, whichever the scheduler's prediction_type is) computed from the
frozen painter's own unsteered prediction on real (noised) images — exactly
the loss a real ControlNet-augmented forward pass would backprop through
this same point, just without an actual trained ControlNet in the loop.

No injection, no strength sweep, no random-direction control — this is a
pure information-content probe, same shape and same cost as
probe_painter_representations.py, not an intervention.

Usage:
    python experiments/probe_painter_gradients.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      data=mnist_sudoku \\
      +timesteps=[0,10,30,50,70,90] \\
      +num_batches=8

    python experiments/probe_painter_gradients.py \\
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
from sklearn.preprocessing import StandardScaler

from datasets.data_sample import DataSample, collate_data_samples
from factory import build_datasets, build_model

GRID = 9  # sudoku grid size


def align_feature_map(images: torch.Tensor, feat: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Per-sample crop-to-content-bbox + resize — see
    probe_painter_representations.py for the full rationale. A no-op in
    practice for mnist_sudoku."""
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    timesteps = [int(t) for t in cfg.get("timesteps", [0, 10, 30, 50, 70, 90])]
    num_batches = int(cfg.get("num_batches", 8))
    batch_size = int(cfg.get("probe_batch_size", 64))
    bbox_threshold = float(cfg.get("bbox_threshold", 0.05))
    seed = int(cfg.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = instantiate(cfg.diffusion)
    train_ds, _ = build_datasets(cfg)
    model = build_model(cfg, scheduler).to(device)
    model.eval()
    painter = getattr(model, "painter", model)
    painter.eval()

    unet = painter.unet
    prediction_type = scheduler.config.prediction_type

    activations: dict = {}

    def _hook(_module, _inputs, out):
        # For the real (frozen-checkpoint) use case, out has no grad history
        # yet — requires_grad_(True) turns it into a fresh leaf. retain_grad()
        # covers it either way (also correct if some upstream param happens
        # to require grad, e.g. when dry-running against an unfrozen
        # painter_base checkpoint, where out would otherwise be a non-leaf
        # whose .grad is never populated).
        out.requires_grad_(True)
        out.retain_grad()
        activations["mid"] = out
        return out

    handle = unet.mid_block.register_forward_hook(_hook)

    rng = np.random.default_rng(seed)

    print(f"chance level (1/9) = {1/9:.3f}   prediction_type = {prediction_type}")
    for t in timesteps:
        feats_all, labels_all = [], []
        for _ in range(num_batches):
            idx = rng.integers(0, len(train_ds), size=batch_size)
            batch = collate_data_samples([train_ds[int(i)] for i in idx]).to(device)
            images = batch.images
            solution = batch.solution  # (B, 81) in {-100} U {0..8}

            t_batch = torch.full((images.shape[0],), t, device=device, dtype=torch.long)
            noise = torch.randn_like(images)
            x_noisy = scheduler.add_noise(images, noise, t_batch)
            target = noise if prediction_type == "epsilon" else images

            pred = painter(DataSample(x_noisy=x_noisy, timesteps=t_batch), steering=None).pred
            loss = F.mse_loss(pred, target)
            loss.backward()

            grad = activations["mid"].grad  # (B, C, H, W)
            grad = align_feature_map(images, grad, threshold=bbox_threshold)

            assert grad.shape[-1] % GRID == 0, (
                f"mid_block resolution {grad.shape[-1]} not divisible by {GRID}; "
                "adjust pooling for this backbone."
            )
            pool = grad.shape[-1] // GRID
            pooled = F.avg_pool2d(grad, kernel_size=pool)  # (B, C, 9, 9)
            B, C = pooled.shape[:2]
            pooled = pooled.permute(0, 2, 3, 1).reshape(B * GRID * GRID, C)  # (B*81, C)
            labs = solution.reshape(-1)

            valid = labs >= 0
            feats_all.append(pooled[valid].detach().cpu().numpy())
            labels_all.append(labs[valid].cpu().numpy())

        X = np.concatenate(feats_all, axis=0)
        y = np.concatenate(labels_all, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        # Gradients can sit at a wildly different scale than activations
        # (e.g. near a converged loss) — LogisticRegression's default L2
        # strength assumes roughly unit-scale features, so an unstandardized
        # tiny-magnitude gradient can make the fit collapse to predicting a
        # constant class (exactly chance on this class-balanced problem)
        # regardless of whether the gradient actually carries information.
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train_s, y_train)
        train_acc = clf.score(X_train_s, y_train)
        test_acc = clf.score(X_test_s, y_test)

        grad_typical_norm = float(np.linalg.norm(X, axis=1).mean())
        print(f"t={t:>4d}  n={len(y):>6d}  grad_norm={grad_typical_norm:.2e}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    handle.remove()


if __name__ == "__main__":
    main()
