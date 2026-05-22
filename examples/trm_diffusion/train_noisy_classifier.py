"""
train_noisy_classifier.py — Train MNISTCellClassifier with DDPM noise augmentation.

Trains the same architecture as the clean classifier but augments each batch with
noise at uniformly-sampled timesteps, so the classifier generalises to the noisy
x0_pred images produced during diffusion training.

Pass --timestep_cond to also inject a learned timestep embedding into the
features before the FC head. This allows the classifier to use the noise level
as a cue, which improves accuracy at high timesteps significantly.

Usage:
    python train_noisy_classifier.py \
        --save_path  runs/noisy_classifier/classifier_noisy.pt \
        --mnist_root data/mnist \
        --cell_size  16 \
        --num_timesteps 100 \
        [--timestep_cond]

The saved checkpoint is fully compatible with load_or_train_classifier and
evaluate_grids — it stores a "timestep_cond" flag so the right model class
is instantiated on load.  Use load_noisy_classifier() from this module to
load it correctly; the returned model has the same forward signature as
MNISTCellClassifier when called without a timestep argument.
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from eval.mnist_eval import MNISTCellClassifier, _load_mnist_digits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Timestep-conditioned classifier ───────────────────────────────────────────

class MNISTCellClassifierTimestep(nn.Module):
    """Same encoder as MNISTCellClassifier but with a learned timestep embedding.

    The embedding is added to the flattened encoder features before the FC head,
    letting the model adapt its predictions based on the noise level.

    forward(x, t=None):
        x: (B, 1, H, W) float in [0, 1]
        t: (B,) long timestep indices, or None (falls back to noise-unaware mode)
    """

    def __init__(self, cell_size: int = 16, num_timesteps: int = 100, t_emb_dim: int = 64):
        super().__init__()
        self.cell_size = cell_size
        self.num_timesteps = num_timesteps
        feat_dim = 64 * 4 * 4   # matches MNISTCellClassifier encoder output

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.t_emb  = nn.Embedding(num_timesteps, t_emb_dim)
        self.t_proj = nn.Linear(t_emb_dim, feat_dim)
        self.fc     = nn.Linear(feat_dim, 9)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.encoder(x).flatten(1)          # (B, feat_dim)
        if t is not None:
            feats = feats + self.t_proj(self.t_emb(t))
        return self.fc(feats)


def make_alpha_bars(num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    return torch.cumprod(1.0 - betas, dim=0)   # (T,)


def add_ddpm_noise_at(imgs: torch.Tensor, alpha_bars: torch.Tensor,
                      t_idx: torch.Tensor) -> torch.Tensor:
    """Add DDPM noise at the given per-sample timestep indices.

    imgs:      (B, 1, H, W) float in [0, 1]
    alpha_bars: (T,) precomputed cumulative product schedule
    t_idx:     (B,) long indices into alpha_bars
    """
    ab  = alpha_bars[t_idx].to(imgs.device)
    sa  = ab.sqrt().view(-1, 1, 1, 1)
    s1a = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
    return (sa * imgs + s1a * torch.randn_like(imgs)).clamp(0.0, 1.0)


def load_noisy_classifier(path: str, device: torch.device):
    """Load a checkpoint saved by this script.  Returns an eval-mode model.

    The returned model accepts forward(x, t=None):
      - plain MNISTCellClassifier when timestep_cond=False
      - MNISTCellClassifierTimestep when timestep_cond=True
    Both accept an optional t argument so call sites are uniform.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cell_size = ckpt.get("cell_size", 16)
    if ckpt.get("timestep_cond", False):
        model = MNISTCellClassifierTimestep(
            cell_size=cell_size,
            num_timesteps=ckpt.get("noise_timesteps", 100),
        ).to(device)
    else:
        model = MNISTCellClassifier(cell_size=cell_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(
        f"Loaded {'timestep-conditioned' if ckpt.get('timestep_cond') else 'noisy'} "
        f"classifier from {path}"
    )
    return model.eval()


def train(args):
    device = torch.device(args.device)

    logger.info(f"Loading MNIST digits from {args.mnist_root} …")
    images, labels = _load_mnist_digits(args.mnist_root, "train")

    if args.cell_size != 28:
        t      = torch.from_numpy(images[:, None]).float()
        t      = F.interpolate(t, size=(args.cell_size, args.cell_size),
                               mode="bilinear", align_corners=False)
        images = t.squeeze(1).numpy()

    dataset = TensorDataset(
        torch.from_numpy(images[:, None]),
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    alpha_bars = make_alpha_bars(args.num_timesteps, args.beta_start, args.beta_end).to(device)

    if args.timestep_cond:
        model = MNISTCellClassifierTimestep(
            cell_size=args.cell_size, num_timesteps=args.num_timesteps,
        ).to(device)
        logger.info(f"Using timestep-conditioned classifier")
    else:
        model = MNISTCellClassifier(cell_size=args.cell_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f"Training with noise augmentation T={args.num_timesteps} for {args.epochs} epochs …")
    model.train()
    for epoch in range(args.epochs):
        total = correct = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.shape[0]
            t_idx = torch.randint(0, args.num_timesteps, (B,), device=device)
            imgs  = add_ddpm_noise_at(imgs, alpha_bars, t_idx)
            logits = model(imgs, t_idx) if args.timestep_cond else model(imgs)
            loss   = F.cross_entropy(logits, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == lbls).sum().item()
            total   += len(lbls)
        logger.info(f"  epoch {epoch + 1}/{args.epochs}  train_acc={correct / total:.4f}")

    # ── Evaluate on clean images ───────────────────────────────────────────────
    logger.info("Evaluating on clean test images …")
    test_images, test_labels = _load_mnist_digits(args.mnist_root, "test")
    if args.cell_size != 28:
        t           = torch.from_numpy(test_images[:, None]).float()
        t           = F.interpolate(t, size=(args.cell_size, args.cell_size),
                                    mode="bilinear", align_corners=False)
        test_images = t.squeeze(1).numpy()

    test_ds     = TensorDataset(torch.from_numpy(test_images[:, None]),
                                torch.from_numpy(test_labels))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        total_clean = correct_clean = 0
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            # Pass t=0 for timestep-conditioned model (clean image = t=0)
            t_zeros = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
            logits  = model(imgs, t_zeros) if args.timestep_cond else model(imgs)
            correct_clean += (logits.argmax(1) == lbls).sum().item()
            total_clean   += len(lbls)

        logger.info(f"  clean acc = {correct_clean / total_clean:.4f}")
        T = args.num_timesteps
        for t_eval in [T // 10, T // 4, T // 2, 3 * T // 4, T - 1]:
            t_fixed = torch.full((512,), t_eval, dtype=torch.long, device=device)
            total_n = correct_n = 0
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                B = imgs.shape[0]
                t_b   = t_fixed[:B]
                noisy = add_ddpm_noise_at(imgs, alpha_bars, t_b)
                logits = model(noisy, t_b) if args.timestep_cond else model(noisy)
                correct_n += (logits.argmax(1) == lbls).sum().item()
                total_n   += len(lbls)
            logger.info(f"  t={t_eval:3d}  noisy acc = {correct_n / total_n:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "cell_size":       args.cell_size,
        "noise_timesteps": args.num_timesteps,
        "beta_start":      args.beta_start,
        "beta_end":        args.beta_end,
        "timestep_cond":   args.timestep_cond,
    }, args.save_path)
    logger.info(f"Saved → {args.save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save_path",     default="runs/noisy_classifier/classifier_noisy.pt")
    p.add_argument("--mnist_root",    default="data/mnist")
    p.add_argument("--cell_size",     type=int,   default=16)
    p.add_argument("--num_timesteps", type=int,   default=100,
                   help="Number of DDPM timesteps — must match the diffusion model")
    p.add_argument("--beta_start",    type=float, default=0.0001)
    p.add_argument("--beta_end",      type=float, default=0.02)
    p.add_argument("--epochs",        type=int,   default=20,
                   help="More epochs than clean training because task is harder")
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timestep_cond", action="store_true",
                   help="Add a learned timestep embedding to the classifier head")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()