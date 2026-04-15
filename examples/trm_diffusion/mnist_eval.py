"""
mnist_eval.py – Digit-level evaluation for MNIST Sudoku models.

Provides:
  MNISTCellClassifier      – small CNN classifying a single cell → digit 1-9.
  train_mnist_classifier   – trains and saves the classifier.
  load_or_train_classifier – loads or trains on demand.
  evaluate_grids           – classifies cells in generated images, returns cell/puzzle acc.
  sample_grids             – DDIM sampling helper used during training eval.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Classifier ────────────────────────────────────────────────────────────────

class MNISTCellClassifier(nn.Module):
    """Classifies a single Sudoku cell image into digit class 0–8 (digit 1–9).

    Works at any cell_size thanks to AdaptiveAvgPool2d.
    """

    def __init__(self, cell_size: int = 28):
        super().__init__()
        self.cell_size = cell_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x).flatten(1))


# ── Dataset helper ────────────────────────────────────────────────────────────

def _load_mnist_digits(mnist_root: str, split: str = "train"):
    """Return (images, labels) for MNIST digits 1-9.

    images: float32 (N, 28, 28) in [0, 1]
    labels: int64   (N,) in [0, 8]  (digit d → class d-1)

    Reads from numpy cache if present, otherwise falls back to torchvision.
    """
    cache_path = os.path.join(mnist_root, f"mnist_{split}_cache.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        imgs_list, lbls_list = [], []
        for d in range(1, 10):
            key = f"digit_{d}"
            if key in data:
                imgs = data[key].astype(np.float32)          # (N, 28, 28)
                imgs_list.append(imgs)
                lbls_list.append(np.full(len(imgs), d - 1, dtype=np.int64))
        return np.concatenate(imgs_list), np.concatenate(lbls_list)

    from torchvision import datasets, transforms
    ds = datasets.MNIST(
        mnist_root, train=(split == "train"), download=True,
        transform=transforms.ToTensor(),
    )
    images = ds.data.numpy().astype(np.float32) / 255.0   # (N, 28, 28)
    labels = ds.targets.numpy()
    mask   = labels > 0     # exclude digit 0
    return images[mask], (labels[mask] - 1).astype(np.int64)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mnist_classifier(
    mnist_root: str,
    cell_size: int,
    save_path: str,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
) -> MNISTCellClassifier:
    """Train an MNISTCellClassifier on digits 1-9 and save it."""
    logger.info("Training MNIST cell classifier …")
    images, labels = _load_mnist_digits(mnist_root, "train")

    if cell_size != 28:
        t      = torch.from_numpy(images[:, None]).float()
        t      = F.interpolate(t, size=(cell_size, cell_size), mode="bilinear", align_corners=False)
        images = t.squeeze(1).numpy()

    dataset = TensorDataset(
        torch.from_numpy(images[:, None]),   # (N, 1, H, W)
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MNISTCellClassifier(cell_size=cell_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total = correct = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss   = F.cross_entropy(logits, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == lbls).sum().item()
            total   += len(lbls)
        logger.info(f"  epoch {epoch + 1}/{epochs}  acc={correct / total:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cell_size": cell_size}, save_path)
    logger.info(f"Classifier saved → {save_path}")
    return model.eval()


def load_or_train_classifier(
    path: str,
    mnist_root: str,
    cell_size: int,
    device: torch.device,
) -> MNISTCellClassifier:
    """Load classifier from *path* if it exists, otherwise train and save it there."""
    if os.path.exists(path):
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = MNISTCellClassifier(ckpt.get("cell_size", cell_size)).to(device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded MNIST classifier from {path}")
    else:
        model = train_mnist_classifier(mnist_root, cell_size, path, device)
    return model.eval()


# ── Grid evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_grids(
    images:     torch.Tensor,          # (B, 1, H, W) float32 [0, 1]
    solutions:  torch.Tensor,          # (B, 81)      int64   [0-8]
    classifier: MNISTCellClassifier,
    cell_size:  int,
) -> dict:
    """Classify every cell in *images* and compare to *solutions*.

    Returns {"cell_acc": float, "puzzle_acc": float}.
    """
    device    = next(classifier.parameters()).device
    images    = images.to(device)
    solutions = solutions.to(device)

    B = images.shape[0]
    # unfold → (B, 1, 9, 9, cell_size, cell_size)
    cells = images.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    # → (B*81, 1, cell_size, cell_size)
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)

    preds = classifier(cells).argmax(dim=1).reshape(B, 81)
    sol   = solutions.reshape(B, 81)

    correct    = preds == sol
    cell_acc   = correct.float().mean().item()
    puzzle_acc = correct.all(dim=1).float().mean().item()
    return {"cell_acc": cell_acc, "puzzle_acc": puzzle_acc}


# ── DDIM sampling ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_grids(
    model,
    conditions:          torch.Tensor,        # (B, 1, H, W)
    num_train_timesteps: int,
    beta_schedule:       str,
    num_steps:           int,
    device:              torch.device,
    puzzle_ids:          torch.Tensor | None = None,  # (B,) long
) -> torch.Tensor:
    """DDIM-sample a batch of grids conditioned on *conditions*.

    Returns (B, 1, H, W) float32 in [0, 1].
    """
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
    )
    ddim.set_timesteps(num_steps)

    conditions = conditions.to(device)
    if puzzle_ids is not None:
        puzzle_ids = puzzle_ids.to(device)
    x = torch.randn_like(conditions)

    model.eval()
    for t in ddim.timesteps:
        ts          = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        noise_pred, _ = model(x, ts, conditions, puzzle_ids=puzzle_ids)
        x           = ddim.step(noise_pred, t, x).prev_sample

    return x.clamp(0.0, 1.0)
