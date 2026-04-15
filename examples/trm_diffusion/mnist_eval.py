"""
mnist_eval.py – Digit-level evaluation for MNIST Sudoku models.

Provides:
  MNISTCellClassifier      – small CNN classifying a single cell → digit 1-9.
  train_mnist_classifier   – trains and saves the classifier.
  load_or_train_classifier – loads or trains on demand.
  evaluate_grids           – classifies cells in generated images, returns cell/puzzle acc.
  sample_grids             – DDIM sampling with thinker trajectory tracking.
  make_panel_image         – composites puzzle | [thinker] | model output | true solution.
  plot_thinker_ts_curve    – plots thinker accuracy vs denoising timestep (mean ± std).
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
    puzzle_ids:          torch.Tensor | None = None,   # (B,) long
    solutions:           torch.Tensor | None = None,   # (B, 81) int64 [0-8]
) -> dict:
    """DDIM-sample and collect thinker stats along the denoising trajectory.

    Returns dict:
      'generated'          : (B, 1, H, W) float32 [0, 1]
      'best_thinker_preds' : (B, N) int64    — thinker argmax at most-confident step
      'best_thinker_ts'    : list[int] len B — timestep of that prediction
      'ts_cell_acc'        : list[(t, float)] — per-denoising-step mean cell acc
      'ts_puzzle_acc'      : list[(t, float)] — per-denoising-step mean puzzle acc
    The thinker and timestep entries are only present when the model returns sudoku logits.
    The ts_* entries also require solutions to be provided.
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
    if solutions is not None:
        solutions = solutions.to(device)
    B = conditions.shape[0]
    x = torch.randn_like(conditions)

    # Lazy-initialised on first encounter of sudoku_logits
    has_logits   = False
    best_conf    = None
    best_preds   = None
    best_ts_vals = None
    N_logits     = None

    ts_cell_acc:   list[tuple[int, float]] = []
    ts_puzzle_acc: list[tuple[int, float]] = []

    model.eval()
    for t in ddim.timesteps:
        ts         = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred, sudoku_logits = model(x, ts, conditions, puzzle_ids=puzzle_ids)

        if sudoku_logits is not None:
            if not has_logits:
                has_logits   = True
                N_logits     = sudoku_logits.shape[1]
                best_conf    = torch.full((B,), -1e9, device=device)
                best_preds   = torch.zeros(B, N_logits, dtype=torch.long, device=device)
                best_ts_vals = torch.zeros(B, dtype=torch.long, device=device)

            preds = sudoku_logits.argmax(dim=-1)                              # (B, N)
            probs = torch.softmax(sudoku_logits.float(), dim=-1)              # (B, N, C)
            conf  = probs.max(dim=-1).values.mean(dim=-1)                     # (B,)

            update       = conf > best_conf
            best_conf    = torch.where(update, conf, best_conf)
            best_preds   = torch.where(
                update.unsqueeze(-1).expand_as(best_preds), preds, best_preds
            )
            best_ts_vals = torch.where(
                update,
                torch.full((B,), int(t), device=device, dtype=torch.long),
                best_ts_vals,
            )

            if solutions is not None:
                tgts    = solutions[:B, :N_logits]
                correct = preds == tgts
                ts_cell_acc.append((int(t),   correct.float().mean().item()))
                ts_puzzle_acc.append((int(t), correct.all(dim=1).float().mean().item()))

        x = ddim.step(noise_pred, t, x).prev_sample

    result: dict = {"generated": x.clamp(0.0, 1.0)}
    if has_logits:
        result["best_thinker_preds"] = best_preds.cpu()
        result["best_thinker_ts"]    = best_ts_vals.cpu().tolist()
    if ts_cell_acc:
        result["ts_cell_acc"]   = ts_cell_acc
        result["ts_puzzle_acc"] = ts_puzzle_acc
    return result


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _sudoku_grid_img(digits_0idx: np.ndarray, label: str = "", size: int = 144) -> np.ndarray:
    """Render a 9×9 sudoku digit grid as a (size, size, 3) uint8 RGB numpy array.

    digits_0idx: (81,) int array, values 0-8 (displayed as 1-9).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as _PILImage

    fig, ax = plt.subplots(figsize=(3, 3), dpi=max(1, size // 3))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(10):
        lw = 1.5 if i % 3 == 0 else 0.4
        ax.axhline(i, color="black", linewidth=lw)
        ax.axvline(i, color="black", linewidth=lw)
    grid = np.asarray(digits_0idx, dtype=int).reshape(9, 9)
    fs = max(5, size // 18)
    for r in range(9):
        for c in range(9):
            ax.text(c + 0.5, 8.5 - r, str(grid[r, c] + 1),
                    ha="center", va="center", fontsize=fs, color="black")
    top = 0.88 if label else 1.0
    if label:
        ax.set_title(label, fontsize=max(5, size // 20), pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.01)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[..., :3]
    img = np.array(_PILImage.fromarray(buf).resize((size, size), _PILImage.LANCZOS))
    plt.close(fig)
    return img


def _mnist_to_rgb(t: torch.Tensor, size: int = 144) -> np.ndarray:
    """(1, H, W) float [0,1] grayscale tensor → (size, size, 3) uint8 RGB."""
    from PIL import Image as _PILImage
    arr = t.squeeze(0).clamp(0, 1).cpu().float().numpy()
    arr = (arr * 255).astype(np.uint8)
    img = _PILImage.fromarray(arr, mode="L").resize((size, size), _PILImage.LANCZOS)
    return np.stack([np.array(img)] * 3, axis=-1)


def make_panel_image(
    condition:     torch.Tensor,               # (1, H, W) float [0,1]  – puzzle shown to model
    generated:     torch.Tensor,               # (1, H, W) float [0,1]  – model output
    solution:      np.ndarray,                 # (81,) int [0-8]         – true solution
    thinker_preds: np.ndarray | None = None,   # (81,) int [0-8]         – thinker prediction
    thinker_t:     int | None       = None,    # denoising timestep label
    img_size:      int              = 144,
) -> np.ndarray:
    """Build a horizontal panel image for visual evaluation.

    Layout (width depends on whether thinker is present):
      puzzle  |  [thinker solution]  |  model output  |  true solution

    Returns (img_size, width, 3) uint8 RGB numpy array.
    """
    sep = np.full((img_size, 4, 3), 200, dtype=np.uint8)   # light-gray 4-px separator
    parts = [_mnist_to_rgb(condition, img_size)]
    if thinker_preds is not None:
        label = f"thinker t={thinker_t}" if thinker_t is not None else "thinker"
        parts += [sep, _sudoku_grid_img(thinker_preds, label=label, size=img_size)]
    parts += [sep, _mnist_to_rgb(generated, img_size)]
    parts += [sep, _sudoku_grid_img(solution, label="true", size=img_size)]
    return np.concatenate(parts, axis=1)


def plot_thinker_ts_curve(
    ts_cell_acc:   dict,   # {t_int: [batch_mean, ...]}
    ts_puzzle_acc: dict,   # {t_int: [batch_mean, ...]}
) -> np.ndarray:
    """Plot thinker cell/puzzle accuracy vs denoising timestep (mean ± std bands).

    ts_cell_acc / ts_puzzle_acc: dicts mapping timestep → list of per-batch means.
    Returns (H, W, 3) uint8 RGB numpy array suitable for wandb.Image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_sorted  = sorted(ts_cell_acc.keys(), reverse=True)   # high noise → low noise
    ta         = np.array(ts_sorted, dtype=float)
    cell_mean  = np.array([np.mean(ts_cell_acc[t])   for t in ts_sorted])
    cell_std   = np.array([np.std(ts_cell_acc[t])    for t in ts_sorted])
    puzz_mean  = np.array([np.mean(ts_puzzle_acc[t]) for t in ts_sorted])
    puzz_std   = np.array([np.std(ts_puzzle_acc[t])  for t in ts_sorted])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ta, cell_mean, color="tab:blue",   label="cell acc")
    ax.fill_between(ta, cell_mean - cell_std, cell_mean + cell_std,
                    alpha=0.25, color="tab:blue")
    ax.plot(ta, puzz_mean, color="tab:orange", label="puzzle acc")
    ax.fill_between(ta, puzz_mean - puzz_std, puzz_mean + puzz_std,
                    alpha=0.25, color="tab:orange")
    ax.invert_xaxis()   # left = high noise, right = clean
    ax.set_xlabel("timestep  (← denoising direction)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Thinker accuracy along denoising trajectory")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return buf
