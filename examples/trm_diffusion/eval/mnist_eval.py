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

import dataclasses
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = MNISTCellClassifier(ckpt.get("cell_size", cell_size)).to(device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded MNIST classifier from {path}")
    else:
        model = train_mnist_classifier(mnist_root, cell_size, path, device)
    return model.eval()


# ── Grid evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_grids(
    images:      torch.Tensor,               # (B, 1, H, W) float32 [0, 1]
    solutions:   torch.Tensor,               # (B, 81)      int64   [0-8]
    classifier:  MNISTCellClassifier,
    cell_size:   int,
    given_masks: torch.Tensor | None = None, # (B, 81) bool — True = given cell
) -> dict:
    """Classify every cell in *images* and compare to *solutions*.

    cell_acc   — accuracy on blank (inferred) cells only when given_masks supplied.
    puzzle_acc — accuracy requiring all 81 cells correct (given + blank).

    Returns {"cell_acc": float, "puzzle_acc": float, "preds": (B,81) cpu int64}.
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
    correct = preds == sol                           # (B, 81)

    # Puzzle accuracy: every cell must be correct.
    puzzle_acc = correct.all(dim=1).float().mean().item()

    # Cell accuracy: blank cells only.
    if given_masks is not None:
        blank   = ~given_masks.to(device)            # (B, 81) bool
        n_blank = blank.sum()
        cell_acc = (correct[blank].float().mean().item()
                    if n_blank > 0 else correct.float().mean().item())
    else:
        cell_acc = correct.float().mean().item()

    return {"cell_acc": cell_acc, "puzzle_acc": puzzle_acc, "preds": preds.cpu()}


# ── DDIM sampling ─────────────────────────────────────────────────────────────

def _build_denoising_schedule(
    num_train_timesteps: int,
    beta_schedule: str,
    prediction_type: str,
    num_steps: int,
    schedule_segments: list[str] | None = None,
):
    """Return a list of (timestep: int, DDIMScheduler) pairs for the denoising loop.

    With schedule_segments (e.g. ["10:1", "100:99"]):
      Each token "N:k" means: use the N-step DDIM schedule and take the next k
      timesteps from it (continuing from wherever the previous segment left off).
      Each segment gets its own DDIMScheduler so the correct prev_timestep step
      size (num_train_timesteps // N) is used for that slice.

    Without schedule_segments: standard single-schedule DDIM with num_steps.
    """
    from diffusers import DDIMScheduler

    def _make_ddim(n):
        s = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        s.set_timesteps(n)
        return s

    if not schedule_segments:
        ddim = _make_ddim(num_steps)
        return [(int(t), ddim) for t in ddim.timesteps]

    pairs: list[tuple[int, object]] = []
    last_t = float("inf")
    for seg in schedule_segments:
        n_total, n_take = (int(x) for x in seg.split(":"))
        ddim = _make_ddim(n_total)
        # Only take timesteps strictly below the last one already processed.
        ts = [int(t) for t in ddim.timesteps if int(t) < last_t][:n_take]
        for t in ts:
            pairs.append((t, ddim))
        if ts:
            last_t = ts[-1]
    return pairs


@torch.no_grad()
def sample_grids(
    model,
    base_sample,                                   # DataSample with static condition fields
    num_train_timesteps: int,
    beta_schedule:       str,
    num_steps:           int,
    device:              torch.device,
    prediction_type:     str                       = "epsilon",
    solutions:           torch.Tensor | None       = None,   # (B, 81) int64 [0-8]
    painter_size:        int | None               = None,
    given_masks:         torch.Tensor | None       = None,   # (B, 81) bool — True = given cell
    schedule_segments:   list[str] | None          = None,   # e.g. ["10:1", "100:99"]
    cfg_scale:           float                    = 1.0,
    noisy_guidance_fn    = None,  # Callable[[int, int], float] | None
                                  # (t, T) → s; blends pred = pred_v1 + s*(pred_v0 - pred_v1)
                                  # pred_v0 uses enc_x_noisy=zeros; painter still denoises real x_t
) -> dict:
    """DDIM-sample and collect thinker stats along the denoising trajectory.

    base_sample should contain static condition fields (e.g. spatial_conditions,
    puzzle_id).  x_noisy and timesteps are overwritten per step.

    Returns dict:
      'generated'                  : (B, 1, H, W) float32 [0, 1]
      'best_thinker_preds'         : (B, N) int64    — thinker argmax at most-confident step
      'best_thinker_ts'            : list[int] len B — timestep of that prediction
      'mean_thinker_preds'         : (B, N) int64    — plurality-vote across all trajectory steps
      'thinker_deviation_from_best': float           — mean fraction of cells that differ from
                                                       best_thinker_preds across the trajectory
      'ts_cell_acc'                : list[(t, float)] — per-denoising-step mean cell acc
      'ts_puzzle_acc'              : list[(t, float)] — per-denoising-step mean puzzle acc
      'thinker_cell_acc_best'      : float  — max cell acc over trajectory  (requires solutions)
      'thinker_cell_acc_mean'      : float  — mean cell acc over trajectory (requires solutions)
      'thinker_puzzle_acc_best'    : float  — max puzzle acc over trajectory
      'thinker_puzzle_acc_mean'    : float  — mean puzzle acc over trajectory
    All thinker/deviation entries are absent when the model returns no sudoku logits.
    The acc_best/mean entries also require solutions to be provided.
    """
    denoising_schedule = _build_denoising_schedule(
        num_train_timesteps, beta_schedule, prediction_type,
        num_steps, schedule_segments,
    )

    # Move static condition fields to device
    base_sample = dataclasses.replace(base_sample, **{
        f.name: getattr(base_sample, f.name).to(device)
        for f in dataclasses.fields(base_sample)
        if getattr(base_sample, f.name) is not None
    })
    B = next(
        getattr(base_sample, f.name).shape[0]
        for f in dataclasses.fields(base_sample)
        if getattr(base_sample, f.name) is not None
    )

    if solutions is not None:
        solutions = solutions.to(device)

    if painter_size is None:
        painter_size = getattr(getattr(model, "model_cfg", None), "painter_size", None)
    if painter_size is None:
        raise ValueError("painter_size is required (or set model.model_cfg.painter_size)")
    x = torch.randn(B, 1, painter_size, painter_size, device=device)

    token_offset = getattr(model, "token_offset", 0)

    has_logits   = False
    best_conf    = None
    best_preds   = None
    best_ts_vals = None
    N_logits     = None

    ts_cell_acc:    list[tuple[int, float]] = []
    ts_puzzle_acc:  list[tuple[int, float]] = []
    all_preds_list: list[torch.Tensor]      = []

    T = num_train_timesteps
    model.eval()
    for t, active_sched in tqdm(denoising_schedule):
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        step_sample = dataclasses.replace(base_sample, x_noisy=x, timesteps=ts)

        result = model(step_sample)
        noise_pred = result.pred
        sudoku_logits = result.logits

        # CFG blend
        if cfg_scale > 1.0:
            null_result = model(model.null_condition_sample(step_sample))
            noise_pred = null_result.pred + cfg_scale * (noise_pred - null_result.pred)

        # Noisy-image guidance: encoder sees zeros, painter still denoises real x_t
        if noisy_guidance_fn is not None:
            s = float(noisy_guidance_fn(int(t), T))
            if s != 0.0:
                clean_sample = dataclasses.replace(step_sample, enc_x_noisy=torch.zeros_like(x))
                clean_pred = model(clean_sample).pred
                if cfg_scale > 1.0:
                    clean_null_pred = model(model.null_condition_sample(clean_sample)).pred
                    clean_pred = clean_null_pred + cfg_scale * (clean_pred - clean_null_pred)
                noise_pred = noise_pred + s * (clean_pred - noise_pred)

        if sudoku_logits is not None:
            if not has_logits:
                has_logits   = True
                N_logits     = sudoku_logits.shape[1]
                best_conf    = torch.full((B,), -1e9, device=device)
                best_preds   = torch.zeros(B, N_logits, dtype=torch.long, device=device)
                best_ts_vals = torch.zeros(B, dtype=torch.long, device=device)

            preds = sudoku_logits.argmax(dim=-1)                              # (B, N)
            all_preds_list.append(preds.cpu())
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

            if solutions is not None and N_logits <= solutions.shape[1]:
                tgts    = solutions[:B, :N_logits] + token_offset
                correct = preds == tgts                        # (B, N)

                # Puzzle accuracy: all cells.
                puzz_a = correct.all(dim=1).float().mean().item()

                # Cell accuracy: blank cells only.
                if given_masks is not None:
                    gm      = given_masks[:B, :N_logits].to(device)  # (B, N)
                    blank   = ~gm
                    n_blank = blank.sum()
                    cell_a  = (correct[blank].float().mean().item()
                               if n_blank > 0 else correct.float().mean().item())
                else:
                    cell_a = correct.float().mean().item()

                ts_cell_acc.append((int(t),   cell_a))
                ts_puzzle_acc.append((int(t), puzz_a))

        x = active_sched.step(noise_pred, t, x).prev_sample

    result: dict = {"generated": x.clamp(0.0, 1.0)}
    if has_logits:
        best_preds_cpu = best_preds.cpu()
        result["best_thinker_preds"] = best_preds_cpu
        result["best_thinker_ts"]    = best_ts_vals.cpu().tolist()

        if all_preds_list:
            all_preds = torch.stack(all_preds_list, dim=0)  # (T, B, N) on CPU
            # Mean Hamming distance from the best-confidence prediction
            result["thinker_deviation_from_best"] = (
                (all_preds != best_preds_cpu.unsqueeze(0)).float().mean().item()
            )
            # Plurality vote across the trajectory
            result["mean_thinker_preds"] = torch.mode(all_preds, dim=0).values  # (B, N)

            if ts_cell_acc:
                cell_vals = [a for _, a in ts_cell_acc]
                puzz_vals = [a for _, a in ts_puzzle_acc]
                result["thinker_cell_acc_best"]   = float(max(cell_vals))
                result["thinker_cell_acc_mean"]   = float(np.mean(cell_vals))
                result["thinker_puzzle_acc_best"] = float(max(puzz_vals))
                result["thinker_puzzle_acc_mean"] = float(np.mean(puzz_vals))

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
