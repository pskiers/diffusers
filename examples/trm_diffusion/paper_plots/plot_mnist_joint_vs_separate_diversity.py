"""paper_plots/plot_mnist_joint_vs_separate_diversity.py — qualitative figure
showing that digits drawn by the joint model are visually less diverse than
those drawn by the separate (thinker + painter) model.

Inputs are the 3-panel sample images in runs/joint_vs_separate/
({joint,separate}_{6,7}.png: conditioning | generated grid | true grid). We
cut the middle panel into its 9x9 cells (16px each), drop the given cells
(non-empty in the conditioning panel), and group the generated cells by the
digit they depict. The labels are hand-transcribed from the generated
panels (the joint model's grids are not always valid sudokus, so the true
grid can't be used as labels). Cells labelled None are unreadable and are
skipped.

Figure: one row per model, one column per selected digit; each column is a
mosaic of that model's generated instances of the digit (chosen at random
with a fixed seed, not cherry-picked).

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_mnist_joint_vs_separate_diversity.py \\
      --out paper_plots/mnist_joint_vs_separate_diversity.pdf
"""
from __future__ import annotations

import argparse
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "joint_vs_separate")
CELL = 16
COND_X0, GEN_X0 = 0, 148

# Hand-transcribed digits of the generated (middle) panels, row-major.
LABELS = {
    "joint_6": [
        "458732196", "321596784", "796481523", "132947865", "675823419",
        "8491653?2", "584679231", "913258647", "267314958",
    ],
    "separate_6": [
        "258734196", "391256784", "674981523", "132547968", "567893412",
        "849162375", "485679231", "916325847", "723418659",
    ],
    "joint_7": [
        "921583624", "348672519", "672941853", "729436581", "586219437",
        "134758962", "413895276", "867324195", "295167348",
    ],
    "separate_7": [
        "921387654", "548692713", "673145892", "719463528", "856219437",
        "234578961", "465831279", "387926145", "192754386",
    ],
}
MODELS = {"joint": "Joint", "separate": "Separate"}


def load_cells(name: str, include_givens: bool = False) -> dict[int, list[np.ndarray]]:
    img = np.asarray(Image.open(os.path.join(RUN_DIR, f"{name}.png")).convert("L"), dtype=np.float32) / 255.0
    cond = img[:, COND_X0:COND_X0 + 9 * CELL]
    gen = img[:, GEN_X0:GEN_X0 + 9 * CELL]
    by_digit: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
    for r, c in itertools.product(range(9), range(9)):
        sl = (slice(r * CELL, (r + 1) * CELL), slice(c * CELL, (c + 1) * CELL))
        label = LABELS[name][r][c]
        if label == "?" or (not include_givens and cond[sl].max() > 60 / 255.0):
            continue  # unreadable sample, or given cell (redrawn by the model, but conditioned on a real digit)
        by_digit[int(label)].append(gen[sl])
    return by_digit


def gather(model: str, include_givens: bool = False) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
    for name in LABELS:
        if name.startswith(model + "_"):
            for d, cells in load_cells(name, include_givens).items():
                out[d].extend(cells)
    return out


def diversity(cells: list[np.ndarray]) -> float:
    """Mean pairwise L2 distance between intensity-normalised cells."""
    x = np.stack([c.ravel() / (np.linalg.norm(c) + 1e-8) for c in cells])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    n = len(cells)
    return float(d.sum() / (n * (n - 1)))


def mosaic(cells: list[np.ndarray], rows: int, cols: int, gap: int, rng) -> np.ndarray:
    n = rows * cols
    idx = rng.choice(len(cells), size=n, replace=False) if len(cells) >= n else np.arange(len(cells))
    canvas = np.ones((rows * CELL + (rows - 1) * gap, cols * CELL + (cols - 1) * gap), dtype=np.float32)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        canvas[r * (CELL + gap):r * (CELL + gap) + CELL, c * (CELL + gap):c * (CELL + gap) + CELL] = cells[i]
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_plots/mnist_joint_vs_separate_diversity.pdf")
    ap.add_argument("--digits", default="2,5,6,8,9")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_givens", action="store_true")
    ap.add_argument("--debug_montage", default=None, help="save every cell grouped by label, for label checking")
    args = ap.parse_args()

    data = {m: gather(m, args.include_givens) for m in MODELS}
    for d in range(1, 10):
        print(f"digit {d}: " + "  ".join(
            f"{m} n={len(data[m][d]):2d} div={diversity(data[m][d]):.3f}" for m in MODELS))

    if args.debug_montage:
        fig, axes = plt.subplots(18, 1, figsize=(12, 18))
        for ax, (d, m) in zip(axes, itertools.product(range(1, 10), MODELS)):
            ax.imshow(np.concatenate(data[m][d] + [np.ones((CELL, 2))] * 0, axis=1), cmap="gray", vmin=0, vmax=1)
            ax.set_ylabel(f"{m[0]}{d}", rotation=0, labelpad=12)
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(args.debug_montage, dpi=150, bbox_inches="tight")
        return

    digits = [int(d) for d in args.digits.split(",")]
    gap = 1
    fig, axes = plt.subplots(len(MODELS), len(digits), figsize=(1.0 * len(digits) + 0.5, 2.2),
                             gridspec_kw=dict(wspace=0.08, hspace=0.08))
    for i, (m, title) in enumerate(MODELS.items()):
        rng = np.random.default_rng(args.seed)
        for j, d in enumerate(digits):
            ax = axes[i, j]
            ax.imshow(mosaic(data[m][d], args.rows, args.cols, gap, rng), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
            if i == 0:
                ax.set_title(str(d), pad=3)
            if j == 0:
                ax.set_ylabel(title, labelpad=4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", args.out)


if __name__ == "__main__":
    main()
