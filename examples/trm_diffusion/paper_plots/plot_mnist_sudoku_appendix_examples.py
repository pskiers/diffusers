"""paper_plots/plot_mnist_sudoku_appendix_examples.py — appendix figure
showing N puzzle/solution pairs for the MNIST-Sudoku task: the conditioning
(given cells rendered as real MNIST digits, blanks black) next to the full
solved grid (also rendered as real MNIST digits). Uses the real
MNISTSudokuDataset — no synthetic digits, no reimplementation of the grid
renderer.

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_mnist_sudoku_appendix_examples.py \\
      --out paper_plots/mnist_sudoku_appendix_examples.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_BOX_LINE = "#555555"
COLOR_CELL_LINE = "#cfcfcf"


def _draw_grid_lines(ax, size: int) -> None:
    """3x3-box lines (dark) over cell lines (light), matching this repo's
    other sudoku figures (paper_plots/plot_qualitative_dip_example.py)."""
    cell = size / 9.0
    for i in range(10):
        pos = i * cell
        lw = 1.3 if i % 3 == 0 else 0.5
        color = COLOR_BOX_LINE if i % 3 == 0 else COLOR_CELL_LINE
        ax.plot([pos, pos], [0, size], color=color, linewidth=lw, zorder=2)
        ax.plot([0, size], [pos, pos], color=color, linewidth=lw, zorder=2)


def _draw_panel(ax, img: np.ndarray, title: str) -> None:
    size = img.shape[0]
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, extent=(0, size, size, 0), zorder=1)
    _draw_grid_lines(ax, size)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sudoku-dir", default="data/sudoku-extreme-1k-aug-1000/test")
    ap.add_argument("--mnist-root", default="data/mnist")
    ap.add_argument("--n", type=int, default=3, help="number of puzzle/solution pairs to show")
    ap.add_argument("--cell-size", type=int, default=28, help="pixels per digit cell (28 = native MNIST resolution)")
    ap.add_argument("--seed", type=int, default=0, help="MNIST-instance-per-cell RNG seed")
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="explicit puzzle indices; overrides --n")
    ap.add_argument("--out", default="paper_plots/mnist_sudoku_appendix_examples.pdf")
    args = ap.parse_args()

    ds = MNISTSudokuDataset(
        sudoku_dir=args.sudoku_dir,
        mnist_root=args.mnist_root,
        cell_size=args.cell_size,
        mnist_split="test",
        mask_given=True,
        seed=args.seed,
        same_digit_images=True,
    )

    if args.indices is not None:
        indices = args.indices
    else:
        # Evenly spaced through the split so the examples aren't all near
        # each other / all the same difficulty pocket.
        indices = list(np.linspace(0, len(ds) - 1, args.n, dtype=int))

    # Horizontal layout: n (puzzle, solution) pairs side by side in a single
    # row, so the whole figure is a short wide strip that fits a single
    # figure slot in a single-column paper rather than a tall stack.
    n = len(indices)
    panel_in = 1.55
    pair_gap_in = 0.14   # gap between puzzle and solution within a pair
    group_gap_in = 0.30  # gap between consecutive pairs
    fig_w = n * (2 * panel_in + pair_gap_in) + (n - 1) * group_gap_in
    fig_h = panel_in + 0.35

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        1, n, wspace=group_gap_in / panel_in, left=0, right=1, top=0.86, bottom=0.02,
    )

    for col, idx in enumerate(indices):
        sample = ds[idx]
        puzzle_img = sample.spatial_conditions[0].numpy()
        solution_img = sample.images[0].numpy()

        pair = outer[0, col].subgridspec(1, 2, wspace=pair_gap_in / panel_in)
        ax_puzzle = fig.add_subplot(pair[0, 0])
        ax_solution = fig.add_subplot(pair[0, 1])
        _draw_panel(ax_puzzle, puzzle_img, "Puzzle")
        _draw_panel(ax_solution, solution_img, "Solution")

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.06)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
