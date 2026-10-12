"""paper_plots/plot_amaze_maze_appendix_examples.py — appendix figure showing
one example puzzle/solution pair per maze shape (square, hexagon, triangle,
circle) from the AMAZE benchmark.

There's no dataset class to draw from here -- the AMAZE examples are
pre-rendered images checked in at runs/examples_amaze/*.webp, so this script
just loads and lays them out (same single-row gallery layout as
plot_mnist_sudoku_appendix_examples.py). Each shape's native aspect ratio is
preserved rather than stretched to a uniform square (square/circle are
square, hexagon is wide, triangle is a tall narrow strip).

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_amaze_maze_appendix_examples.py \\
      --out paper_plots/amaze_maze_appendix_examples.pdf
"""
from __future__ import annotations

import argparse
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
    "axes.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EXAMPLES_DIR = "runs/examples_amaze"

# (display label, puzzle filename stem, solution filename stem)
MAZE_SHAPES = [
    ("Square", "maze_sq_puzzle", "maze_sq_solution"),
    ("Hexagon", "maze_hex_puzzle", "maze_hex_solution"),
    ("Triangle", "maze_tr_puzzle", "maze_tr_solution"),
    ("Circle", "maze_circle_puzzle", "maze_circle_solution"),
]


def _load(examples_dir: str, stem: str) -> np.ndarray:
    path = os.path.join(examples_dir, f"{stem}.webp")
    return np.array(Image.open(path).convert("RGB"))


def _draw_panel(ax, img: np.ndarray, title: str) -> None:
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), zorder=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default=EXAMPLES_DIR)
    ap.add_argument("--out", default="paper_plots/amaze_maze_appendix_examples.pdf")
    args = ap.parse_args()

    n = len(MAZE_SHAPES)
    panel_in = 1.55
    pair_gap_in = 0.14   # gap between puzzle and solution within a pair
    group_gap_in = 0.35  # gap between consecutive shapes
    fig_w = n * (2 * panel_in + pair_gap_in) + (n - 1) * group_gap_in
    fig_h = panel_in + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        1, n, wspace=group_gap_in / panel_in, left=0, right=1, top=0.82, bottom=0.02,
    )

    for col, (label, puzzle_stem, solution_stem) in enumerate(MAZE_SHAPES):
        pair = outer[0, col].subgridspec(1, 2, wspace=pair_gap_in / panel_in)
        ax_puzzle = fig.add_subplot(pair[0, 0])
        ax_solution = fig.add_subplot(pair[0, 1])
        _draw_panel(ax_puzzle, _load(args.examples_dir, puzzle_stem), f"{label} -- Puzzle")
        _draw_panel(ax_solution, _load(args.examples_dir, solution_stem), f"{label} -- Solution")

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.06)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
