"""paper_plots/plot_bagel_samples.py — gallery of BAGEL-generated samples:
one column per task (square/hexagon/triangle/circle mazes, then Queens), two
samples stacked per column.

Samples are pre-rendered images at runs/bagel_samples/*.jpg. Each column's
width is set from its images' native aspect ratio (all rows share one height),
so the wide hexagon and narrow triangle mazes are not stretched or padded.

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_bagel_samples.py --out paper_plots/bagel_samples.pdf
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

SAMPLES_DIR = "runs/bagel_samples"

# (column title, [top image, bottom image])
COLUMNS = [
    ("Maze (Square)", ["square2.jpg", "square_11.jpg"]),
    ("Maze (Hexagon)", ["hexagon2.jpg", "hexagon_16.jpg"]),
    ("Maze (Triangle)", ["triangle1.jpg", "triangle2.jpg"]),
    ("Maze (Circle)", ["circle2.jpg", "circle_5.jpg"]),
    ("Queens", ["queens1.jpg", "queens2.jpg"]),
]


def _load(samples_dir: str, name: str) -> np.ndarray:
    return np.array(Image.open(os.path.join(samples_dir, name)).convert("RGB"))


def _draw_panel(ax, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), interpolation="lanczos")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    ap.add_argument("--out", default="paper_plots/bagel_samples.pdf")
    args = ap.parse_args()

    imgs = [[_load(args.samples_dir, n) for n in names] for _, names in COLUMNS]
    # Column width relative to row height: the widest image in that column.
    aspects = [max(im.shape[1] / im.shape[0] for im in col) for col in imgs]

    panel_h = 1.6   # inches per row
    col_gap = 0.15  # inches between columns
    row_gap = 0.12  # inches between rows
    title_h = 0.3
    fig_w = panel_h * sum(aspects) + col_gap * (len(COLUMNS) - 1)
    fig_h = 2 * panel_h + row_gap + title_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, len(COLUMNS),
        width_ratios=aspects,
        wspace=col_gap / (panel_h * np.mean(aspects)),
        hspace=row_gap / panel_h,
        left=0, right=1, bottom=0, top=1 - title_h / fig_h,
    )

    for c, ((title, _), col_imgs) in enumerate(zip(COLUMNS, imgs)):
        for r, img in enumerate(col_imgs):
            ax = fig.add_subplot(gs[r, c])
            _draw_panel(ax, img)
            if r == 0:
                ax.set_title(title, fontsize=11, pad=4)

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
