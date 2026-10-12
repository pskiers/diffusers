"""paper_plots/plot_maze_local_vs_global_qualitative.py -- qualitative figure
showing DM failing to fix a globally-checkable mistake (add: a dead-end
detour that only reveals itself as wrong once you check where the real goal
is) while PaTh fixes it.

Images are pre-rendered and checked in at
runs/mazes_local_vs_global/add/ex_{1,2}/{dit_input_with_error,
dit_output_from_corrupted,trm_output_from_corrupted}.png -- this script just
lays them out (same imshow-panel convention as
plot_amaze_maze_appendix_examples.py). dit_input_with_error.png is the
shared corrupted starting board seen by both models (there is no separate
trm_input file -- both models denoise the same corrupted input).

Layout: one 2-example x 3-column (corrupted input / DM output / PaTh
output) grid. Each output panel gets a green (mistake fixed) or red
(mistake not fixed) border -- red for DM, green for PaTh.

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_maze_local_vs_global_qualitative.py
"""
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

COLOR_DM = "#2c2c2a"
COLOR_THINKER = "#8b3fd9"
BORDER_FIXED = "#2E7D32"
BORDER_UNFIXED = "#C62828"

EXAMPLES_DIR = "runs/mazes_local_vs_global"
MODE = "add"
EXAMPLES = ["ex_1", "ex_2"]
# (filename, column title, title color, key into FIXED -- None for the
# "Mistake" column, which is an input, not an output, so it gets no border)
COLUMNS = [
    ("dit_input_with_error.png", "Mistake", "black", None),
    ("dit_output_from_corrupted.png", "DM", COLOR_DM, "dit"),
    ("trm_output_from_corrupted.png", "PaTh", COLOR_THINKER, "trm"),
]
FIXED = {"dit": False, "trm": True}


def _load(example, fname):
    path = os.path.join(EXAMPLES_DIR, MODE, example, fname)
    return np.array(Image.open(path).convert("RGB"))


def _draw_panel(ax, img, title=None, title_color="black", border_color=None):
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), zorder=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3.5)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title is not None:
        ax.set_title(title, color=title_color, fontweight="bold")


def main():
    n_ex = len(EXAMPLES)
    n_col = len(COLUMNS)
    panel_in = 1.45
    col_gap_in = 0.06
    row_gap_in = 0.06

    fig_w = n_col * panel_in + (n_col - 1) * col_gap_in
    fig_h = n_ex * panel_in + (n_ex - 1) * row_gap_in + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        n_ex, n_col, wspace=col_gap_in / panel_in, hspace=row_gap_in / panel_in,
        left=0.03, right=0.99, top=0.90, bottom=0.02,
    )

    for row, example in enumerate(EXAMPLES):
        for col, (fname, col_title, color, key) in enumerate(COLUMNS):
            ax = fig.add_subplot(grid[row, col])
            title = col_title if row == 0 else None
            border_color = None if key is None else (BORDER_FIXED if FIXED[key] else BORDER_UNFIXED)
            _draw_panel(ax, _load(example, fname), title=title, title_color=color, border_color=border_color)

    fig.savefig("paper_plots/maze_local_vs_global_qualitative.pdf", bbox_inches="tight", pad_inches=0.06)
    print("Wrote paper_plots/maze_local_vs_global_qualitative.pdf")


if __name__ == "__main__":
    main()
