"""Maze (per shape) & Queens: Pass@1 vs. puzzle difficulty, PaTh vs. DM.

Shows the PaTh/DM accuracy gap widening as puzzles get larger/harder. Maze
is broken out per shape (square, hexagon, triangle, circle) rather than
averaged -- averaging blurred the trend since the shapes scale differently
(e.g. circle is much harder than square at the same size), so a per-shape
view shows the widening gap more clearly in each.

All 4 maze shapes and all 7 queens scales are included -- an earlier version
of this figure (built on a since-fixed buggy eval) excluded the triangle
shape and queens scale=4 as degenerate floor points where both models
scored ~0%. Under the corrected eval, neither is degenerate: triangle shows
a normal scaling trend for DM (99.6% -> 53% from size 5 to 16) with PaTh
staying near-ceiling almost throughout, and queens scale=4 is trivial for
both models (100%), not a floor. The OOD scale=12 point is still excluded
as this figure is about in-distribution difficulty scaling, not
generalization.

Data: runs/amaze_metrics_{maze,queens}_{dm,trm}.json

Usage:
    python paper_plots/plot_maze_queens_difficulty_curve.py
Produces paper_plots/maze_queens_difficulty_curve.pdf
"""
import json

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_DM = "#2c2c2a"
COLOR_THINKER = "#8b3fd9"

MAZE_SHAPES = ["square", "hexagon", "triangle", "circle"]
MAZE_SIZES = [5, 7, 8, 9, 11, 13, 16]
QUEENS_SCALES = [4, 5, 6, 7, 8, 9, 10]


def _load(path):
    with open(path) as f:
        return json.load(f)


def maze_shape_curve(data, shape):
    return [100.0 * data["per_shape"][shape][str(n)]["pass1"] for n in MAZE_SIZES]


def queens_curve(data):
    return [100.0 * data["per_scale"][str(n)]["pass1"] for n in QUEENS_SCALES]


maze_dm_raw = _load("runs/amaze_metrics_maze_dm.json")
maze_trm_raw = _load("runs/amaze_metrics_maze_trm.json")
queens_dm = queens_curve(_load("runs/amaze_metrics_queens_dm.json"))
queens_trm = queens_curve(_load("runs/amaze_metrics_queens_trm.json"))

PANELS = [
    (MAZE_SIZES, maze_shape_curve(maze_dm_raw, "square"), maze_shape_curve(maze_trm_raw, "square"),
     "Maze size", "Maze -- Square"),
    (MAZE_SIZES, maze_shape_curve(maze_dm_raw, "hexagon"), maze_shape_curve(maze_trm_raw, "hexagon"),
     "Maze size", "Maze -- Hexagon"),
    (MAZE_SIZES, maze_shape_curve(maze_dm_raw, "triangle"), maze_shape_curve(maze_trm_raw, "triangle"),
     "Maze size", "Maze -- Triangle"),
    (MAZE_SIZES, maze_shape_curve(maze_dm_raw, "circle"), maze_shape_curve(maze_trm_raw, "circle"),
     "Maze size", "Maze -- Circle"),
    (QUEENS_SCALES, queens_dm, queens_trm, "Board size ($N$-Queens)", "Queens"),
]

fig, axes = plt.subplots(1, 5, figsize=(11.8, 2.8), sharey=True)

for ax, x, dm, trm, xlabel, title in [(axes[i], *p) for i, p in enumerate(PANELS)]:
    ax.plot(x, dm, marker="o", markersize=4.5, linewidth=2.2, color=COLOR_DM, label="DM", zorder=3)
    ax.plot(x, trm, marker="o", markersize=4.5, linewidth=2.2, color=COLOR_THINKER, label="PaTh", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Pass@1 (%)")
axes[0].set_ylim(-3, 103)

fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0.0, 0.0, 1.0, 0.86))
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)
fig.savefig("paper_plots/maze_queens_difficulty_curve.pdf", bbox_inches="tight")
print("Wrote paper_plots/maze_queens_difficulty_curve.pdf")
