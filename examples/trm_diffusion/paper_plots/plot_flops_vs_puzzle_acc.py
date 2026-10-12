"""Puzzle accuracy vs. real FLOPs: 3 DM checkpoint sizes (each swept over
DDIM step count) + 2 TRM reasoning-budget policies (flat n_sup, and a
Gaussian-bump-schedule + carry + steering-freeze "wise" allocation matched
to the same total-budget grid).

The small (2M-param) DM checkpoint (runs/mnist_unet_painter.pt) is
deliberately excluded: it has no condition_encoder and in_channels=1 (no
room to concat a condition image, unlike the 20M/80M/200M models' concat
backbones) — verified structurally (models/painter_base.py's UNetPainter.
forward passes only x_noisy/timesteps to the UNet when no condition_encoder/
steering is given) and empirically (0% puzzle_acc at both 2 and 20 DDIM
steps). It's the TRM's frozen steered base, not a standalone puzzle solver,
so it isn't a meaningful "small DM" data point for this comparison.

Data comes from real sweep runs:
  - experiments/sweep_dm_flops_vs_acc.py  → runs/flops_sweep_dm_{20m,80m,200m}.json
  - experiments/sweep_trm_flops_vs_acc.py → runs/flops_sweep_trm_{flat,wise}.json

Usage:
    python paper_plots/plot_flops_vs_puzzle_acc.py
Produces paper_plots/flops_vs_puzzle_acc.pdf
"""
import json

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette, extended from the 2-color COLOR_DM/
# COLOR_THINKER convention used elsewhere in paper_plots/ to the 5 curves here.
COLOR_DM_20M = "#D55E00"
COLOR_DM_80M = "#CC79A7"
COLOR_DM_200M = "#999999"
COLOR_TRM_FLAT = "#56B4E9"
COLOR_TRM_WISE = "#0072B2"

CURVES = [
    {"path": "runs/flops_sweep_dm_20m.json", "kind": "dm", "label": "DM 20M", "color": COLOR_DM_20M, "marker": "*"},
    {"path": "runs/flops_sweep_dm_80m.json", "kind": "dm", "label": "DM 80M", "color": COLOR_DM_80M, "marker": "*"},
    {"path": "runs/flops_sweep_dm_200m.json", "kind": "dm", "label": "DM 200M", "color": COLOR_DM_200M, "marker": "*"},
    {"path": "runs/flops_sweep_trm_flat.json", "kind": "trm", "label": "PaTh", "color": COLOR_TRM_FLAT, "marker": "*"},
    {"path": "runs/flops_sweep_trm_wise.json", "kind": "trm", "label": "PaTh (efficient)", "color": COLOR_TRM_WISE, "marker": "*"},
]


def _load_points(path: str) -> tuple[list[float], list[float]]:
    with open(path) as f:
        data = json.load(f)
    points = sorted(data["points"], key=lambda p: p["flops"])
    flops = [p["flops"] for p in points]
    acc = [100.0 * p["puzzle_acc"] for p in points]
    return flops, acc


fig, ax = plt.subplots(figsize=(6.9, 3.4))

for c in CURVES:
    try:
        flops, acc = _load_points(c["path"])
    except FileNotFoundError:
        print(f"skipping {c['label']}: {c['path']} not found (run the corresponding sweep script first)")
        continue
    ax.plot(
        flops, acc, marker=c["marker"], markersize=0, linewidth=4.0, color=c["color"],
        label=c["label"], zorder=3,
    )

ax.set_xscale("log")
ax.set_xlabel("FLOPs")
ax.set_ylabel("Puzzle accuracy (%)")
ax.set_ylim(-2, 75)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(frameon=True, loc="best")

fig.tight_layout(pad=0.0)
fig.savefig("paper_plots/flops_vs_puzzle_acc.pdf", bbox_inches="tight")
print("Wrote paper_plots/flops_vs_puzzle_acc.pdf")
