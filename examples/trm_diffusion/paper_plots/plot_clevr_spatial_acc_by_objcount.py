"""Spatial accuracy vs. object count: standalone painter (DM) vs. painter+thinker.

Usage:
    python paper_plots/plot_clevr_spatial_acc_by_objcount.py
Produces paper_plots/clevr_spatial_acc_by_objcount.pdf
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OBJECT_COUNTS = list(range(3, 11))

DM = [
    0.9301848049281314, 0.884437596302003, 0.8278791036998436, 0.7677479855515421,
    0.7166414904330313, 0.6733536514794954, 0.6408758459530819, 0.6203003816324019,
]
PAINTER_THINKER = [
    0.9621315969842358, 0.9578178694158076, 0.9446071764828957, 0.9244830263250758,
    0.9081611781550419, 0.8856159292727771, 0.8714876933130037, 0.8524223476183214,
]

COLOR_DM = "#2c2c2a"
COLOR_THINKER = "#8b3fd9"

fig, ax = plt.subplots(figsize=(3.35, 2.55))

ax.plot(
    OBJECT_COUNTS, PAINTER_THINKER,
    marker="o", markersize=5, linewidth=2.4, color=COLOR_THINKER,
    label="PaTh", zorder=3,
)
ax.plot(
    OBJECT_COUNTS, DM,
    marker="s", markersize=5, linewidth=2.4, color=COLOR_DM,
    label="DM", zorder=3,
)

ax.set_xlabel("Number of objects")
ax.set_ylabel("Spatial accuracy")
ax.set_xticks(OBJECT_COUNTS)
ax.set_ylim(0.55, 1.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(frameon=False, loc="lower left")

fig.tight_layout(pad=0.3)
fig.savefig("paper_plots/clevr_spatial_acc_by_objcount.pdf", bbox_inches="tight")
