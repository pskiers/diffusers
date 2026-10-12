"""Mean violated sudoku constraints vs. sampling progress: PaTh vs. DM.

Data transcribed directly from user-provided run output (t=100, pure noise,
has no measurable grid yet, so the curve starts at t=95).

Usage:
    python paper_plots/plot_sudoku_violated_constraints_vs_timestep.py
Produces paper_plots/sudoku_violated_constraints_vs_timestep.pdf
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

COLOR_DM = "#2c2c2a"
COLOR_THINKER = "#8b3fd9"

T_STARTS = [100, 95, 85, 70, 55, 40, 25, 10, 0]
PCT_SAMPLING_PATH = [100 - t for t in T_STARTS]
PATH = [27.0, 6.199, 4.297, 2.750, 1.301, 0.857, 0.791, 0.938, 1.018]

PCT_SAMPLING_DM = [100 - t for t in T_STARTS]
DM = [27.0, 25.955, 23.490, 16.586, 12.500, 11.893, 11.764, 11.553, 10.598]

fig, ax = plt.subplots(figsize=(3.35, 2.55))

ax.plot(
    PCT_SAMPLING_PATH, PATH,
    marker="o", markersize=5, linewidth=2.4, color=COLOR_THINKER,
    label="PaTh", zorder=3,
)
ax.plot(
    PCT_SAMPLING_DM, DM,
    marker="s", markersize=5, linewidth=2.4, color=COLOR_DM,
    label="DM", zorder=3,
)

ax.set_xlabel("Sampling completed (%)")
ax.set_ylabel("Mean violated constraints")
ax.set_xticks(sorted(set(PCT_SAMPLING_PATH) | set(PCT_SAMPLING_DM)))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(frameon=False, loc="upper right")

fig.tight_layout(pad=0.3)
fig.savefig("paper_plots/sudoku_violated_constraints_vs_timestep.pdf", bbox_inches="tight")
print("Wrote paper_plots/sudoku_violated_constraints_vs_timestep.pdf")
