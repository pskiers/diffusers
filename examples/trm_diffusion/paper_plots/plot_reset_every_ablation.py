"""Reset-every ablation: puzzle accuracy as a function of how often the
thinker's carried state (z scratchpad) is reset, at several reasoning budgets
(n_sup). Shows that resetting state hurts, and hurts much more when the
reasoning budget is small -- at n_sup=16 the choice of reset_every barely
matters, at n_sup=1 it is the difference between 56% and 33%.

Data transcribed directly from the user-provided sweep table.

Usage:
    python paper_plots/plot_reset_every_ablation.py
Produces paper_plots/reset_every_ablation.pdf
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RESET_EVERY = [20, 5, 2, 1]
N_SUP = [1, 2, 4, 8, 16]

# Row order matches N_SUP, column order matches RESET_EVERY. Values in %.
ACC = {
    1: [56.4, 55.8, 48.8, 32.9],
    2: [61.8, 61.9, 60.1, 56.0],
    4: [65.8, 65.7, 65.3, 64.3],
    8: [68.7, 68.6, 68.4, 68.1],
    16: [70.7, 70.6, 70.5, 70.2],
}

# Sequential single-hue ramp (light -> dark), light = reset rarely (reset_every
# large), dark = reset every step, so the eye reads "reset aggressiveness" as
# one ordered quantity.
cmap = plt.get_cmap("Blues")
COLORS = {r: cmap(t) for r, t in zip(RESET_EVERY, np.linspace(0.35, 0.95, len(RESET_EVERY)))}

fig, ax = plt.subplots(figsize=(6.9, 4.4))

x = np.arange(len(N_SUP))
for i, r in enumerate(RESET_EVERY):
    y = [ACC[n][i] for n in N_SUP]
    ax.plot(
        x, y, marker="o", markersize=8, linewidth=2.5, color=COLORS[r],
        label=f"reset every {r}", zorder=3,
    )

ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in N_SUP])
ax.set_xlabel("Reasoning budget (n_sup)")
ax.set_ylabel("Puzzle accuracy (%)")
ax.set_ylim(28, 75)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(frameon=True, loc="lower right")

fig.tight_layout(pad=0.0)
fig.savefig("paper_plots/reset_every_ablation.pdf", bbox_inches="tight")
print("Wrote paper_plots/reset_every_ablation.pdf")
