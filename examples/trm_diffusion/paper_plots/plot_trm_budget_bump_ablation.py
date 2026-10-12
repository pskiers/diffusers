"""Gaussian reasoning-budget bump ablation: where is a fixed total reasoning
budget spent most efficiently, from experiments/ablate_trm_loop_budget.py's
"schedule" axis (+ablation.schedule_directions=[bump]).

Every (width, peak) cell reallocates the SAME total budget as the flat
(uniform n_sup) baseline at the same schedule_flat_n -- schedule_variants
always includes a "flat" entry alongside whatever schedule_directions are
swept. So delta-vs-flat, not raw accuracy, is the efficiency signal: it's
the only metric that isolates the effect of *where* the budget sits from
the effect of *how much* of it there is.

Diverging heatmap: red = reallocating hurts, blue = reallocating helps,
white/0 = no better than spending it uniformly. Row = width (how
concentrated the bump is, as std / trajectory length), column = peak
(where it's centered, 0=first/highest-noise step .. 1=last step).

Data transcribed directly from the user-provided sweep table; flat
baseline (66.1%) from the same run's "flat" schedule_variants entry.

Usage:
    python paper_plots/plot_trm_budget_bump_ablation.py
Produces paper_plots/trm_budget_bump_ablation.pdf
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

WIDTHS = [0.08, 0.2, 0.4]
PEAKS = [0.1, 0.3, 0.5, 0.7]
FLAT_ACC = 66.1

# Row order matches WIDTHS, column order matches PEAKS. Values in %.
ACC = np.array([
    [69.8, 59.9, 55.3, 55.1],
    [69.3, 66.2, 58.4, 54.8],
    [68.0, 66.7, 65.3, 62.8],
])

diff = ACC - FLAT_ACC
vabs = np.abs(diff).max()

fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.imshow(diff, cmap="RdBu", vmin=-vabs, vmax=vabs, aspect="auto")

ax.set_xticks(range(len(PEAKS)))
ax.set_xticklabels([str(p) for p in PEAKS])
ax.set_yticks(range(len(WIDTHS)))
ax.set_yticklabels([str(w) for w in WIDTHS])
ax.set_xlabel("Peak")
ax.set_ylabel("Width")
for spine in ax.spines.values():
    spine.set_visible(False)

for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        v = diff[i, j]
        text_color = "white" if abs(v) > 0.6 * vabs else "black"
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=15, color=text_color)

fig.tight_layout(pad=0.3)
fig.savefig("paper_plots/trm_budget_bump_ablation.pdf", bbox_inches="tight")
print("Wrote paper_plots/trm_budget_bump_ablation.pdf")
