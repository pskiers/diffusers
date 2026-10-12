"""Sudoku-extreme corruption-recovery: PaTh vs. DM heatmaps + LaTeX tables.

PaTh data is read directly from the real probe output
(runs/full_ablation/violation_sensitivity_full/easy_{condition}_n{N}_f{F}.json,
from experiments/violation_sensitivity_probe.py, checkpoint sudoku-painter-thinker.pt).

DM data for the same sweep was never found on disk (exhaustively searched --
neither sudoku_unet_concat_t100-30k.pt nor t1000-30k.pt's saved
concat_violation_sensitivity/ results match) and is transcribed here directly
from the user-provided run output instead.

Produces:
  - paper_plots/sudoku_violation_sensitivity_heatmaps.pdf: PaTh vs DM,
    recovered_and_valid_rate, non_violating only, rows = % of sampling
    completed when the mistake is injected (10,30,50,70,90), columns =
    number of corrupted cells (1,2,4,8,16,32,64).
  - paper_plots/sudoku_violation_sensitivity_tables.tex: 4 LaTeX tables
    (recovered x violating/non_violating, kept x violating/non_violating),
    PaTh and DM side by side in each.

Usage:
    python paper_plots/plot_sudoku_violation_sensitivity.py
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

COLOR_DM = "#D55E00"
COLOR_THINKER = "#0072B2"

N_CELLS = [1, 2, 4, 8, 16, 32, 64]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
PCT = [10, 30, 50, 70, 90]
FRAC_TO_IDX = {0.1: 2, 0.3: 6, 0.5: 10, 0.7: 13, 0.9: 17}
CONDITIONS = ["violating", "non_violating"]

THINKER_DIR = "runs/full_ablation/violation_sensitivity_full"

# DM (concat) data manually transcribed from the user-provided run output --
# see module docstring. Row order = PCT, column order = N_CELLS. Values in %.
DM_PLAIN = {
    "violating": {
        "recovered": [
            [49.2, 50.0, 49.6, 52.2, 50.5, 49.0, 44.7],
            [85.9, 88.3, 86.9, 85.0, 77.6, 58.8, 42.4],
            [97.7, 96.9, 96.5, 95.8, 92.0, 74.2, 36.8],
            [77.3, 72.7, 77.9, 70.6, 57.7, 32.3, 12.1],
            [10.9, 7.0, 8.2, 6.9, 6.3, 3.6, 1.9],
        ],
        "kept": [
            [0.0, 0.0, 0.2, 0.2, 0.3, 0.2, 0.3],
            [0.0, 0.0, 1.4, 0.0, 0.4, 0.5, 0.5],
            [1.6, 0.0, 1.2, 0.9, 1.0, 1.5, 2.7],
            [14.1, 22.7, 15.0, 20.9, 28.4, 38.1, 45.4],
            [86.7, 88.3, 87.1, 89.4, 88.4, 90.3, 90.2],
        ],
    },
    "non_violating": {
        "recovered": [
            [48.4, 51.2, 55.9, 50.6, 49.1, 45.9, 40.9],
            [89.0, 90.5, 82.2, 80.9, 67.2, 42.8, 30.6],
            [95.3, 95.3, 93.7, 90.3, 78.9, 39.6, 19.9],
            [76.8, 74.1, 70.1, 61.4, 43.0, 17.8, 8.0],
            [7.1, 12.7, 5.7, 7.1, 4.7, 2.2, 1.1],
        ],
        "kept": [
            [15.1, 16.5, 19.6, 18.5, 18.4, 20.6, 21.7],
            [5.5, 4.3, 6.9, 9.1, 13.8, 24.7, 31.4],
            [2.4, 2.8, 3.9, 6.0, 12.6, 33.9, 43.9],
            [17.6, 19.6, 23.4, 28.3, 43.1, 60.2, 67.2],
            [88.2, 82.1, 90.2, 87.1, 91.0, 93.6, 94.4],
        ],
    },
}

DM_VALID = {
    "violating": [
        [1.6, 4.7, 6.2, 4.4, 3.1, 1.6, 2.3],
        [56.2, 48.0, 38.3, 35.2, 25.0, 7.0, 0.9],
        [72.7, 77.3, 57.8, 52.3, 32.8, 4.7, 0.0],
        [54.7, 39.1, 29.7, 8.6, 0.0, 0.0, 0.0],
        [8.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    "non_violating": [
        [4.0, 2.0, 6.3, 4.3, 4.0, 3.6, 1.3],
        [52.0, 50.6, 41.2, 32.5, 13.2, 0.5, 0.0],
        [70.9, 71.7, 48.2, 42.8, 14.4, 0.0, 0.0],
        [58.4, 41.2, 23.0, 3.8, 0.0, 0.0, 0.0],
        [0.0, 4.7, 2.8, 0.0, 0.0, 0.0, 0.0],
    ],
}


def _load_thinker(condition, metric):
    """metric in {'recovered', 'kept', 'recovered_valid'}. Returns a
    (len(PCT), len(N_CELLS)) array of percentages."""
    field = {
        "recovered": "recovery_rate",
        "kept": "adapt_rate",
        "recovered_valid": "recovered_and_valid_rate",
    }[metric]
    mat = np.zeros((len(PCT), len(N_CELLS)))
    for i, f in enumerate(FRACS):
        for j, n in enumerate(N_CELLS):
            path = f"{THINKER_DIR}/easy_{condition}_n{n}_f{f}.json"
            with open(path) as fh:
                d = json.load(fh)
            idx = FRAC_TO_IDX[f]
            key = f"{condition}/n_cells={n}/corrupt_idx={idx}"
            mat[i, j] = d["results"][key][field] * 100
    return mat


# ── Heatmaps (non_violating, recovered+valid) ───────────────────────────────


def make_heatmaps():
    th_mat = _load_thinker("non_violating", "recovered_valid")
    dm_mat = np.array(DM_VALID["non_violating"])
    vmax = max(th_mat.max(), dm_mat.max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, mat, title, color in [
        (axes[0], th_mat, "PaTh", COLOR_THINKER),
        (axes[1], dm_mat, "DM", COLOR_DM),
    ]:
        ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(N_CELLS)))
        ax.set_xticklabels([str(n) for n in N_CELLS])
        ax.set_xlabel("Number of corrupted cells")
        ax.set_title(title, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                text_color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color=text_color)

    axes[0].set_yticks(range(len(PCT)))
    axes[0].set_yticklabels([f"{p}%" for p in PCT])
    axes[0].set_ylabel("Mistake simulated at % of sampling")

    fig.savefig("paper_plots/sudoku_violation_sensitivity_heatmaps.pdf", bbox_inches="tight")
    print("Wrote paper_plots/sudoku_violation_sensitivity_heatmaps.pdf")


# ── Diff heatmap: kept(non_violating) - kept(violating) ─────────────────────
#
# A model doing real constraint-checking should be MORE likely to keep
# (fail to fix) a non_violating mistake than a violating one -- violating
# digits directly contradict a given clue (locally checkable), non_violating
# ones only reveal themselves through full puzzle-solving. So a positive
# value here (kept_non_violating > kept_violating) is the signature of real
# constraint-sensitivity; near-zero or negative means the model treats both
# kinds of mistake the same way (pattern-matching, not actually checking).


def make_kept_diff_heatmap():
    th_diff = _load_thinker("non_violating", "kept") - _load_thinker("violating", "kept")
    dm_diff = np.array(DM_PLAIN["non_violating"]["kept"]) - np.array(DM_PLAIN["violating"]["kept"])
    vabs = max(np.abs(th_diff).max(), np.abs(dm_diff).max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    cmap = plt.get_cmap("RdBu_r")

    for ax, mat, title, color in [
        (axes[0], th_diff, "PaTh", COLOR_THINKER),
        (axes[1], dm_diff, "DM", COLOR_DM),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
        ax.set_xticks(range(len(N_CELLS)))
        ax.set_xticklabels([str(n) for n in N_CELLS])
        ax.set_xlabel("Number of corrupted cells")
        ax.set_title(title, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                text_color = "white" if abs(v) > 0.6 * vabs else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5, color=text_color)

    axes[0].set_yticks(range(len(PCT)))
    axes[0].set_yticklabels([f"{p}%" for p in PCT])
    axes[0].set_ylabel("Mistake simulated at % of sampling")

    fig.savefig("paper_plots/sudoku_kept_diff_heatmaps.pdf", bbox_inches="tight")
    print("Wrote paper_plots/sudoku_kept_diff_heatmaps.pdf")


# ── LaTeX tables (recovered / kept, x violating / non_violating) ───────────


def _fmt(v):
    return f"{v:.1f}"


def _table(th_mat, dm_mat, caption, label):
    ncol = len(N_CELLS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{c" + "c" * ncol * 2 + "}",
        r"\toprule",
        r" & \multicolumn{%d}{c}{PaTh} & \multicolumn{%d}{c}{DM} \\" % (ncol, ncol),
        rf"\cmidrule(lr){{2-{1 + ncol}}} \cmidrule(lr){{{2 + ncol}-{1 + 2 * ncol}}}",
    ]
    header = [r"\% sampling"] + [f"$n{{=}}{n}$" for n in N_CELLS] * 2
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for i, p in enumerate(PCT):
        row = [str(p)] + [_fmt(v) for v in th_mat[i]] + [_fmt(v) for v in dm_mat[i]]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_tables():
    tables = []
    for condition in CONDITIONS:
        th_mat = _load_thinker(condition, "recovered_valid")
        dm_mat = np.array(DM_VALID[condition])
        cond_label = "violating" if condition == "violating" else "non-violating"
        caption = (
            f"Recovered-and-valid rate (\\%) for {cond_label} corruptions, as a function "
            "of \\% of sampling completed when the mistake is injected and number of corrupted cells."
        )
        label = f"sudoku_recovered_valid_{condition}"
        tables.append(_table(th_mat, dm_mat, caption, label))
    for condition in CONDITIONS:
        th_mat = _load_thinker(condition, "kept")
        dm_mat = np.array(DM_PLAIN[condition]["kept"])
        cond_label = "violating" if condition == "violating" else "non-violating"
        caption = (
            f"Kept rate (\\%) for {cond_label} corruptions, as a function "
            "of \\% of sampling completed when the mistake is injected and number of corrupted cells."
        )
        label = f"sudoku_kept_{condition}"
        tables.append(_table(th_mat, dm_mat, caption, label))
    out = "\n\n".join(tables) + "\n"
    with open("paper_plots/sudoku_violation_sensitivity_tables.tex", "w") as f:
        f.write(out)
    print("Wrote paper_plots/sudoku_violation_sensitivity_tables.tex")
    print("\n" + out)


def main():
    make_heatmaps()
    make_kept_diff_heatmap()
    make_tables()


if __name__ == "__main__":
    main()
