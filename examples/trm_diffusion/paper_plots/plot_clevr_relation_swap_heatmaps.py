"""Relation-swap probe: PaTh vs. DM heatmaps + LaTeX tables (cycle_length=2).

Reads the 7-object-controlled relation-swap sweep results
(runs/clevr_relation_swap_7obj_{dm,thinker}_full_v2.json) and produces:
  - paper_plots/clevr_relation_swap_heatmaps.pdf: two heatmaps side by side
    (PaTh, DM), all_success_rate for cycle_length=2, n_groups in {1,2,3},
    t_start in {10,30,50,70,90} (95 and scratch excluded to match the
    5-timestep table the figure mirrors).
  - paper_plots/clevr_relation_swap_tables.tex: booktabs LaTeX source for
    three cycle_length=2 tables (all_success_rate, success_rate, stuck_rate),
    each with PaTh and DM side by side.

Usage:
    python paper_plots/plot_clevr_relation_swap_heatmaps.py
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

DM_PATH = "runs/clevr_relation_swap_7obj_dm_bigger_full.json"
THINKER_PATH = "runs/clevr_relation_swap_7obj_thinker_full_v2.json"

CYCLE_LENGTH = 2
N_GROUPS = [1, 2, 3]
T_STARTS = [10, 30, 50, 70, 90]  # 95 and scratch excluded

# t_start is the noise level the corrupted run STARTS from (0=clean,
# 100=pure noise), i.e. how much of the sampling trajectory remains once the
# mistake is introduced -- the opposite of "% of sampling completed so far".
# The heatmap displays the latter (pct = 100 - t_start), ascending top to
# bottom, since that's the more intuitive read for "when does the mistake
# happen": e.g. t_start=90 means only 10% of sampling has happened yet.
TSTART_TO_PCT = {t: 100 - t for t in T_STARTS}
HEATMAP_ROW_ORDER = sorted(T_STARTS, key=lambda t: TSTART_TO_PCT[t])


def _load(path):
    with open(path) as f:
        return json.load(f)["results"]


def _matrix(results, metric, cycle_length=CYCLE_LENGTH, n_groups=N_GROUPS, t_starts=T_STARTS):
    mat = np.full((len(t_starts), len(n_groups)), np.nan)
    for i, t in enumerate(t_starts):
        for j, ng in enumerate(n_groups):
            r = results.get(f"t_start={t}/cycle_length={cycle_length}/n_groups={ng}")
            if r is not None and r.get(metric) is not None:
                mat[i, j] = r[metric]
    return mat


# ── Heatmaps ─────────────────────────────────────────────────────────────────


def make_heatmaps(dm_results, th_results):
    dm_mat = _matrix(dm_results, "all_success_rate", t_starts=HEATMAP_ROW_ORDER)
    th_mat = _matrix(th_results, "all_success_rate", t_starts=HEATMAP_ROW_ORDER)
    vmax = max(np.nanmax(dm_mat), np.nanmax(th_mat))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75), sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, mat, title, color in [
        (axes[0], th_mat, "PaTh", COLOR_THINKER),
        (axes[1], dm_mat, "DM", COLOR_DM),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(N_GROUPS)))
        ax.set_xticklabels([str(n) for n in N_GROUPS])
        ax.set_xlabel("Number of swaps")
        ax.set_title(title, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                text_color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    axes[0].set_yticks(range(len(HEATMAP_ROW_ORDER)))
    axes[0].set_yticklabels([f"{TSTART_TO_PCT[t]}%" for t in HEATMAP_ROW_ORDER])
    axes[0].set_ylabel("Mistake simulated at % of sampling")

    fig.savefig("paper_plots/clevr_relation_swap_heatmaps.pdf", bbox_inches="tight")
    print("Wrote paper_plots/clevr_relation_swap_heatmaps.pdf")


# ── LaTeX tables ─────────────────────────────────────────────────────────────


def _fmt(v):
    return f"{v:.3f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "--"


def _table(dm_results, th_results, metric, caption, label):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{c" + "ccc" * 2 + "}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{3}{c}{PaTh} & \multicolumn{3}{c}{DM} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(
        r"$t_{\mathrm{start}}$ & $n_g{=}1$ & $n_g{=}2$ & $n_g{=}3$ & $n_g{=}1$ & $n_g{=}2$ & $n_g{=}3$ \\"
    )
    lines.append(r"\midrule")
    for t in T_STARTS:
        row = [str(t)]
        for results in (th_results, dm_results):
            for ng in N_GROUPS:
                r = results.get(f"t_start={t}/cycle_length={CYCLE_LENGTH}/n_groups={ng}")
                row.append(_fmt(r[metric] if r is not None else None))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_tables(dm_results, th_results):
    tables = [
        _table(
            dm_results, th_results, "all_success_rate",
            f"All-success rate (every corrupted object correctly placed) for cycle length $=${CYCLE_LENGTH}, "
            "as a function of noise level and number of disjoint swap groups $n_g$.",
            "clevr_swap_all_success",
        ),
        _table(
            dm_results, th_results, "success_rate",
            f"Per-object success rate for cycle length $=${CYCLE_LENGTH}.",
            "clevr_swap_success",
        ),
        _table(
            dm_results, th_results, "stuck_rate",
            f"Per-object stuck (failure) rate for cycle length $=${CYCLE_LENGTH}.",
            "clevr_swap_stuck",
        ),
    ]
    out = "\n\n".join(tables) + "\n"
    with open("paper_plots/clevr_relation_swap_tables.tex", "w") as f:
        f.write(out)
    print("Wrote paper_plots/clevr_relation_swap_tables.tex")
    print("\n" + out)


def main():
    dm_results = _load(DM_PATH)
    th_results = _load(THINKER_PATH)
    make_heatmaps(dm_results, th_results)
    make_tables(dm_results, th_results)


if __name__ == "__main__":
    main()
