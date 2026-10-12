"""Relation-swap probe: PaTh vs. DM, three derived views on data already
computed by the {3,5,7,10}-object sweeps (no new generation/scoring, just
different slices/aggregates of the saved results).

  1. Heatmap with object count (3/5/7/10) on the x-axis instead of number of
     swaps -- fixes n_groups=1 (exactly one swap), cycle_length=2, and shows
     all_success_rate across the same 5 "% of sampling" rows as the other
     heatmaps.
  2. Line plot of per-object success_rate ("recovered"), averaged over the
     feasible n_groups values (1,2,3) at cycle_length=2, for the 7-object
     sweep only.
  3. Same, but averaged over n_groups AND then over object count -- for each
     object count, first average across its own feasible n_groups values
     (as in plot 2), THEN average those per-object-count numbers together
     with equal weight per object count (not a flat average over every
     (object_count, n_groups) cell, which would let n=10's 5 n_groups values
     outweigh n=3's 1).

Usage:
    python paper_plots/plot_clevr_relation_swap_objcount_and_avg.py
Produces:
    paper_plots/clevr_relation_swap_heatmap_by_nobjects.pdf
    paper_plots/clevr_relation_swap_avg_success_7obj.pdf
    paper_plots/clevr_relation_swap_avg_success_all_nobjects.pdf
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

# User-specified colors for this figure set (distinct from the
# COLOR_DM/COLOR_THINKER used in the other clevr_relation_swap_* figures).
COLOR_THINKER = "#8b3fd9"
COLOR_DM = "#2c2c2a"

CYCLE_LENGTH = 2
T_STARTS = [10, 30, 50, 70, 90]
TSTART_TO_PCT = {t: 100 - t for t in T_STARTS}
HEATMAP_ROW_ORDER = sorted(T_STARTS, key=lambda t: TSTART_TO_PCT[t])
OBJECT_COUNTS = [3, 4, 5, 6, 7, 8, 9, 10]

# DM = the bigger clevr_unet_concat_painter checkpoint (runs/clevr_unet_concat_bigger.pt),
# same as plot_clevr_relation_swap_heatmaps_by_objcount.py.
PATHS = {
    3: ("runs/clevr_relation_swap_3obj_dm_bigger_full.json", "runs/clevr_relation_swap_3obj_thinker_full.json"),
    4: ("runs/clevr_relation_swap_4obj_dm_bigger_full.json", "runs/clevr_relation_swap_4obj_thinker_full.json"),
    5: ("runs/clevr_relation_swap_5obj_dm_bigger_full.json", "runs/clevr_relation_swap_5obj_thinker_full.json"),
    6: ("runs/clevr_relation_swap_6obj_dm_bigger_full.json", "runs/clevr_relation_swap_6obj_thinker_full.json"),
    7: ("runs/clevr_relation_swap_7obj_dm_bigger_full.json", "runs/clevr_relation_swap_7obj_thinker_full_v2.json"),
    8: ("runs/clevr_relation_swap_8obj_dm_bigger_full.json", "runs/clevr_relation_swap_8obj_thinker_full.json"),
    9: ("runs/clevr_relation_swap_9obj_dm_bigger_full.json", "runs/clevr_relation_swap_9obj_thinker_full.json"),
    10: ("runs/clevr_relation_swap_10obj_dm_bigger_full.json", "runs/clevr_relation_swap_10obj_thinker_full.json"),
}

# {3,5,7,10} were computed with the full cycle_length/n_groups grid; {4,6,8,9}
# were added later (via a SLURM array job) with ONLY cycle_length=2,
# n_groups=1 -- so "feasible n_groups" here means "actually computed", not
# just mathematically fits (cycle_length * n_groups <= n_objects), which
# would wrongly claim n_groups=2 exists for e.g. n=4.
AVAILABLE_N_GROUPS = {3: [1], 4: [1], 5: [1, 2], 6: [1], 7: [1, 2, 3], 8: [1], 9: [1], 10: [1, 2, 3, 4, 5]}


def _load(path):
    with open(path) as f:
        return json.load(f)["results"]


def _feasible_n_groups(n_objects):
    return AVAILABLE_N_GROUPS[n_objects]


def _get(results, t_start, n_groups, metric):
    r = results.get(f"t_start={t_start}/cycle_length={CYCLE_LENGTH}/n_groups={n_groups}")
    return r[metric] if r is not None and r.get(metric) is not None else None


# ── 1. Heatmap: object count x % sampling, n_groups=1 (exactly one swap) ────


def make_objcount_heatmap(object_counts=OBJECT_COUNTS, suffix=""):
    all_results = {n: (_load(PATHS[n][0]), _load(PATHS[n][1])) for n in object_counts}

    dm_mat = np.full((len(HEATMAP_ROW_ORDER), len(object_counts)), np.nan)
    th_mat = np.full((len(HEATMAP_ROW_ORDER), len(object_counts)), np.nan)
    for i, t in enumerate(HEATMAP_ROW_ORDER):
        for j, n in enumerate(object_counts):
            dm_results, th_results = all_results[n]
            dm_mat[i, j] = _get(dm_results, t, 1, "all_success_rate")
            th_mat[i, j] = _get(th_results, t, 1, "all_success_rate")

    vmax = max(np.nanmax(dm_mat), np.nanmax(th_mat))

    width = 3.6 + 0.5 * len(object_counts)
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.75), sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, mat, title, color in [
        (axes[0], th_mat, "PaTh", COLOR_THINKER),
        (axes[1], dm_mat, "DM", COLOR_DM),
    ]:
        ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(object_counts)))
        ax.set_xticklabels([str(n) for n in object_counts])
        ax.set_xlabel("Number of objects")
        ax.set_title(title, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                text_color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=9, color=text_color)

    axes[0].set_yticks(range(len(HEATMAP_ROW_ORDER)))
    axes[0].set_yticklabels([f"{TSTART_TO_PCT[t]}%" for t in HEATMAP_ROW_ORDER])
    axes[0].set_ylabel("Mistake simulated at % of sampling")

    out_path = f"paper_plots/clevr_relation_swap_heatmap_by_nobjects{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── 2 & 3. Line plots of success_rate averaged over swaps (/ + objects) ─────


def _avg_success_per_tstart(results, n_groups_list):
    """Mean success_rate across n_groups_list, one value per t_start."""
    out = []
    for t in T_STARTS:
        vals = [_get(results, t, ng, "success_rate") for ng in n_groups_list]
        vals = [v for v in vals if v is not None]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


GAP_HIGHLIGHT = "#E69F00"  # warm amber -- reads as "diverging/at-risk", distinct from both curve colors


def _line_plot(pct_x, th_y, dm_y, ylabel, out_path, figsize=(3.35, 2.55), highlight_span=None):
    """highlight_span: optional (x0, x1) to tint the background over the
    region where DM falls off fastest relative to PaTh -- e.g. (10, 30).
    The gap between the two curves is also shaded everywhere (amber), so its
    width -- widening fast in the highlighted band, narrowing after -- reads
    as "DM drops off quickly there, PaTh stays comparatively flat" without
    needing a separate annotation."""
    fig, ax = plt.subplots(figsize=figsize)

    if highlight_span is not None:
        ax.axvspan(*highlight_span, color=GAP_HIGHLIGHT, alpha=0.12, zorder=0, linewidth=0)

    ax.fill_between(pct_x, dm_y, th_y, color=GAP_HIGHLIGHT, alpha=0.28, zorder=1, linewidth=0)

    ax.plot(pct_x, th_y, marker="o", markersize=5, linewidth=2.4, color=COLOR_THINKER, label="PaTh", zorder=3)
    ax.plot(pct_x, dm_y, marker="s", markersize=5, linewidth=2.4, color=COLOR_DM, label="DM", zorder=3)
    ax.set_xlabel("Mistake simulated at % of sampling")
    ax.set_ylabel(ylabel)
    ax.set_xticks(pct_x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_avg_7obj():
    dm_results, th_results = _load(PATHS[7][0]), _load(PATHS[7][1])
    n_groups_list = _feasible_n_groups(7)  # [1, 2, 3]

    pct_x = [TSTART_TO_PCT[t] for t in T_STARTS]
    dm_y = _avg_success_per_tstart(dm_results, n_groups_list)
    th_y = _avg_success_per_tstart(th_results, n_groups_list)

    _line_plot(
        pct_x, th_y, dm_y,
        "Mean recovered rate (1-3 swaps)",
        "paper_plots/clevr_relation_swap_avg_success_7obj.pdf",
        highlight_span=(10, 30),
    )


def make_avg_all_nobjects(object_counts=OBJECT_COUNTS, suffix=""):
    # Per-object-count average first (same as make_avg_7obj, generalized),
    # THEN average those numbers with equal weight per object count -- NOT a
    # flat mean over every (object_count, n_groups) cell, which would let
    # n=10 (5 n_groups values) dominate n=3 (only 1).
    per_n_dm = []
    per_n_th = []
    for n in object_counts:
        dm_results, th_results = _load(PATHS[n][0]), _load(PATHS[n][1])
        n_groups_list = _feasible_n_groups(n)
        per_n_dm.append(_avg_success_per_tstart(dm_results, n_groups_list))
        per_n_th.append(_avg_success_per_tstart(th_results, n_groups_list))

    dm_y = np.nanmean(np.stack(per_n_dm), axis=0)
    th_y = np.nanmean(np.stack(per_n_th), axis=0)
    pct_x = [TSTART_TO_PCT[t] for t in T_STARTS]

    _line_plot(
        pct_x, th_y, dm_y,
        "Mean recovered rate\n(all swaps & object counts)",
        f"paper_plots/clevr_relation_swap_avg_success_all_nobjects{suffix}.pdf",
        figsize=(3.35, 2.75),
        highlight_span=(10, 30),
    )


def main():
    make_objcount_heatmap()
    make_avg_7obj()
    make_avg_all_nobjects()

    # Exploratory: same two figures with the 3-object column dropped.
    no3 = [n for n in OBJECT_COUNTS if n not in (3,4)]
    make_objcount_heatmap(object_counts=no3, suffix="_5t010")
    make_avg_all_nobjects(object_counts=no3, suffix="_5to10")


if __name__ == "__main__":
    main()
