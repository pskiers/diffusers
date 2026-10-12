"""Relation-swap probe: PaTh vs. DM heatmaps, one per object count (3/5/7/10).

Same style/metric as plot_clevr_relation_swap_heatmaps.py (all_success_rate,
cycle_length=2, t_start in {10,30,50,70,90}), but n_groups is only swept as
far as it fits in that many objects (cycle_length * n_groups <= object count),
so each panel has a different number of columns:
  3 objects  -> n_groups in {1}
  5 objects  -> n_groups in {1, 2}
  7 objects  -> n_groups in {1, 2, 3}
  10 objects -> n_groups in {1, 2, 3, 4, 5}

Usage:
    python paper_plots/plot_clevr_relation_swap_heatmaps_by_objcount.py
Produces paper_plots/clevr_relation_swap_heatmaps_{n}obj.pdf for n in 3,5,7,10.
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

CYCLE_LENGTH = 2
T_STARTS = [10, 30, 50, 70, 90]  # 95 and scratch excluded
TSTART_TO_PCT = {t: 100 - t for t in T_STARTS}
HEATMAP_ROW_ORDER = sorted(T_STARTS, key=lambda t: TSTART_TO_PCT[t])

# DM = the bigger clevr_unet_concat_painter checkpoint (runs/clevr_unet_concat_bigger.pt:
# block_out_channels=[256,512,512], layers_per_block=3, cross_attention_dim=1024),
# which superseded the original (smaller) DM checkpoint everywhere. PaTh/thinker
# is unrelated to this checkpoint and reuses its earlier sweep (n=7 has a
# corrected _v2 dataset -- retry-fixed t=10 thinker run, plus the
# all_success_rate/mean_d_true diagnostics added after the first 7-object sweep).
PATHS = {
    3: ("runs/clevr_relation_swap_3obj_dm_bigger_full.json", "runs/clevr_relation_swap_3obj_thinker_full.json"),
    5: ("runs/clevr_relation_swap_5obj_dm_bigger_full.json", "runs/clevr_relation_swap_5obj_thinker_full.json"),
    7: ("runs/clevr_relation_swap_7obj_dm_bigger_full.json", "runs/clevr_relation_swap_7obj_thinker_full_v2.json"),
    10: ("runs/clevr_relation_swap_10obj_dm_bigger_full.json", "runs/clevr_relation_swap_10obj_thinker_full.json"),
}


def _load(path):
    with open(path) as f:
        return json.load(f)["results"]


def _matrix(results, metric, n_groups, t_starts=HEATMAP_ROW_ORDER):
    mat = np.full((len(t_starts), len(n_groups)), np.nan)
    for i, t in enumerate(t_starts):
        for j, ng in enumerate(n_groups):
            r = results.get(f"t_start={t}/cycle_length={CYCLE_LENGTH}/n_groups={ng}")
            if r is not None and r.get(metric) is not None:
                mat[i, j] = r[metric]
    return mat


def make_heatmap(n_objects):
    dm_path, th_path = PATHS[n_objects]
    dm_results = _load(dm_path)
    th_results = _load(th_path)

    n_groups = [ng for ng in range(1, 6) if CYCLE_LENGTH * ng <= n_objects]

    dm_mat = _matrix(dm_results, "all_success_rate", n_groups)
    th_mat = _matrix(th_results, "all_success_rate", n_groups)
    vmax = max(np.nanmax(dm_mat), np.nanmax(th_mat))

    width = 3.2 + 0.7 * len(n_groups)
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.75), sharey=True)
    cmap = plt.get_cmap("viridis")

    for ax, mat, title, color in [
        (axes[0], th_mat, "PaTh", COLOR_THINKER),
        (axes[1], dm_mat, "DM", COLOR_DM),
    ]:
        ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(n_groups)))
        ax.set_xticklabels([str(n) for n in n_groups])
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
                ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=9, color=text_color)

    axes[0].set_yticks(range(len(HEATMAP_ROW_ORDER)))
    axes[0].set_yticklabels([f"{TSTART_TO_PCT[t]}%" for t in HEATMAP_ROW_ORDER])
    axes[0].set_ylabel("Mistake simulated at % of sampling")

    out_path = f"paper_plots/clevr_relation_swap_heatmaps_{n_objects}obj.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    for n_objects in [3, 5, 7, 10]:
        make_heatmap(n_objects)


if __name__ == "__main__":
    main()
