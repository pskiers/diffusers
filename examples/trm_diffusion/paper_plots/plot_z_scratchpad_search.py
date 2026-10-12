"""Camera-ready figures for the TRM z_L "scratchpad" hypothesis (see
experiments/z_scratchpad_probe.py): does the fast recurrent state z_L
search/rethink rather than just monotonically refine toward the committed
answer y?

The x-axis is the FULL combined progression, not just the sparse denoising-
step index: every denoising step contributes n_sup * H_cycles (16 * 3 = 48
by default) y-update sub-positions, and every one of those gives a metric
reading — concatenated back-to-back across every instrumented denoising
step, with no gaps, PROVIDED experiments/z_scratchpad_probe.py was run with
one probe.denoise_fractions entry per actual denoising step (e.g. all 20 of
them for num_inference_steps=20), not a sparse subset.

Produces 3 separate PDFs from one z_scratchpad_probe.py output JSON:

  <out_prefix>_violations.pdf         — y's own violations vs the mean
      violations of the L_cycles z's decoded between that y and the next
      (from y_fine_curve / z_curve).
  <out_prefix>_repair_regression.pdf  — repair_rate vs regression_rate
      between consecutive y's (from y_fine_transitions_by_position).
  <out_prefix>_dip.pdf                — viol_dip_rate vs
      viol_dip_magnitude_given_dip — rate ("how often the worst z in a
      group is more violated than the y it started from") vs conditional
      magnitude ("how much worse, when it happens") — see
      z_scratchpad_probe.py's _dip_rates_from_counts docstring for why the
      conditional (not unconditional) magnitude is the honest pairing with
      rate here (from dip_by_group).

Usage:
    python paper_plots/plot_z_scratchpad_search.py \\
      --input runs/z_scratchpad_full_20steps.json \\
      --out_prefix paper_plots/z_scratchpad \\
      --num_inference_steps 20
"""

from __future__ import annotations

import argparse
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_Y = "#2c2c2a"
COLOR_Z = "#8b3fd9"
FIGSIZE = (3.35, 2.55)

_FRAC_RE = re.compile(r"frac=([0-9.]+)")


def _sorted_buckets(results: dict, num_inference_steps: int) -> list:
    """[(denoising_step_idx, bucket_dict), ...] sorted by denoise fraction —
    robust to dict/JSON ordering, and recovers the real step index (for
    tick labels) from the "frac=<x>" bucket label, matching how
    z_scratchpad_probe.py derives it (round(frac * (num_inference_steps - 1)))."""
    items = []
    for label, r in results.items():
        frac = float(_FRAC_RE.match(label).group(1))
        step_idx = round(frac * (num_inference_steps - 1))
        items.append((step_idx, r))
    items.sort(key=lambda p: p[0])
    return items


def _boundary_ticks(buckets: list, points_per_step: int, every: int = 5) -> tuple:
    """x tick position (start of each bucket's block) + real step index label,
    only every `every`-th bucket (plus the last) to avoid a cluttered axis."""
    ticks, labels = [], []
    for i, (step_idx, _r) in enumerate(buckets):
        if i % every == 0:
            ticks.append(i * points_per_step)
            labels.append(str(100 // len(buckets) * step_idx))
    ticks.append(len(buckets) * points_per_step)
    labels.append("100")
    return ticks, labels


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)


def plot_violations(buckets: list, out_path: str) -> None:
    y_all, z_all = [], []
    for _step_idx, r in buckets:
        n_groups = len(r["y_fine_curve"])
        L = len(r["z_curve"]) // n_groups
        y_all.extend(p["violations"] for p in r["y_fine_curve"])
        z_all.extend(
            float(np.mean([r["z_curve"][g * L + l]["violations"] for l in range(L)]))
            for g in range(n_groups)
        )
    x = np.arange(len(y_all))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, z_all, linewidth=1.1, color=COLOR_Z, label="z (mean over group)", zorder=3)
    ax.plot(x, y_all, linewidth=1.5, color=COLOR_Y, label="y (committed answer)", zorder=4)
    ticks, labels = _boundary_ticks(buckets, len(buckets[0][1]["y_fine_curve"]))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Constraint violations")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_repair_regression(buckets: list, out_path: str, gaussian_sigma: float = 20.0) -> None:
    """Plain global Gaussian — no outlier rejection here. Checked directly
    against the data: the largest value in this whole series is the
    genuine, legitimate peak right at step 0 (repair rate starts high and
    decays fast as reasoning first stabilizes), not a pathological outlier
    the way the dip plot's steering-collapse spike is — an outlier filter
    here would incorrectly flag that real fast decay and flatten it (see
    plot_dip's _hampel_smooth docstring for the analogous failure mode
    this avoids)."""
    repair_all, regress_all = [], []
    for _step_idx, r in buckets:
        repair_all.extend(p["repair_rate"] for p in r["y_fine_transitions_by_position"])
        regress_all.extend(p["regression_rate"] for p in r["y_fine_transitions_by_position"])
    x = np.arange(len(repair_all))
    n_per_step = len(buckets[0][1]["y_fine_transitions_by_position"])
    if gaussian_sigma > 0:
        repair_all = gaussian_filter1d(repair_all, gaussian_sigma, mode="nearest")
        regress_all = gaussian_filter1d(regress_all, gaussian_sigma, mode="nearest")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, repair_all, linewidth=3, color=COLOR_Y, label="repair rate", zorder=3)
    ax.plot(x, regress_all, linewidth=3, color=COLOR_Z, label="regression rate", zorder=3)
    ticks, labels = _boundary_ticks(buckets, n_per_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Rate")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _interpolate_nan(values) -> np.ndarray:
    """viol_dip_magnitude_given_dip is nan wherever a group had zero dip
    events — "0 magnitude" would be a misleading value (see
    z_scratchpad_probe.py's _dip_rates_from_counts: it reads as "the dip
    that didn't happen would have been zero-sized", and visually whipsaws
    the plotted curve between 0 and whatever a nearby group's real, nonzero
    conditional magnitude is, common late in the trajectory once dips
    become rare). Treating it as genuinely missing and linearly
    interpolating between the surviving defined points instead gives a
    smooth trend through the gaps — nan runs at either end of the whole
    sequence are filled with the nearest defined value (np.interp's
    default constant-extrapolation behavior)."""
    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    return np.interp(idx, idx[valid], arr[valid])


def _hampel_smooth(values, window: int, tail_start_frac: float = 0.75, n_sigmas: float = 3.0) -> np.ndarray:
    """Hampel filter, restricted to the last `tail_start_frac` fraction of
    the sequence: at each position IN THAT TAIL, compares the value to a
    robust local baseline (the median of a `window`-wide neighborhood,
    which may reach back before the tail starts) using a robust local
    spread estimate (that neighborhood's median absolute deviation, MAD,
    scaled by 1.4826 to be a consistent estimator of std under normality)
    — flags it an outlier only if it's more than n_sigmas of THAT local
    spread away, then replaces just the flagged points with a linear
    interpolation through the surviving ones (see _interpolate_nan).

    The tail restriction matters: running this over the WHOLE sequence
    also flags the genuine, fast peak right at step 0 (verified directly
    — with no restriction it replaces the first ~10 points with a flat
    linear ramp, destroying real signal), because a window wide enough to
    correctly catch the wide steering-collapse anomaly is, at the start,
    also wide enough to make a short LEGITIMATE transient look anomalous
    relative to the plateau it settles into. Restricting eligibility to a
    tail we already know (from independent evidence — see
    experiments/z_scratchpad_probe_readout.py's steering-collapse finding)
    is where the real anomaly lives sidesteps that false positive rather
    than trying to tune one window to satisfy both ends at once.

    window=1 (or less) is a no-op."""
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr
    n = len(arr)
    tail_start = int(n * tail_start_frac)
    half = window // 2
    cleaned = arr.copy()
    for i in range(tail_start, n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = arr[lo:hi]
        m = np.median(w)
        mad = np.median(np.abs(w - m))
        threshold = max(n_sigmas * 1.4826 * mad, 1e-9)
        if abs(arr[i] - m) > threshold:
            cleaned[i] = np.nan
    return _interpolate_nan(cleaned)


def plot_dip(
    buckets: list, out_path: str, smooth_window: int = 400, tail_start_frac: float = 0.75,
    gaussian_sigma: float = 20.0,
) -> None:
    rate_all, mag_all = [], []
    for _step_idx, r in buckets:
        rate_all.extend(p["viol_dip_rate"] for p in r["dip_by_group"])
        mag_all.extend(p["viol_dip_magnitude_given_dip"] for p in r["dip_by_group"])
    x = np.arange(len(rate_all))
    mag_all = _interpolate_nan(mag_all)
    # Hampel first (reject the real-but-narrow collapse spike without
    # bending toward it), THEN Gaussian-smooth what's left (ordinary local
    # jitter, not a modeling artifact) into a clean line — applying the
    # Gaussian before Hampel would instead blur the spike's edges into a
    # wide hump the outlier check could no longer recognize as anomalous.
    rate_all = _hampel_smooth(rate_all, smooth_window, tail_start_frac)
    mag_all = _hampel_smooth(mag_all, smooth_window, tail_start_frac)
    if gaussian_sigma > 0:
        rate_all = gaussian_filter1d(rate_all, gaussian_sigma, mode="nearest")
        mag_all = gaussian_filter1d(mag_all, gaussian_sigma, mode="nearest")

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    l1, = ax1.plot(x, rate_all, linewidth=3, color=COLOR_Y, label="rate", zorder=3)
    l2, = ax2.plot(x, mag_all, linewidth=3, color=COLOR_Z, label="magnitude", zorder=3)
    n_per_step = len(buckets[0][1]["dip_by_group"])
    ticks, labels = _boundary_ticks(buckets, n_per_step)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Sampling completed (%)")
    ax1.set_ylabel("Rethink rate", color=COLOR_Y)
    ax2.set_ylabel("Rethink magnitude", color=COLOR_Z)
    ax1.tick_params(axis="y", labelcolor=COLOR_Y)
    ax2.tick_params(axis="y", labelcolor=COLOR_Z)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(handles=[l1, l2], frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="z_scratchpad_probe.py output JSON")
    ap.add_argument("--out_prefix", required=True, help="e.g. paper_plots/z_scratchpad")
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument(
        "--repair_gaussian_sigma", type=float, default=20.0,
        help="repair/regression plot: global Gaussian smoothing sigma (0 = no smoothing)",
    )
    ap.add_argument(
        "--dip_smooth_window", type=int, default=400,
        help="dip plot: Hampel-filter window (outlier rejection + interpolation, not averaging), "
             "only applied within the tail (see --dip_tail_start_frac) — needs to be wide relative "
             "to the anomaly it's meant to reject (1 = no smoothing)",
    )
    ap.add_argument(
        "--dip_tail_start_frac", type=float, default=0.75,
        help="dip plot: Hampel outlier rejection only runs on the last this-fraction of the "
             "sequence, so it can't mistake the genuine early peak for an anomaly",
    )
    ap.add_argument(
        "--dip_gaussian_sigma", type=float, default=20.0,
        help="dip plot: Gaussian smoothing sigma applied after Hampel cleanup, for a clean line (0 = no smoothing)",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)
    buckets = _sorted_buckets(d["results"], args.num_inference_steps)

    plot_violations(buckets, f"{args.out_prefix}_violations.pdf")
    plot_repair_regression(buckets, f"{args.out_prefix}_repair_regression.pdf", args.repair_gaussian_sigma)
    plot_dip(
        buckets, f"{args.out_prefix}_dip.pdf", args.dip_smooth_window, args.dip_tail_start_frac,
        args.dip_gaussian_sigma,
    )
    print(f"Saved {args.out_prefix}_violations.pdf, _repair_regression.pdf, _dip.pdf")


if __name__ == "__main__":
    main()
