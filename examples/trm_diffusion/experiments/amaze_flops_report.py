"""experiments/amaze_flops_report.py — figures for the AMAZE FLOPs / "where reasoning is needed" analysis.

Reads the JSONs written by experiments/amaze_flops_sweep.py under <dir>/{report_split,tune_split}/ (any missing file
is skipped), writes PNGs to <dir>/report/ and logs ALL of them to ONE new wandb run (project PaTh-rebuttal, group
flops) unless --no-wandb.  No stars / special symbols: plain lines, small circle markers.

  held-out boards (report_split/)
    <task>_03_flops_vs_exact.png     exact vs FLOPs: every DiT (<task>_dm_<label>.json) + PaTh flat (+ PaTh wise if run)
    <task>_04_flops_by_size.png      PaTh flat, one line per board size
    <task>_04_flops_by_shape.png     PaTh flat, one line per shape
  tuning boards (tune_split/)
    <task>_01_z_change.png           mean relative z_H change per reasoning iteration (0-1, fraction of the old z_H)
    <task>_02_cosine.png             same-board cosine of the final z_H between denoising steps (raw / board-specific)
    <task>_05_cut_at_k.png           exact vs % of denoising steps that reason (all boards)
    <task>_05_cut_at_k_by_size.png   the same, one panel per board size
    <task>_06_carry.png              exact vs n_sup, one line per state-reset policy
    <task>_07_allocation.png         budget bumps (columns = centre, rows = width), colour = change in exact vs uniform,
                                     + the schedule derived from the z_H-change profile

    python experiments/amaze_flops_report.py --dir runs/flops_analysis/maze --tasks maze
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.markersize": 4, "lines.linewidth": 1.8})

PATH_COLOR = "#0072B2"
DIT_COLORS = ["#D55E00", "#CC79A7", "#E69F00", "#7f7f7f"]
BLUES = {"every20": "#9ecae1", "every5": "#6baed6", "every2": "#3182bd", "every1": "#08519c"}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _pct(v):
    return 100.0 * v


# ── held-out boards ──────────────────────────────────────────────────────────


def fig_flops_vs_exact(d: Path, task: str, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    drawn = False
    for i, p in enumerate(sorted((d / "report_split").glob(f"{task}_dm_*.json"))):
        r = _load(p)
        pts = sorted(r["points"], key=lambda q: q["flops"])
        ax.plot([q["flops"] for q in pts], [_pct(q["exact"]) for q in pts], "-o", color=DIT_COLORS[i % 4],
                label=f"DiT {p.stem.split('_dm_', 1)[1]}")
        drawn = True
    for mode, label, color in (("trm_flat", "PaTh", PATH_COLOR), ("trm_wise", "PaTh wise", "#009E73")):
        r = _load(d / "report_split" / f"{task}_{mode}.json")
        if r:
            pts = sorted(r["points"], key=lambda q: q["flops"])
            ax.plot([q["flops"] for q in pts], [_pct(q["exact"]) for q in pts], "-o", color=color, label=label)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per sample")
    ax.set_ylabel("Exact-solve rate (%)")
    ax.set_title(f"{task}: accuracy vs compute (held-out boards)")
    ax.legend()
    return _save(fig, out)


def _fig_by(d: Path, task: str, key: str, out: Path, title: str):
    r = _load(d / "report_split" / f"{task}_trm_flat.json")
    if not r or key not in r["points"][0]:
        return None
    pts = sorted(r["points"], key=lambda q: q["flops"])
    names = sorted({n for q in pts for n in q[key]}, key=lambda n: (len(n), n))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(names):
        ax.plot([q["flops"] for q in pts], [_pct(q[key][n]["exact"]) for q in pts], "-o",
                color=cmap(i / max(1, len(names) - 1)), label=n)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per sample")
    ax.set_ylabel("Exact-solve rate (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, out)


# ── tuning boards: the z_H profile ───────────────────────────────────────────


def _ts_labels(r, n):
    ts = r.get("timesteps") or list(range(n))
    return [str(t) for t in ts]


def fig_z_change(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_profile.json")
    if not r or "profile" not in r:
        return None
    rel = r["profile"]["rel_change_mean"]
    n_it = max(len(row) for row in rel)
    M = np.array([row + [np.nan] * (n_it - len(row)) for row in rel])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.grid(False)
    ax.set_yticks(range(len(rel)), _ts_labels(r, len(rel)))
    ax.set_xticks(range(n_it), [str(i + 1) for i in range(n_it)])
    ax.set_xlabel("reasoning iteration")
    ax.set_ylabel("denoising timestep t (top = pure noise)")
    ax.set_title("How much z_H changes at each reasoning iteration")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("change vs the previous iteration\n(fraction of the previous z_H norm; >1 shown as 1)")
    return _save(fig, out)


def fig_cosine(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_profile.json")
    if not r or "profile" not in r or "cos_raw" not in r["profile"]:
        return None
    pr = r["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, key, title in ((axes[0], "cos_raw", "raw"),
                           (axes[1], "cos_centered", "board-specific part (board mean removed)")):
        C = np.array(pr[key])
        lab = _ts_labels(r, len(C))
        im = ax.imshow(C, cmap="Blues", vmin=float(np.floor(np.nanmin(C) * 20) / 20), vmax=1.0)
        ax.grid(False)
        ax.set_xticks(range(len(C)), lab, rotation=90, fontsize=7)
        ax.set_yticks(range(len(C)), lab, fontsize=7)
        ax.set_xlabel("denoising timestep t")
        ax.set_ylabel("denoising timestep t")
        ax.set_title(f"Final z_H, same-board cosine: {title}", fontsize=10)
        fig.colorbar(im, ax=ax).set_label("cosine similarity (1 = identical)")
    return _save(fig, out)


# ── tuning boards: where to cut ──────────────────────────────────────────────


def _cut_panel(ax, pts, getter, ref_label=True):
    fr = [100 * q["reason_frac"] for q in pts]
    ys = [getter(q) for q in pts]
    ax.plot(fr, ys, "-o", color=PATH_COLOR)
    full = next((y for q, y in zip(pts, ys) if q["reason_frac"] >= 1.0), None)
    if full is not None:
        ax.axhline(full, color="0.55", ls="--", lw=1)
        if ref_label:
            ax.text(2, full + 1.0, "reason every step", color="0.45", fontsize=8)
    ax.set_xlim(0, 102)
    ax.set_xlabel("Reasoning in the first (% of denoising steps)")
    ax.set_ylabel("Exact-solve rate (%)")


def fig_cut(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_cut_at_k.json")
    if not r:
        return None
    pts = sorted(r["points"], key=lambda q: q["reason_frac"])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    _cut_panel(ax, pts, lambda q: _pct(q["exact"]))
    ax.set_title("Where to stop reasoning (all boards)")
    return _save(fig, out)


def fig_cut_by_size(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_cut_at_k.json")
    if not r:
        return None
    pts = sorted(r["points"], key=lambda q: q["reason_frac"])
    sizes = sorted({s for q in pts for s in q["per_scale"]}, key=int)
    ncol = min(4, len(sizes))
    nrow = int(np.ceil(len(sizes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False, sharey=True)
    for ax, s in zip(axes.flat, sizes):
        _cut_panel(ax, pts, lambda q, s=s: _pct(q["per_scale"][s]["exact"]), ref_label=False)
        ax.set_title(f"board size {s}")
    for ax in list(axes.flat)[len(sizes):]:
        ax.axis("off")
    return _save(fig, out)


# ── tuning boards: carry ─────────────────────────────────────────────────────


def fig_carry(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_carry.json")
    if not r:
        return None
    variants = []
    for q in r["points"]:
        if q["variant"] not in variants:
            variants.append(q["variant"])
    b = r.get("measured_boundary_step")
    ad = r.get("adaptive_reset_steps", [])
    names = {"every20": "reset every 20", "every5": "reset every 5", "every2": "reset every 2",
             "every1": "reset every 1 (fresh)", "regime": f"reset once, at step {b} (measured boundary)",
             "adaptive": f"reset map from cosine ({len(ad) - 1} resets, steps {', '.join(str(x) for x in ad if x)})",
             "fresh": "fresh each step", "carry_all": "carry everywhere", "carry_regime": "carry within a regime"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for v in variants:
        pts = sorted([q for q in r["points"] if q["variant"] == v], key=lambda q: q["n_sup"])
        style = dict(color=BLUES.get(v, "#555555"))
        if v == "regime":
            style = dict(color="#D55E00", ls="--")
        elif v == "adaptive":
            style = dict(color="#009E73", ls="--")
        ax.plot([q["n_sup"] for q in pts], [_pct(q["exact"]) for q in pts], "-o" if "ls" not in style else "o",
                label=names.get(v, v), **style)
    ax.set_xscale("log", base=2)
    ns = sorted({q["n_sup"] for q in r["points"]})
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Reasoning budget (n_sup) per denoising step")
    ax.set_ylabel("Exact-solve rate (%)")
    ax.set_title("Reusing the thinker state across denoising steps")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out)


# ── tuning boards: allocation ────────────────────────────────────────────────


def _bump_curve(S, fn, ratio, peak, width, n=400):
    """The reasoning budget (x uniform) of a Gaussian bump, as a SMOOTH curve: the same formula as
    ablate_trm_loop_budget._matched_budget_schedule before integer rounding (floor `lo`, height `hi`, rescaled so the
    total over the S denoising steps equals the uniform budget). The run itself used the rounded integers, min 1."""
    lo = max(1, round(fn / ratio))
    hi = max(lo + 1, round(fn * ratio))
    sigma = max(width * S, 1e-6)
    center = peak * (S - 1)

    def raw(pos):
        return lo + (hi - lo) * np.exp(-0.5 * ((pos - center) / sigma) ** 2)

    scale = (fn * S) / raw(np.arange(S)).sum()
    pos = np.linspace(0, S - 1, n)
    return 100.0 * (1 - pos / (S - 1)), raw(pos) * scale / fn


_SERIF = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix"}
_GOOD, _BAD, _NEUTRAL = "#2e8b3a", "#a11d1d", "#7a7a7a"


def _load_alloc(d: Path, task: str):
    """Bump grid at flat_n=16 (base PaTh, fresh state every step): merge tune_split/<task>_alloc16_*.json. Returns
    (flat, bumps, S, fn, ratio) with flat = mean of the (identical-config) flat baselines, or None."""
    files = sorted((d / "tune_split").glob(f"{task}_alloc16_*.json"))
    rs = [json.loads(f.read_text()) for f in files]
    if not rs:
        return None
    pts = [q for r in rs for q in r["points"]]
    flats = [q for q in pts if q["direction"] == "flat"]
    bumps = {(q["peak"], q["width"]): q for q in pts if q["direction"] == "bump"}
    if not flats or not bumps:
        return None
    S = len(flats[0]["schedule"])
    flat_exact = float(np.mean([q["exact"] for q in flats]))
    return flat_exact, list(bumps.values()), S, float(np.mean(flats[0]["schedule"])), float(rs[0].get("alloc_ratio", 3.0))


def fig_alloc_heatmap(d: Path, task: str, out: Path):
    """Sudoku-style: delta exact vs the uniform schedule (same total reasoning), rows = width, columns = peak."""
    L = _load_alloc(d, task)
    if L is None:
        return None
    flat, bumps, S, fn, ratio = L
    peaks = sorted({q["peak"] for q in bumps})
    widths = sorted({q["width"] for q in bumps})
    diff = np.full((len(widths), len(peaks)), np.nan)
    for q in bumps:
        diff[widths.index(q["width"]), peaks.index(q["peak"])] = _pct(q["exact"] - flat)
    vabs = max(1.0, np.nanmax(np.abs(diff)))
    with plt.rc_context({**_SERIF, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14}):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.imshow(diff, cmap="RdBu", vmin=-vabs, vmax=vabs, aspect="auto")
        ax.set_xticks(range(len(peaks)), [f"{p:g}" for p in peaks])
        ax.set_yticks(range(len(widths)), [f"{w:g}" for w in widths])
        ax.set_xlabel("Peak")
        ax.set_ylabel("Width")
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(len(widths)):
            for j in range(len(peaks)):
                v = diff[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=15,
                            color="white" if abs(v) > 0.6 * vabs else "black")
        fig.tight_layout(pad=0.3)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return out


def fig_alloc_curves(d: Path, task: str, out: Path):
    """The schedules behind the heatmap: budget (x uniform) along the trajectory, one panel per width, one curve per
    peak, bold number = delta exact vs uniform (points). Green = better than uniform, red = worse, grey = within noise."""
    L = _load_alloc(d, task)
    if L is None:
        return None
    flat, bumps, S, fn, ratio = L
    peaks = sorted({q["peak"] for q in bumps})
    widths = sorted({q["width"] for q in bumps})
    with plt.rc_context({**_SERIF, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(widths), figsize=(3.6 * len(widths), 3.0), sharey=True, squeeze=False)
        for ax, w in zip(axes[0], widths):
            ax.axhline(1.0, color="0.55", ls="--", lw=1)
            ax.text(99, 1.0, "uniform", ha="right", va="bottom", fontsize=9, color="0.35",
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.85), zorder=5)
            for pk in peaks:
                q = next((b for b in bumps if b["peak"] == pk and b["width"] == w), None)
                if q is None:
                    continue
                dv = _pct(q["exact"] - flat)
                c = _GOOD if dv > 0.7 else _BAD if dv < -0.7 else _NEUTRAL
                x, y = _bump_curve(S, int(round(fn)), ratio, pk, w)
                x = 100.0 - x
                ax.plot(x, y, color=c, lw=2)
                i = int(np.argmax(y))
                ax.annotate(f"{dv:+.1f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 4), ha="center",
                            fontsize=11, fontweight="bold", color=c)
            ax.set_title(f"width {w:g}", fontsize=12)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Sampling trajectory (%)", fontsize=12)
            ax.set_ylim(0, None)
        axes[0][0].set_ylabel("Recursion budget (\u00d7 uniform)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["maze"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="PaTh-rebuttal")
    ap.add_argument("--wandb-group", default="flops")
    ap.add_argument("--wandb-name", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    out_dir = Path(a.out_dir) if a.out_dir else d / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs_all = {}
    for task in a.tasks:
        o = lambda name: out_dir / f"{task}_{name}.png"
        figs = {
            "01_z_change": fig_z_change(d, task, o("01_z_change")),
            "02_cosine": fig_cosine(d, task, o("02_cosine")),
            "03_flops_vs_exact": fig_flops_vs_exact(d, task, o("03_flops_vs_exact")),
            "04_flops_by_size": _fig_by(d, task, "per_scale", o("04_flops_by_size"), f"{task}: PaTh, by board size"),
            "04_flops_by_shape": _fig_by(d, task, "per_shape", o("04_flops_by_shape"), f"{task}: PaTh, by shape"),
            "05_cut_at_k": fig_cut(d, task, o("05_cut_at_k")),
            "05_cut_at_k_by_size": fig_cut_by_size(d, task, o("05_cut_at_k_by_size")),
            "06_carry": fig_carry(d, task, o("06_carry")),
            "07_allocation_heatmap": fig_alloc_heatmap(d, task, o("07_allocation_heatmap")),
            "07_allocation_curves": fig_alloc_curves(d, task, o("07_allocation_curves")),
        }
        for k, p in figs.items():
            print(f"{task}/{k}: {p}")
            if p:
                figs_all[f"{task}/{k}"] = p

    if a.no_wandb or not figs_all:
        return
    try:
        import wandb
        kw = dict(project=a.wandb_project, group=a.wandb_group, job_type="analysis", tags=["flops", "report", "figures_v2"],
                  name=a.wandb_name or "flops_figures_" + "_".join(a.tasks))
        try:
            run = wandb.init(settings=wandb.Settings(init_timeout=90), **kw)
        except Exception as e:
            print(f"wandb online failed ({e!r}); offline")
            run = wandb.init(mode="offline", **kw)
        run.log({k: wandb.Image(str(p)) for k, p in figs_all.items()})
        run.finish()
    except Exception as e:
        print(f"wandb unavailable ({e!r}); figures only")


if __name__ == "__main__":
    main()
