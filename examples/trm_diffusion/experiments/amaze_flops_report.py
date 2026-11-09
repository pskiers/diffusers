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


def fig_alloc(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_alloc.json")
    if not r:
        return None
    pts = r["points"]
    flat = next((q for q in pts if q["direction"] == "flat"), None)
    bumps = [q for q in pts if q["direction"] == "bump"]
    prof = next((q for q in pts if q["direction"] == "profile"), None)
    if flat is None or not bumps:
        return None
    S = len(flat["schedule"])
    uni = float(np.mean(flat["schedule"]))
    x = [100.0 * (1 - i / (S - 1)) for i in range(S)]          # % of the sampling trajectory remaining
    delta = {id(q): _pct(q["exact"] - flat["exact"]) for q in pts}
    vmax = max(1.0, max(abs(v) for v in delta.values()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")
    widths = sorted({q["width"] for q in bumps})
    peaks = sorted({q["peak"] for q in bumps})
    nrow = len(widths) + (1 if prof else 0)
    fig, axes = plt.subplots(nrow, 1, figsize=(8.6, 2.5 * nrow + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]

    def draw(ax, q, label_x=None):
        y = [n / uni for n in q["schedule"]]
        c = cmap(norm(delta[id(q)]))
        ax.plot(x, y, "-o", color=c, ms=3)
        ax.fill_between(x, 0, y, color=c, alpha=0.18)
        i = int(np.argmax(y))
        ax.annotate(f"{delta[id(q)]:+.1f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8, color="0.2")

    for ax, w in zip(axes, widths):
        for pk in peaks:
            q = next((b for b in bumps if b["width"] == w and b["peak"] == pk), None)
            if q:
                draw(ax, q)
        ax.axhline(1.0, color="0.4", ls=":", lw=1)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.0) * 1.18)
        ax.set_ylabel("budget (x uniform)")
        ax.text(0.99, 0.92, f"width = {w:g}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if prof:
        ax = axes[-1]
        draw(ax, prof)
        ax.axhline(1.0, color="0.4", ls=":", lw=1)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.0) * 1.18)
        ax.set_ylabel("budget (x uniform)")
        ax.text(0.99, 0.92, f"from z_H change (eps = {r.get('alloc_profile_eps', '?')})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
    axes[-1].set_xlim(100, 0)
    axes[-1].set_xlabel("Sampling trajectory (% of sampling remaining)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    fig.colorbar(sm, cax=cax).set_label("change in exact vs uniform (points)")
    fig.suptitle(f"Same total reasoning (avg n_sup {uni:g}) moved along the trajectory; "
                 f"uniform exact = {_pct(flat['exact']):.1f}%", fontsize=10, y=0.995)
    fig.savefig(out, dpi=150)
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
            "07_allocation": fig_alloc(d, task, o("07_allocation")),
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
