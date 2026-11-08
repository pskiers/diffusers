"""experiments/amaze_flops_report.py — figures for the AMAZE FLOPs / "where to cut" sweeps.

Reads the JSONs written by experiments/amaze_flops_sweep.py under <dir>/{report_split,tune_split}/ (any missing file
is skipped) and writes PNGs to <dir>/report/ (and to wandb, project amaze_flops, unless --no-wandb):

  report split (held-out boards; feeds the final plot)
    <task>_flops_vs_acc.png     exact-solve rate vs real FLOPs: every DiT size (dm) + PaTh flat (+ PaTh wise if run)
    <task>_flops_by_size.png    PaTh flat: accuracy vs FLOPs, one line per board size
    <task>_flops_by_shape.png   PaTh flat: accuracy vs FLOPs, one line per shape
  tune split (phase 1: used to DEFINE the wise TRM)
    <task>_p1_profile.png       relative z_H change per (denoising step, reasoning iteration); iterations needed to
                                fall below each eps, per denoising step; same-board z_H cosine between steps
    <task>_p1_cut_at_k.png      accuracy vs fraction of denoising steps that reason (+ vs FLOPs)
    <task>_p1_converge.png      accuracy vs FLOPs for in-loop early exit (eps x fresh/carry-in-regime); iterations/step
    <task>_p1_carry.png         accuracy vs n_sup for fresh / carry-everywhere / carry-within-regime

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

COL_TRM = "#0072B2"
DM_COLORS = ["#D55E00", "#CC79A7", "#999999", "#E69F00"]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── report split ─────────────────────────────────────────────────────────────


def fig_flops_vs_acc(d: Path, task: str, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    drawn = False
    for i, p in enumerate(sorted((d / "report_split").glob(f"{task}_dm_*.json"))):
        r = _load(p)
        pts = sorted(r["points"], key=lambda q: q["flops"])
        label = f"DiT {p.stem.split('_dm_', 1)[1]}"
        ax.plot([q["flops"] for q in pts], [100 * q["exact"] for q in pts], "-*", color=DM_COLORS[i % 4], label=label,
                lw=2, ms=10)
        drawn = True
    for mode, label, color in (("trm_flat", "PaTh (trained policy)", "#56B4E9"), ("trm_wise", "PaTh wise", COL_TRM)):
        r = _load(d / "report_split" / f"{task}_{mode}.json")
        if r:
            pts = sorted(r["points"], key=lambda q: q["flops"])
            ax.plot([q["flops"] for q in pts], [100 * q["exact"] for q in pts], "-*", color=color, label=label, lw=2, ms=10)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per sample (FlopCounterMode)")
    ax.set_ylabel("Exact-solve rate (%), held-out boards")
    ax.set_title(f"{task}: accuracy vs compute")
    ax.grid(alpha=0.3)
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
        ax.plot([q["flops"] for q in pts], [100 * q[key][n]["exact"] for q in pts], "-o",
                color=cmap(i / max(1, len(names) - 1)), label=n)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per sample")
    ax.set_ylabel("Exact-solve rate (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out)


# ── phase 1 (tune split) ─────────────────────────────────────────────────────


def fig_profile(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_profile.json")
    if not r or "profile" not in r:
        return None, {}
    pr, ts = r["profile"], r["timesteps"]
    rel = [row for row in pr["rel_change_mean"]]
    n_it = max(len(row) for row in rel)
    M = np.array([row + [np.nan] * (n_it - len(row)) for row in rel])
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    im = axes[0].imshow(np.log10(np.clip(M, 1e-6, None)), aspect="auto", cmap="viridis_r")
    axes[0].set_yticks(range(len(ts)), [str(t) for t in ts])
    axes[0].set_xticks(range(n_it), [str(i + 1) for i in range(n_it)])
    axes[0].set_xlabel("reasoning iteration (n_sup index)")
    axes[0].set_ylabel("denoising timestep t (first = pure noise)")
    axes[0].set_title("log10 mean relative z_H change per iteration")
    fig.colorbar(im, ax=axes[0])
    summary = {}
    for eps, c in ((0.1, "#D55E00"), (0.03, "#0072B2"), (0.01, "#009E73"), (0.003, "#CC79A7")):
        need = [next((i + 1 for i, v in enumerate(row) if v < eps), len(row)) for row in rel]
        axes[1].plot(range(len(ts)), need, "-o", color=c, label=f"eps={eps}")
        summary[f"iters_to_converge/eps={eps}"] = float(np.mean(need))
    axes[1].set_xticks(range(len(ts)), [str(t) for t in ts], rotation=90, fontsize=7)
    axes[1].set_xlabel("denoising timestep t")
    axes[1].set_ylabel("iterations until mean change < eps (capped at n_sup)")
    axes[1].set_title("reasoning iterations needed per denoising step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    for ax, key, title in ((axes[2], "cos_raw", "same-board cosine of final z_H (raw)"),
                           (axes[3], "cos_centered", "same-board cosine (board-specific part)")):
        if key in pr:
            C = np.array(pr[key])
            im = ax.imshow(C, cmap="Blues", vmin=min(0.5, np.nanmin(C)), vmax=1.0, origin="lower")
            ax.set_xticks(range(len(ts)), [str(t) for t in ts], rotation=90, fontsize=7)
            ax.set_yticks(range(len(ts)), [str(t) for t in ts], fontsize=7)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            adj = [C[i, i + 1] for i in range(len(ts) - 1)]
            summary[f"{key}/min_adjacent"] = float(min(adj))
            summary[f"{key}/argmin_adjacent_step"] = int(np.argmin(adj)) + 1  # first step of the new regime
    return _save(fig, out), summary


def fig_cut(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_cut_at_k.json")
    if not r:
        return None
    pts = sorted(r["points"], key=lambda q: q["reason_frac"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fr = [q["reason_frac"] for q in pts]
    axes[0].plot(fr, [100 * q["exact"] for q in pts], "-o", color="k", lw=2.5, label="all boards")
    sizes = sorted({s for q in pts for s in q["per_scale"]}, key=int)
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(sizes):
        axes[0].plot(fr, [100 * q["per_scale"][s]["exact"] for q in pts], "-", alpha=0.7,
                     color=cmap(i / max(1, len(sizes) - 1)), label=f"size {s}")
    axes[0].set_xlabel("fraction of denoising steps that run the thinker (rest reuse the last logits)")
    axes[0].set_ylabel("Exact-solve rate (%)")
    axes[0].set_title("where to cut: reasoning stopped after the first k% of steps")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot([q["flops"] for q in pts], [100 * q["exact"] for q in pts], "-o", color=COL_TRM)
    for q in pts:
        axes[1].annotate(f"{q['reason_frac']:g}", (q["flops"], 100 * q["exact"]), fontsize=8)
    axes[1].set_xlabel("FLOPs per sample")
    axes[1].set_ylabel("Exact-solve rate (%)")
    axes[1].set_title("same, vs FLOPs (labels = reason fraction)")
    axes[1].grid(alpha=0.3)
    return _save(fig, out)


def fig_converge(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_converge.json")
    if not r:
        return None
    ref = None
    cut = _load(d / "tune_split" / f"{task}_cut_at_k.json")
    if cut:
        ref = next((q for q in cut["points"] if q["reason_frac"] == 1.0), None)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    marker = {"fresh": "o", "carry_regime": "s", "carry_all": "^"}
    for q in r["points"]:
        axes[0].scatter(q["flops"], 100 * q["exact"], marker=marker.get(q["variant"], "o"), s=70,
                        label=f"eps={q['eps']:g} {q['variant']}")
        axes[1].plot(range(len(q["iters_per_step"])), q["iters_per_step"], "-", marker=marker.get(q["variant"], "o"),
                     ms=4, label=f"eps={q['eps']:g} {q['variant']}")
    if ref:
        axes[0].scatter(ref["flops"], 100 * ref["exact"], marker="*", s=220, color="k", label="trained policy (no exit)")
    axes[0].set_xlabel("FLOPs per sample (analytic from per-sample iterations)")
    axes[0].set_ylabel("Exact-solve rate (%)")
    axes[0].set_title("exit inside the reasoning loop")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("denoising step index (0 = pure noise)")
    axes[1].set_ylabel("mean reasoning iterations used")
    axes[1].set_title("iterations actually used per denoising step")
    axes[1].grid(alpha=0.3)
    return _save(fig, out)


def fig_carry(d: Path, task: str, out: Path):
    r = _load(d / "tune_split" / f"{task}_carry.json")
    if not r:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for variant, mk in (("fresh", "o"), ("carry_all", "^"), ("carry_regime", "s")):
        pts = sorted([q for q in r["points"] if q["variant"] == variant], key=lambda q: q["n_sup"])
        axes[0].plot([q["n_sup"] for q in pts], [100 * q["exact"] for q in pts], "-" + mk, label=variant)
        axes[1].plot([q["flops"] for q in pts], [100 * q["exact"] for q in pts], "-" + mk, label=variant)
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log")
    axes[0].set_xlabel("n_sup per denoising step")
    axes[1].set_xlabel("FLOPs per sample")
    for ax in axes:
        ax.set_ylabel("Exact-solve rate (%)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("carry vs fresh state (regime boundary at step %s)" % r.get("regime_boundary_step"))
    return _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["maze"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="amaze_flops")
    ap.add_argument("--wandb-group", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    out_dir = Path(a.out_dir) if a.out_dir else d / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if not a.no_wandb:
        try:
            import wandb
            kw = dict(project=a.wandb_project, name="report_" + "_".join(a.tasks), group=a.wandb_group,
                      tags=["report", "flops"])
            try:
                run = wandb.init(settings=wandb.Settings(init_timeout=90), **kw)
            except Exception as e:
                print(f"wandb online failed ({e!r}); offline")
                run = wandb.init(mode="offline", **kw)
        except Exception as e:
            print(f"wandb unavailable ({e!r}); figures only")

    for task in a.tasks:
        prof, summary = fig_profile(d, task, out_dir / f"{task}_p1_profile.png")
        figs = {
            "flops_vs_acc": fig_flops_vs_acc(d, task, out_dir / f"{task}_flops_vs_acc.png"),
            "flops_by_size": _fig_by(d, task, "per_scale", out_dir / f"{task}_flops_by_size.png",
                                     f"{task}: PaTh, by board size"),
            "flops_by_shape": _fig_by(d, task, "per_shape", out_dir / f"{task}_flops_by_shape.png",
                                      f"{task}: PaTh, by shape"),
            "p1_profile": prof,
            "p1_cut_at_k": fig_cut(d, task, out_dir / f"{task}_p1_cut_at_k.png"),
            "p1_converge": fig_converge(d, task, out_dir / f"{task}_p1_converge.png"),
            "p1_carry": fig_carry(d, task, out_dir / f"{task}_p1_carry.png"),
        }
        for k, p in figs.items():
            print(f"{task}/{k}: {p}")
        if summary:
            (out_dir / f"{task}_p1_profile_summary.json").write_text(json.dumps(summary, indent=2))
        if run is not None:
            import wandb
            log = {f"{task}/{k}": wandb.Image(str(p)) for k, p in figs.items() if p}
            log.update({f"{task}/p1/{k}": v for k, v in summary.items()})
            run.log(log)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
