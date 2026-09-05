"""experiments/report_maze_corruption.py — tables and figures for the wrong-path probe.

Reads the per-model JSONs written by maze_corruption_recovery_probe.py and
emits, into --out-dir:

    summary.csv            one row per (model, mode, level, t_start) — everything.
    summary.md             the same as readable tables, one per corruption mode.
    heatmap_<mode>.png     level x t_start recovery, one panel per model + a
                           TRM-minus-DiT difference panel.
    curves_<mode>.png      recovery vs t_start, one line per level, both models.
    references_<mode>.png  Pass@1 floor / denoised / ceiling — shows whether the
                           model actually got closer to a valid solve, not just
                           whether it moved pixels.
    qualitative_<mode>_t<t>.png
                           GT | clean context | corrupted | denoised | denoised-
                           from-clean strips, built from the probe's dumped PNGs.

Usage:
    python experiments/report_maze_corruption.py \
        --runs trm=runs/maze_corruption/trm.json dit=runs/maze_corruption/dit.json \
        --out-dir runs/maze_corruption/report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODES = ("add", "wall", "gap")
MODE_TITLE = {
    "add": "ADD — wrong path walked off the drawn prefix, run to a dead end",
    "wall": "WALL — straight shortcut to the target, through walls",
    "gap": "GAP — a contiguous slice of the GT path erased",
}
# What "recovery" means per mode, for axis labels — the probe measures three
# different things under one key and a plot that does not say so is misleading.
RECOVERY_LABEL = {
    "add": "wrong cells erased",
    "wall": "shortcut pixels erased",
    "gap": "erased cells redrawn",
}
KEY_RE = re.compile(r"^t=(\d+)/level=(\d+)/(\w+)$")

# Columns every mode shares, then the ones only one mode reports.
BASE_COLS = [("recovery_rate", "rec"), ("full_recovery_rate", "full"),
             ("floor_pass", "floor@1"), ("denoised_pass", "den@1"), ("ceiling_pass", "ceil@1"),
             ("denoised_violation", "den viol"), ("collateral_break_rate", "collat"),
             ("remaining_gt_recall", "tail rec"), ("mean_corruption_size", "size"),
             ("applied_rate", "applied")]
EXTRA_COLS = {
    "add": [("adapt_rate", "adapt"), ("mean_gt_walk", "gt walk")],
    "wall": [("still_through_wall_rate", "still wall")],
    "gap": [],
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def rows_from(model: str, blob: dict) -> list:
    """Flatten one probe JSON into (model, mode, level, t_start) rows."""
    out = []
    for key, entry in blob["results"].items():
        m = KEY_RE.match(key)
        if not m:
            continue
        t, level, mode = int(m.group(1)), int(m.group(2)), m.group(3)
        if mode == "ceiling":
            continue
        row = {"model": model, "mode": mode, "level": level, "t_start": t}
        for ref in ("floor", "denoised", "ceiling"):
            sub = entry.get(ref) or {}
            for k in ("coverage", "violation", "pass", "n"):
                row[f"{ref}_{k}"] = sub.get(k)
        for k, v in entry.items():
            if isinstance(v, (int, float)):
                row[k] = v
        out.append(row)
    return sorted(out, key=lambda r: (r["mode"], r["level"], r["t_start"]))


def grid(rows: list, model: str, mode: str, field: str) -> tuple:
    """(matrix[level, t_start], levels, t_starts) — NaN where a cell is missing
    or had no board the corruption actually applied to."""
    sel = [r for r in rows if r["model"] == model and r["mode"] == mode]
    levels = sorted({r["level"] for r in sel})
    ts = sorted({r["t_start"] for r in sel})
    mat = np.full((len(levels), len(ts)), np.nan)
    for r in sel:
        if r.get("n_applied", 1):
            mat[levels.index(r["level"]), ts.index(r["t_start"])] = r.get(field, np.nan)
    return mat, levels, ts


def write_tables(rows: list, models: list, out_dir: Path) -> None:
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                    ("model", "mode", "level", "t_start"), k))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = ["# Wrong-path recovery probe", ""]
    for mode in MODES:
        cols = BASE_COLS + EXTRA_COLS[mode]
        lines += [f"## {MODE_TITLE[mode]}", "",
                  f"`rec` = {RECOVERY_LABEL[mode]}. `floor@1`/`den@1`/`ceil@1` = Pass@1 of the "
                  "corrupted input as-is / after denoising / of the same board denoised from a "
                  "clean context. `size` = mean corruption size (cells, or shortcut pixels for "
                  "WALL). `applied` = fraction of boards the corruption could be built on."
                  + (" `gt walk` = correct cells the wrong branch had to follow before an "
                     "opening existed, i.e. how far past the nominal level the drawn context "
                     "actually reaches." if mode == "add" else ""), ""]
        head = "| model | level | t | " + " | ".join(c[1] for c in cols) + " |"
        lines += [head, "|" + "---|" * (len(cols) + 3)]
        for model in models:
            for r in sorted([x for x in rows if x["model"] == model and x["mode"] == mode],
                            key=lambda x: (x["level"], x["t_start"])):
                # A cell the corruption could never be built on has no
                # measurement, only a divide-by-one zero — print it as missing
                # rather than as a real 0.000.
                empty = not r.get("n_applied", 1)
                cells = []
                for k, _ in cols:
                    v = r.get(k)
                    blank = v is None or (empty and k != "applied_rate")
                    cells.append("—" if blank else f"{v:.3f}")
                lines.append(f"| {model} | {r['level']}% | {r['t_start']} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def _annotate(ax, mat, levels, ts) -> None:
    ax.set_xticks(range(len(ts)), [str(t) for t in ts])
    ax.set_yticks(range(len(levels)), [f"{l}%" for l in levels])
    ax.set_xlabel("t_start (noise injected)")
    ax.set_ylabel("level")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.5 else "black")


def heatmaps(rows: list, models: list, out_dir: Path) -> None:
    for mode in MODES:
        n = len(models) + (1 if len(models) == 2 else 0)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
        mats, levels, ts = [], [], []
        for i, model in enumerate(models):
            mat, levels, ts = grid(rows, model, mode, "recovery_rate")
            mats.append(mat)
            ax = axes[0][i]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_title(f"{model.upper()} — {RECOVERY_LABEL[mode]}")
            _annotate(ax, mat, levels, ts)
            fig.colorbar(im, ax=ax, fraction=0.046)
        if len(models) == 2:
            diff = mats[0] - mats[1]
            ax = axes[0][-1]
            lim = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
            im = ax.imshow(diff, vmin=-lim, vmax=lim, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{models[0].upper()} − {models[1].upper()}")
            _annotate(ax, diff, levels, ts)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(MODE_TITLE[mode])
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{mode}.png", dpi=150)
        plt.close(fig)


def curves(rows: list, models: list, out_dir: Path) -> None:
    for mode in MODES:
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6),
                                 squeeze=False, sharey=True)
        for i, model in enumerate(models):
            mat, levels, ts = grid(rows, model, mode, "recovery_rate")
            ax = axes[0][i]
            for j, lv in enumerate(levels):
                ax.plot(ts, mat[j], marker="o", label=f"{lv}%")
            ax.set_title(model.upper())
            ax.set_xlabel("t_start")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel(RECOVERY_LABEL[mode])
            ax.legend(title="level", fontsize=8)
        fig.suptitle(MODE_TITLE[mode])
        fig.tight_layout()
        fig.savefig(out_dir / f"curves_{mode}.png", dpi=150)
        plt.close(fig)


def references(rows: list, models: list, out_dir: Path) -> None:
    """floor / denoised / ceiling Pass@1 side by side.

    recovery_rate on its own is gameable — a model that floods the board with
    blue "recovers" every erased GAP cell. These three bars say whether the
    denoised board is actually a valid solution, and how far it is from what the
    same model produces from an uncorrupted start.
    """
    for mode in MODES:
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8),
                                 squeeze=False, sharey=True)
        for i, model in enumerate(models):
            sel = sorted([r for r in rows if r["model"] == model and r["mode"] == mode
                          and r.get("n_applied", 0)], key=lambda r: (r["level"], r["t_start"]))
            ax = axes[0][i]
            x = np.arange(len(sel))
            for k, (field, lab) in enumerate([("floor_pass", "floor (no denoise)"),
                                              ("denoised_pass", "denoised"),
                                              ("ceiling_pass", "ceiling (clean ctx)")]):
                ax.bar(x + (k - 1) * 0.27, [r.get(field) or 0.0 for r in sel], width=0.27, label=lab)
            ax.set_xticks(x, [f"{r['level']}%\nt{r['t_start']}" for r in sel], fontsize=6)
            ax.set_title(model.upper())
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            if i == 0:
                ax.set_ylabel("Pass@1")
            ax.legend(fontsize=8)
        fig.suptitle(MODE_TITLE[mode] + " — exact-solve rate")
        fig.tight_layout()
        fig.savefig(out_dir / f"references_{mode}.png", dpi=150)
        plt.close(fig)


def qualitative(blobs: dict, out_dir: Path, per_cell: int = 1) -> None:
    """GT | clean context | corrupted | denoised | denoised-from-clean strips.

    One figure per (mode, t_start): rows are level x model, which keeps each
    sheet readable. A single figure over every t_start would be metres tall.
    """
    stages = [("gt", "GT solution"), ("clean", "clean context"), ("corrupt", "corrupted"),
              ("out", "denoised"), ("ceilout", "denoised, clean ctx")]
    t_starts = sorted({t for b in blobs.values() for t in b.get("t_starts", [])})
    for mode in MODES:
        for t in t_starts:
            panels = []
            for model, blob in blobs.items():
                dump = Path(blob.get("dump_dir") or "")
                if not dump.is_dir():
                    continue
                for lv in blob.get("levels", {}).get(mode, []):
                    for idx in range(per_cell):
                        stem = dump / f"{blob['model']}_t{t}_{mode}_p{int(lv * 100)}_{idx}"
                        files = [Path(f"{stem}_{s}.png") for s, _ in stages]
                        if all(f.exists() for f in files):
                            panels.append((f"{model}\n{int(lv * 100)}%", files))
            if not panels:
                continue
            fig, axes = plt.subplots(len(panels), len(stages),
                                     figsize=(2.1 * len(stages), 2.1 * len(panels)), squeeze=False)
            for r, (label, files) in enumerate(panels):
                for c, f in enumerate(files):
                    ax = axes[r][c]
                    ax.imshow(Image.open(f).resize((256, 256)))
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(stages[c][1], fontsize=9)
                    if c == 0:
                        ax.set_ylabel(label, fontsize=8)
            fig.suptitle(f"{MODE_TITLE[mode]}  —  t_start = {t}")
            fig.tight_layout()
            fig.savefig(out_dir / f"qualitative_{mode}_t{t}.png", dpi=120)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=path.json pairs, e.g. trm=runs/.../trm.json dit=.../dit.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--qualitative-per-cell", type=int, default=1)
    args = ap.parse_args()

    blobs, rows, models = {}, [], []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--runs takes name=path pairs, got {spec!r}")
        blob = load(path)
        blobs[name] = blob
        models.append(name)
        rows += rows_from(name, blob)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(rows, models, out_dir)
    heatmaps(rows, models, out_dir)
    curves(rows, models, out_dir)
    references(rows, models, out_dir)
    qualitative(blobs, out_dir, args.qualitative_per_cell)
    print(f"report -> {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
