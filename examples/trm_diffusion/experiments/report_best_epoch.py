#!/usr/bin/env python3
"""Rank the per-epoch AMAZE scores of a fine-tuning run and report which epoch won.

score_amaze_images.py writes ``<gen_dir>/<tag>/amaze_metrics_<task>.json`` for every
scored checkpoint. This reads them all and prints a table ranked by the chosen metric
(Pass@1 by default), marking the best epoch — handy when a run has no validation loss
and the checkpoint must be picked on the real task metric instead.

Usage: report_best_epoch.py <gen_root> <maze|queens> [--by pass1|pass5]
"""
import argparse
import glob
import json
import os


def _pct(x):
    return f"{x * 100:6.2f}%" if isinstance(x, (int, float)) else "    n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_root", help="directory holding <tag>/amaze_metrics_<task>.json subdirs")
    ap.add_argument("task", choices=["maze", "queens"])
    ap.add_argument("--by", choices=["pass1", "pass5"], default="pass1")
    args = ap.parse_args()

    rows = []
    for jf in glob.glob(os.path.join(args.gen_root, "*", f"amaze_metrics_{args.task}.json")):
        try:
            with open(jf) as f:
                overall = json.load(f).get("overall", {})
        except (OSError, ValueError):
            continue
        rows.append({
            "tag": os.path.basename(os.path.dirname(jf)),
            "pass1": overall.get("pass1"),
            "pass5": overall.get("pass5"),
            "coverage": overall.get("coverage"),
            "violation": overall.get("violation"),
        })

    if not rows:
        print(f"No amaze_metrics_{args.task}.json found under {args.gen_root}")
        return 1

    rows.sort(key=lambda r: (r[args.by] if r[args.by] is not None else -1.0), reverse=True)

    print(f"\n=== Per-epoch ranking ({args.task}, sorted by {args.by}) ===")
    print(f"{'checkpoint':<28} {'P@1':>8} {'P@5':>8} {'cov':>8} {'viol':>8}")
    for r in rows:
        print(f"{r['tag']:<28} {_pct(r['pass1'])} {_pct(r['pass5'])} {_pct(r['coverage'])} {_pct(r['violation'])}")

    best = rows[0]
    print(f"\n>>> BEST epoch (by {args.by}): {best['tag']}  "
          f"P@1 {_pct(best['pass1']).strip()}  P@5 {_pct(best['pass5']).strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
