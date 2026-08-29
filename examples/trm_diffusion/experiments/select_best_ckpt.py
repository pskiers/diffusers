#!/usr/bin/env python3
"""Print the fine-tuning checkpoint with the lowest validation MSE.

The FT trainers record the most recent validation metrics next to each checkpoint:
  - Janus: ``<run_dir>/**/checkpoint-<epoch>-<step>/training_state.json`` (keys val_mse/val_loss)
  - Bagel: ``<checkpoints_dir>/<step>/val.json``                          (keys val_mse/val_ce)

Selection mirrors the AMAZE paper's early-stopping criterion (validation MSE). Usage:
  select_best_ckpt.py janus <run_dir>          -> prints the winning .../tfmr directory
  select_best_ckpt.py bagel <checkpoints_dir>  -> prints the winning .../<step> directory

Prints an empty line when no validation metrics were recorded, so the caller can fall back
(e.g. score the last checkpoint instead).
"""
import glob
import json
import os
import sys


def _val(d: dict):
    """Validation score to minimise: prefer MSE (paper criterion), else the CE/total loss."""
    v = d.get("val_mse")
    return d.get("val_loss", d.get("val_ce")) if v is None else v


def main() -> int:
    if len(sys.argv) != 3:
        print("")
        return 0
    backend, root = sys.argv[1], sys.argv[2]

    if backend == "janus":
        metric_files = glob.glob(os.path.join(root, "**", "training_state.json"), recursive=True)

        def ckpt_of(path: str) -> str:
            return os.path.join(os.path.dirname(path), "tfmr")
    else:  # bagel
        metric_files = glob.glob(os.path.join(root, "*", "val.json"))

        def ckpt_of(path: str) -> str:
            return os.path.dirname(path)

    best = None
    for path in metric_files:
        try:
            with open(path) as f:
                score = _val(json.load(f))
        except (OSError, ValueError):
            continue
        ckpt = ckpt_of(path)
        if score is None or not os.path.isdir(ckpt):
            continue
        if best is None or score < best[0]:
            best = (score, ckpt)

    print(best[1] if best else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
