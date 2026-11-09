"""Re-log finished AMAZE evaluations into their wandb runs with the current metric set.

Reads <run_dir>/amaze_metrics_<task>.json (written by amaze_generate_and_calculate_metrics.py /
amaze_det_baselines.py), resumes the run in <run_dir>/wandb_run_id.txt and
  1. deletes the summary scalars of the old metric set (pass1, pass5, mse_inside, mse_outside)
     under amaze_eval/ and gen/,
  2. logs everything again through eval.amaze_eval.log_tables (coverage, violation, pass,
     exact1, exact5), with the sample images from <run_dir>/samples/<combo>/ (the saved
     "condition | generated" strip of the first test board of each cell).
No re-sampling; the JSON is the source.

python experiments/amaze_relog_eval.py --project PaTh-rebuttal runs/gen_hexagon_only_linear_head [more run dirs ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.amaze_eval import WANDB_METRICS, log_tables, maze_sample_key, queens_sample_key  # noqa: E402

OLD_SUFFIXES = ("/pass1", "/pass5", "/mse_inside", "/mse_outside")


def _samples(run_dir: Path, task: str, result: dict) -> dict:
    """{sample_key: {"generated", "condition"}} from the saved strips (condition left, generated right)."""
    import numpy as np
    import torch
    from PIL import Image

    out = {}
    for d in sorted((run_dir / "samples").glob("*")) if (run_dir / "samples").is_dir() else []:
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            continue
        a = torch.from_numpy(np.asarray(Image.open(pngs[0]).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
        w = a.shape[2] // 2
        pair = {"condition": a[:, :, :w], "generated": a[:, :, w:]}
        if task == "maze":
            g, s = d.name.rsplit("_", 1)
            out[maze_sample_key(g, s)] = pair
        else:
            out[queens_sample_key(d.name.lstrip("n"))] = pair
    return out


def relog(run_dir: Path, project: str, entity: str | None):
    import wandb

    jsons = sorted(run_dir.glob("amaze_metrics_*.json"))
    if not jsons:
        print(f"{run_dir}: no amaze_metrics_*.json, skipped")
        return
    result = json.loads(jsons[0].read_text())
    task = result.get("task") or jsons[0].stem.split("_")[-1]
    run_id = (run_dir / "wandb_run_id.txt").read_text().strip()

    api = wandb.Api()
    path = f"{entity or api.default_entity}/{project}/{run_id}"
    old = api.run(path)
    stale = [k for k in old.summary.keys()
             if k.startswith(("amaze_eval/", "gen/")) and k.endswith(OLD_SUFFIXES)]
    for k in stale:
        del old.summary[k]
    old.summary.update()

    run = wandb.init(project=project, entity=entity, id=run_id, resume="must")
    try:
        log_tables(run, task, result, _samples(run_dir, task, result))
        run.summary["amaze_eval/wandb_metrics"] = ",".join(WANDB_METRICS)
    finally:
        run.finish()
    print(f"{run_dir.name}: removed {len(stale)} old summary keys, re-logged ({run_id})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--project", default="PaTh-rebuttal")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    a = ap.parse_args()
    for d in a.run_dirs:
        relog(Path(d), a.project, a.entity)


if __name__ == "__main__":
    main()
