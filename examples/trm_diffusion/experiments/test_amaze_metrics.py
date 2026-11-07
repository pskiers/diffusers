"""Regression test for the patched AMAZE scorer.

Scores the GROUND-TRUTH solution image for every geometry x size. A correct
scorer must return Coverage=1, Violation=0, Pass=1, Exact=1 on the GT itself --
if it does not, the metric is broken before any model is involved.

Run on an aarch64 compute node (the venv is GH200-only):
    sbatch slurm_scripts/test_metrics.sh
or interactively after `source $SCRATCH/trm_helios_venv/bin/activate`:
    python -m experiments.test_amaze_metrics
"""
import os
import sys

import numpy as np
from torchvision import transforms

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))

from datasets.amaze_dataset import AmazeDataset          # noqa: E402
from eval.amaze_eval import AmazeMetrics, MAZE_GEOMETRIES  # noqa: E402

DATA = os.environ.get("MAZE_DATA", "data/amaze/maze")
SIZES = [int(x) for x in os.environ.get("TEST_SCALES", "5,7,8,9,11,13,16").split(",") if x.strip()]
N = int(os.environ.get("TEST_N", "10"))

scorer = AmazeMetrics(task="maze")
to_tensor = transforms.ToTensor()
fails = []

print("%-9s %-4s %9s %10s %7s %7s" % ("geom", "n", "Coverage", "Violation", "Pass", "Exact"))
print("-" * 54)

for geom in MAZE_GEOMETRIES:
    for n in SIZES:
        pq = os.path.join(DATA, geom, "n%d_test.parquet" % n)
        if not os.path.exists(pq):
            print("%-9s %-4d  (no parquet, skipped)" % (geom, n))
            continue

        ds = AmazeDataset(pq, split="test", image_size=144,
                          condition_field="m_original_img", target_field="sol_img",
                          num_channels=3, include_metadata=True)

        rows = []
        for i in range(min(N, len(ds))):
            meta = ds[i].metadata
            sol = meta.get("sol_img")
            if sol is None:
                continue
            # Score the ground truth as if the model had produced it.
            rows.append(scorer._compute_maze_metrics(to_tensor(sol.convert("RGB")), meta))

        if not rows:
            print("%-9s %-4d  (no scorable rows)" % (geom, n))
            continue

        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        cov, vio = agg["gt_cell_coverage"], agg["background_violation"]
        ps, ex = agg["pass"], agg["exact"]
        bad = not (cov > 0.999 and vio < 1e-6 and ps > 0.999 and ex > 0.999)
        if bad:
            fails.append((geom, n, cov, vio, ps, ex))
        print("%-9s %-4d %9.4f %10.4f %7.3f %7.3f%s"
              % (geom, n, cov, vio, ps, ex, "   <-- FAIL" if bad else ""))

print()
if fails:
    print("FAILED on %d combo(s): the scorer cannot score its own ground truth." % len(fails))
    sys.exit(1)
print("PASS: ground truth scores perfectly on every combo (%d puzzles each)." % N)
