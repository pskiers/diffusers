#!/bin/bash -l
#SBATCH --job-name=amaze_verify_eval
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# Check that the generalisation evaluation on THIS cluster reproduces the final
# results: re-run the exact evaluation train_generalisation.sh uses (same script,
# config, data, seeds) on the final PaTh thinker, then compare it cell by cell with
# the reference JSON from Helios (runs/pt_maze_final_thinker/amaze_metrics_maze.json)
# and the generated images pixel by pixel with Helios's saved generations.
#
# The checkpoint is copied to runs/<VERIFY_DIR>/ so the reference run's generations
# and JSON are never overwritten.
#
# Usage (from the trm_diffusion dir; on Athena add --account/--partition, see train_generalisation.sh):
#   sbatch slurm_scripts/sample_amaze/verify_eval_cluster.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
REF_DIR="runs/pt_maze_final_thinker"
VERIFY_DIR="runs/${VERIFY_DIR:-verify_eval_pt_maze_final_thinker}"
mkdir -p "${VERIFY_DIR}"
[[ -f "${VERIFY_DIR}/checkpoint_final.pt" ]] || cp "${REF_DIR}/checkpoint_final.pt" "${VERIFY_DIR}/"

# The evaluation half of train_generalisation.sh, unchanged, on the copied checkpoint.
# No wandb: this is a check, not a result.
export WANDB_MODE=disabled
# The reference JSON has no OOD table (it predates size 10), so skip OOD here.
export MAZE_OOD_SCALES=""
# SKIP_EVAL=true: only redo the comparison below on an existing ${VERIFY_DIR}.
if [[ "${SKIP_EVAL:-false}" != true ]]; then
  RUN_NAME="$(basename "${VERIFY_DIR}")" KIND=trm \
    bash slurm_scripts/sample_amaze/train_generalisation.sh square_8 trm train=false eval=true
fi

# The evaluation activates the venv in its own shell; the comparison needs it here too.
source slurm_scripts/cluster_env.sh
cluster_activate
python - "${REF_DIR}" "${VERIFY_DIR}" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ref_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
ref = json.loads((ref_dir / "amaze_metrics_maze.json").read_text())
new = json.loads((new_dir / "amaze_metrics_maze.json").read_text())

def cells(d):
    out = {}
    for g, by_size in d.get("per_shape", {}).items():
        for s, m in by_size.items():
            out[f"{g}{s}"] = m
    for k in ("overall", "overall_ood"):
        if k in d:
            out[k] = d[k]
    return out

r, n = cells(ref), cells(new)
print(f"{'cell':14s} {'exact1 ref':>10s} {'exact1 new':>10s} {'pass1 ref':>10s} {'pass1 new':>10s}")
worst = 0.0
for k in sorted(set(r) & set(n)):
    e_r, e_n = r[k].get("exact1"), n[k].get("exact1")
    p_r, p_n = r[k].get("pass1"), n[k].get("pass1")
    if None in (e_r, e_n, p_r, p_n):
        continue
    worst = max(worst, abs(e_r - e_n), abs(p_r - p_n))
    print(f"{k:14s} {e_r:10.4f} {e_n:10.4f} {p_r:10.4f} {p_n:10.4f}")
missing = sorted(set(r) ^ set(n))
if missing:
    print("cells only in one of the two JSONs:", missing)
print(f">> largest |difference| in exact1/pass1 over all cells: {worst:.4f}")

# Same seeds -> the images should be (nearly) identical; numeric differences
# between GPU types can only move pixels slightly.
ref_imgs = sorted((ref_dir / "generations").rglob("*.png"))
diffs, missing_imgs = [], 0
for p in ref_imgs:
    q = new_dir / "generations" / p.relative_to(ref_dir / "generations")
    if not q.exists():
        missing_imgs += 1
        continue
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(q).convert("RGB"), dtype=np.int16)
    diffs.append(np.abs(a - b).mean())
diffs = np.array(diffs)
print(f">> images compared: {len(diffs)} (missing here: {missing_imgs}); "
      f"identical: {(diffs == 0).mean():.3f}; mean abs pixel diff {diffs.mean():.3f}/255, "
      f"99th pct {np.percentile(diffs, 99):.3f}, max {diffs.max():.3f}")
EOF
