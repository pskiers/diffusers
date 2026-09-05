#!/bin/bash -l
#SBATCH --job-name=maze_corrupt
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Wrong-path recovery probe on 13x13 square mazes, ablated over TRM
# (Painter-Thinker) and the DiT baseline.
#   ADD  0/20/50%      wrong path walked off the prefix's frontier to a dead end
#   WALL 10/30/50/75%  straight shortcut to the target, through walls
#   GAP  10/30/50/75%  contiguous slice of the full GT path erased
# each re-noised to every t_start in T_STARTS and denoised back down.
#
# Usage:  sbatch slurm_scripts/sample_amaze/maze_corruption_recovery.sh
# Env: NUM_SAMPLES T_STARTS BATCH STEPS DATA OUT_DIR GEN_DATA DUMP_N
#      TRM_CKPT PAINTER_CKPT DIT_CKPT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
# gen_amaze.py writes MAZE_TEST_PER_SCALE (default 100) puzzles per
# (shape, scale), so 100 is the whole n13-square test split. Raising this alone
# does nothing — raise MAZE_TEST_PER_SCALE and regenerate (GEN_DATA=1) as well.
NUM_SAMPLES="${NUM_SAMPLES:-100}"
T_STARTS="${T_STARTS:-[10,30,50,70,90]}"
BATCH="${BATCH:-32}"
# Pinned for BOTH models on purpose. Left unset, each model falls back to its
# own sampling config — configs/sampling/direct.yaml gives DiT 99 steps and
# configs/sampling/cfg.yaml gives the thinker 20 — so at the same t_start the
# two would get a ~5x different denoising budget to repair the corruption with,
# and any TRM-vs-DiT gap would be unreadable. num_train_timesteps is 100
# (configs/diffusion/ddpm.yaml), so STEPS=100 puts every integer t in the
# schedule and every t_start above is valid.
# Cost scales with it: the probe runs 18 denoising passes per batch per t_start
# (7 clean-context references + 11 corrupted variants), and t_start=90 alone is
# 91 steps of those. Drop to 50 (t_starts must stay even) or 20 (multiples of 5)
# if the job runs out of wall clock — just keep both models on the same number.
STEPS="${STEPS:-100}"
# Square only: the ADD/WALL neighbour logic in maze_corruption_lib.py assumes a
# rectangular cell grid, so hex/triangle/circle boards are out of scope here.
DATA="${DATA:-data/amaze/test_maze/square/n13_square_test.parquet}"
OUT_DIR="${OUT_DIR:-runs/maze_corruption}"
DUMP_N="${DUMP_N:-5}"
# The test split is already generated on the cluster; set GEN_DATA=1 only to
# rebuild it (e.g. after changing MAZE_TEST_PER_SCALE).
GEN_DATA="${GEN_DATA:-0}"
TRM_CKPT="${TRM_CKPT:-runs/pt_maze_final_thinker/checkpoint_final.pt}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/pt_maze_final_painter/checkpoint_final.pt}"
DIT_CKPT="${DIT_CKPT:-runs/dit_maze_final/checkpoint_final.pt}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs "${OUT_DIR}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${GEN_DATA}" == "1" ]]; then
  echo "=== [maze corruption] regenerating the maze test split ==="
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" python scripts/gen_amaze.py test maze
fi
[[ -s "${DATA}" ]] || { echo "missing ${DATA} — run with GEN_DATA=1" >&2; exit 1; }
for ck in "${TRM_CKPT}" "${PAINTER_CKPT}" "${DIT_CKPT}"; do
  [[ -s "${ck}" ]] || { echo "missing checkpoint ${ck}" >&2; exit 1; }
done
python - "${DATA}" "${NUM_SAMPLES}" <<'PY'
import sys, pandas as pd
path, want = sys.argv[1], int(sys.argv[2])
n = len(pd.read_parquet(path, columns=["id"]))
print(f"[maze corruption] {path}: {n} puzzles, probing {min(n, want)}")
if n < want:
    print(f"[maze corruption] WARNING: only {n} puzzles available (asked for {want})")
PY

COMMON=( +probe.t_starts="${T_STARTS}" +probe.batch_size="${BATCH}"
         +probe.num_samples="${NUM_SAMPLES}" +probe.data_parquet="${DATA}"
         +probe.num_inference_steps="${STEPS}" +probe.dump_n="${DUMP_N}" )

echo "=== [maze corruption] TRM  (t_starts=${T_STARTS} n=${NUM_SAMPLES} steps=${STEPS}) ==="
srun python experiments/maze_corruption_recovery_probe.py \
  experiment=amaze_thinker_v2_controlnet \
  painter.checkpoint="${PAINTER_CKPT}" \
  +checkpoint="${TRM_CKPT}" \
  +probe.model_name=trm "${COMMON[@]}" \
  +probe.out="${OUT_DIR}/trm.json"

echo "=== [maze corruption] DiT  (t_starts=${T_STARTS} n=${NUM_SAMPLES} steps=${STEPS}) ==="
srun python experiments/maze_corruption_recovery_probe.py \
  experiment=amaze_dit_maze \
  +checkpoint="${DIT_CKPT}" \
  +probe.model_name=dit "${COMMON[@]}" \
  +probe.out="${OUT_DIR}/dit.json"

echo "=== [maze corruption] report ==="
python experiments/report_maze_corruption.py \
  --runs trm="${OUT_DIR}/trm.json" dit="${OUT_DIR}/dit.json" \
  --out-dir "${OUT_DIR}/report"

echo "DONE -> ${OUT_DIR}/{trm,dit}.json and ${OUT_DIR}/report/"
