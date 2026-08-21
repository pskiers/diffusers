#!/bin/bash -l
#SBATCH --job-name=amaze_maze_dit
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Standalone (no-TRM) diffusion baseline for MAZE:
# ALL shapes (square, hexagon, triangle, circle) x ALL sizes. Pixel-space
# concat-conditioned DiT (Palette/SR3 style), CFG off.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_maze_dit.sh [WANDB_PROJECT] [RUN_NAME]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

DIT_STEPS=${DIT_STEPS:-80000}
DIT_MAX_SECONDS=${DIT_MAX_SECONDS:-0}   # 0 = no wall-clock limit
WANDB_PROJECT="${WANDB_PROJECT:-${1:-amaze}}"
RUN_NAME="${RUN_NAME:-${2:-maze_dit${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs

export PYTHONUNBUFFERED=1

if [[ ! -d third_party/amaze/mazes-generator/node_modules ]]; then
  echo "ERROR: third_party/amaze/mazes-generator/node_modules is missing (maze gen needs it)." >&2
  echo "Run ONCE on a login node: module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
  exit 1
fi

# ── All shapes × all sizes (idempotent: skips whatever already exists) ────────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train maze --shape all --size all
DATA_DIR="${PROJECT_ROOT}/data/amaze/train_maze/all_train_size144"

# ── Train the standalone DiT baseline (conditional, CFG off) ─────────────────
srun python train_trm.py experiment=amaze_dit_maze \
  data.amaze_root="${DATA_DIR}" \
  train.num_steps=${DIT_STEPS} \
  train.max_seconds=${DIT_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Maze DiT complete → runs/${RUN_NAME}/checkpoint_final.pt"

# ── Stage 3: final sampling → paper metrics (all 4 geometries × scales) ──────
# The DiT trained on all shapes, so every geometry is in-distribution here.
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py test maze
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_dit_maze \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task=maze \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoint is safe in runs/${RUN_NAME}."
  echo "Metrics (maze, 4 geometries × scales) done — logged into wandb run ${RUN_NAME}."
fi
