#!/bin/bash -l
#SBATCH --job-name=amaze_maze_painter
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

# Maze PAINTER driver (unconditional UNet painter). (SHAPE, SIZE) picks the version:
#   square  8    -> Painter V1
#   hexagon 8    -> Painter V2
#   all     all  -> Painter V3  (all shapes x all sizes)
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_maze_painter.sh <square|hexagon|triangle|circle|all> <N|all> [WANDB_PROJECT] [RUN_NAME]
# Env: PAINTER_STEPS (default 40000). NOTE: maze gen needs the node generator
# (run once on a login node: cd third_party/amaze/mazes-generator && npm install).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

SHAPE="${1:?usage: sbatch train_maze_painter.sh <shape|all> <N|all> [WANDB_PROJECT] [RUN_NAME]}"
SIZE="${2:?usage: sbatch train_maze_painter.sh <shape|all> <N|all> [WANDB_PROJECT] [RUN_NAME]}"
PAINTER_STEPS=${PAINTER_STEPS:-40000}
PAINTER_MAX_SECONDS=${PAINTER_MAX_SECONDS:-0}
WANDB_PROJECT="${WANDB_PROJECT:-${3:-amaze}}"

if [[ "${SHAPE}" == "all" ]]; then
  DATA_SUB="train_maze/all_train_size144"; TAG="all"
elif [[ "${SIZE}" == "all" ]]; then
  DATA_SUB="train_maze/${SHAPE}/all_${SHAPE}_train_size144"; TAG="${SHAPE}_all"
else
  DATA_SUB="train_maze/${SHAPE}/n${SIZE}_${SHAPE}_train_size144"; TAG="${SHAPE}_n${SIZE}"
fi
RUN_NAME="${RUN_NAME:-${4:-maze_painter_${TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

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

# ── Data (idempotent) ────────────────────────────────────────────────────────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train maze --shape "${SHAPE}" --size "${SIZE}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/${DATA_SUB}"

# Painter is UNCONDITIONAL -> the amaze callback auto-skips conditional metrics.
srun python train_trm.py experiment=amaze_unet_painter \
  data.amaze_root="${DATA_DIR}" \
  eval_callbacks=amaze \
  train.num_steps=${PAINTER_STEPS} \
  train.max_seconds=${PAINTER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Maze painter (${SHAPE} ${SIZE}) complete → runs/${RUN_NAME}/checkpoint_final.pt"
