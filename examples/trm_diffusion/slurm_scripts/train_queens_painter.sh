#!/bin/bash -l
#SBATCH --job-name=amaze_queens_painter
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

# Queens PAINTER driver (unconditional UNet painter). SIZE picks the version:
#   7    -> Painter V1  (7x7 only)
#   all  -> Painter V2  (multi-scale mix n=4..10, richness-capped by gen_amaze.py)
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_queens_painter.sh <7|all> [WANDB_PROJECT] [RUN_NAME]
# Env: PAINTER_STEPS (default 40000), PAINTER_MAX_SECONDS (default 0 = no limit).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

SIZE="${1:?usage: sbatch train_queens_painter.sh <7|all> [WANDB_PROJECT] [RUN_NAME]}"
PAINTER_STEPS=${PAINTER_STEPS:-40000}
PAINTER_MAX_SECONDS=${PAINTER_MAX_SECONDS:-0}
WANDB_PROJECT="${WANDB_PROJECT:-${2:-amaze}}"

case "${SIZE}" in
  7)   TAG="n7";  DATA_SUB="train_queens/n7_train_size144" ;;
  all) TAG="all"; DATA_SUB="train_queens/all_train_size144" ;;
  *)   echo "SIZE must be 7 (V1) or all (V2), got '${SIZE}'" >&2; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-${3:-queens_painter_${TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

# ── Data (idempotent) ────────────────────────────────────────────────────────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train queens --size "${SIZE}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/${DATA_SUB}"

# Painter is UNCONDITIONAL -> the amaze_queens callback auto-skips conditional
# metrics (only logs a few sample images).
srun python train_trm.py experiment=amaze_unet_painter \
  data.amaze_root="${DATA_DIR}" \
  eval_callbacks=amaze_queens \
  train.num_steps=${PAINTER_STEPS} \
  train.max_seconds=${PAINTER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Queens painter (${SIZE}) complete → runs/${RUN_NAME}/checkpoint_final.pt"
