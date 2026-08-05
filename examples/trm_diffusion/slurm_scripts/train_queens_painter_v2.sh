#!/bin/bash -l
#SBATCH --job-name=amaze_queens_painter_v2
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Painter V2 — unconditional UNet painter trained on the MULTI-SCALE queens mix
# (n=4..10, richness-capped by gen_amaze.py). The painter is a plain 144x144 UNet
# independent of the reasoning grid, so this one checkpoint is a drop-in for both
# Thinker V1 (combo 2) and Thinker V2 (combo 3).
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_queens_painter_v2.sh [WANDB_PROJECT] [RUN_NAME]
# Tip: pre-generate data once with
#   sbatch slurm_scripts/gen_amaze.sh train queens --size all
# to avoid spending GPU time on CPU-bound generation (the call below is idempotent).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

PAINTER_STEPS=${PAINTER_STEPS:-40000}
PAINTER_MAX_SECONDS=${PAINTER_MAX_SECONDS:-0}   # 0 = no wall-clock limit
WANDB_PROJECT="${WANDB_PROJECT:-${1:-amaze}}"
RUN_NAME="${RUN_NAME:-${2:-queens_painter_v2${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs

export PYTHONUNBUFFERED=1

# ── Multi-scale queens data (idempotent: skips whatever already exists) ───────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train queens --size all
DATA_DIR="${PROJECT_ROOT}/data/amaze/train_queens/all_train_size144"

# ── Stage 1: standalone unconditional painter on the mixed set ───────────────
# The painter is UNCONDITIONAL, so eval must NOT score conditional queens metrics
# (Pass@1 / coverage / violation are meaningless without a puzzle->solution map).
# eval_callbacks=image_gen only logs sample images (ImageGenEvalCallback returns
# no metrics). (The amaze_queens callback would also auto-skip scoring because the
# painter's condition_keys are empty — image_gen just makes that explicit.)
srun python train_trm.py experiment=amaze_unet_painter \
  data.amaze_root="${DATA_DIR}" \
  eval_callbacks=image_gen \
  train.num_steps=${PAINTER_STEPS} \
  train.max_seconds=${PAINTER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Painter V2 complete → runs/${RUN_NAME}/checkpoint_final.pt"
