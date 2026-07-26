#!/bin/bash -l
# Task: maze, 8x8 SQUARE. Both stages (painter → thinker) for 30k steps each.
#
# Submit from the trm_diffusion project root:
#     sbatch slurm_scripts/maze_square_8x8_30k.sh [WANDB_PROJECT] [RUN_NAME]
#
#SBATCH --job-name=amaze_maze_sq_30k
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

set -euo pipefail

# ── User config ──────────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # dir containing train_trm.py
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/ablation_uncond"
DATASET="maze_square_8x8"
STEPS=30000
# maze 8x8 -> 8x8 cells -> GRID=8, SEQ_LEN=64; CELL_SIZE=18 gives 144//18=8.
# SEQ_LEN == GRID*GRID and GRID == 144 // CELL_SIZE.
CELL_SIZE=18
SEQ_LEN=64
GRID=8
WANDB_PROJECT="${WANDB_PROJECT:-${1:-amaze}}"
RUN_NAME="${RUN_NAME:-${2:-${DATASET}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs

export PYTHONUNBUFFERED=1

# pre-run bash scripts/generate_amaze_datasets.sh "${DATASET}"
OUT_ROOT="${PROJECT_ROOT}/data/amaze" PYTHON="${VENV}/bin/python" \
  bash scripts/generate_amaze_datasets.sh "${DATASET}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/${DATASET}"

# ── Stage 1: standalone painter ──────────────────────────────────────────────
srun python train_trm.py experiment=amaze_unet_painter \
  data.amaze_root="${DATA_DIR}" \
  train.num_steps=${STEPS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}_painter"

# ── Stage 2: thinker (TRM) with frozen painter ───────────────────────────────
srun python train_trm.py experiment=amaze_thinker_v1_controlnet \
  data.amaze_root="${DATA_DIR}" \
  painter.checkpoint="runs/${RUN_NAME}_painter/checkpoint_final.pt" \
  data.cell_size=${CELL_SIZE} \
  thinker.seq_len=${SEQ_LEN} \
  translator.grid=${GRID} \
  train.num_steps=${STEPS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}_thinker"

echo "Experiment maze square 8x8, ${STEPS} steps complete."
