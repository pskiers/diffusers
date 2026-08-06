#!/bin/bash -l
#SBATCH --job-name=amaze_queens_dit_v3
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

# Standalone (no-TRM) diffusion baseline for QUEENS — "V3".
# Pixel-space concat-conditioned DiT (Palette/SR3 style), CFG off, trained on the
# multi-scale queens mix (n=4..10), then scored across n=4..10.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_queens_dit_v3.sh [WANDB_PROJECT] [RUN_NAME]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

DIT_STEPS=${DIT_STEPS:-80000}
DIT_MAX_SECONDS=${DIT_MAX_SECONDS:-0}   # 0 = no wall-clock limit
WANDB_PROJECT="${WANDB_PROJECT:-${1:-amaze}}"
RUN_NAME="${RUN_NAME:-${2:-queens_dit_v3${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

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

# ── Train the standalone DiT baseline (conditional, CFG off) ─────────────────
srun python train_trm.py experiment=amaze_dit_queens_v3 \
  data.amaze_root="${DATA_DIR}" \
  train.num_steps=${DIT_STEPS} \
  train.max_seconds=${DIT_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Queens DiT V3 complete → runs/${RUN_NAME}/checkpoint_final.pt"

# ── Multi-scale paper metrics (n=4..10), logged into the same wandb run ──────
# The DiT is a standalone painter (mode painter_base), so the whole model is the
# checkpoint — no separate painter.checkpoint is needed.
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_dit_queens_v3 \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task=queens \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoint is safe in runs/${RUN_NAME}."
  echo "Metrics (queens, n=4..10) done — logged into wandb run ${RUN_NAME}."
fi
