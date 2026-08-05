#!/bin/bash -l
#SBATCH --job-name=amaze_queens_thinker_v2
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

# Thinker V2 — TRM on a 12x12 = 144-token grid, trained on the MULTI-SCALE queens
# mix (n=4..10). Requires a frozen painter checkpoint:
#   * Painter V2 (multi-scale) -> combo 3 (Painter V2 + Thinker V2)
#   * Painter V1 (7x7 only)    -> runnable, but the painter cannot render n != 7
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_queens_thinker_v2.sh <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]
#   e.g. sbatch slurm_scripts/train_queens_thinker_v2.sh runs/queens_painter_v2/checkpoint_final.pt

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

PAINTER_CKPT="${PAINTER_CKPT:-${1:?usage: sbatch train_queens_thinker_v2.sh <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]}}"
THINKER_STEPS=${THINKER_STEPS:-40000}
THINKER_MAX_SECONDS=${THINKER_MAX_SECONDS:-0}   # 0 = no wall-clock limit
WANDB_PROJECT="${WANDB_PROJECT:-${2:-amaze}}"
RUN_NAME="${RUN_NAME:-${3:-queens_thinker_v2${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs

export PYTHONUNBUFFERED=1

if [[ ! -f "${PAINTER_CKPT}" ]]; then
  echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2
  echo "Train one first: sbatch slurm_scripts/train_queens_painter_v2.sh" >&2
  exit 1
fi

# ── Multi-scale queens data (idempotent: skips whatever already exists) ───────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train queens --size all
DATA_DIR="${PROJECT_ROOT}/data/amaze/train_queens/all_train_size144"

# ── Stage 2: thinker V2 (12x12 grid) steering the frozen painter ─────────────
# cell_size=12 / seq_len=144 / translator.grid=12 are baked into the experiment.
srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
  data.amaze_root="${DATA_DIR}" \
  painter.checkpoint="${PAINTER_CKPT}" \
  eval_callbacks=amaze_queens \
  train.num_steps=${THINKER_STEPS} \
  train.max_seconds=${THINKER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Thinker V2 complete → runs/${RUN_NAME}/checkpoint_final.pt"

# ── Stage 3: multi-scale paper metrics (n=4..10), logged into the same wandb run
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_thinker_v2_controlnet \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task=queens \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoints are safe in runs/${RUN_NAME}."
  echo "Metrics (queens, n=4..10) done — logged into wandb run ${RUN_NAME}."
fi
