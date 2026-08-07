#!/bin/bash -l
#SBATCH --job-name=amaze_queens_thinker
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

# Queens THINKER driver (TRM + frozen painter, then Stage-3 metrics n=4..10).
# SIZE picks the reasoning grid / data; PAINTER_CKPT is the frozen painter to
# steer (the thinker adapts to THAT painter, so the combo is defined here):
#   7   -> grid 7  (cell20/seq49), 7x7 data.  Painter V1 -> combo PT V1+V1;
#                                              Painter V2 -> combo PT V2+V1.
#   all -> grid 12 (cell12/seq144), mixed data. Painter V2 -> combo PT V2+V2.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_queens_thinker.sh <7|all> <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]
# Env: THINKER_STEPS (default 40000), SAMPLES (Pass@K, default 5), RUN_METRICS (default 1).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

SIZE="${1:?usage: sbatch train_queens_thinker.sh <7|all> <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]}"
PAINTER_CKPT="${2:?need a frozen painter checkpoint as arg 2}"
THINKER_STEPS=${THINKER_STEPS:-40000}
THINKER_MAX_SECONDS=${THINKER_MAX_SECONDS:-0}
WANDB_PROJECT="${WANDB_PROJECT:-${3:-amaze}}"

case "${SIZE}" in
  7)   EXP=amaze_thinker_v1_controlnet; CELL=20; SEQ=49;  GRID=7;  DATA_SUB="train_queens/n7_train_size144" ;;
  all) EXP=amaze_thinker_v2_controlnet; CELL=12; SEQ=144; GRID=12; DATA_SUB="train_queens/all_train_size144" ;;
  *)   echo "SIZE must be 7 (grid 7) or all (grid 12), got '${SIZE}'" >&2; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-${4:-queens_thinker_${SIZE}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

if [[ ! -f "${PAINTER_CKPT}" ]]; then
  echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2
  echo "Train one first: sbatch slurm_scripts/train_queens_painter.sh <7|all>" >&2
  exit 1
fi

# ── Data (idempotent) ────────────────────────────────────────────────────────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train queens --size "${SIZE}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/${DATA_SUB}"

# ── Stage 2: thinker with the frozen painter ─────────────────────────────────
srun python train_trm.py experiment=${EXP} \
  data.amaze_root="${DATA_DIR}" \
  painter.checkpoint="${PAINTER_CKPT}" \
  data.cell_size=${CELL} thinker.seq_len=${SEQ} translator.grid=${GRID} \
  eval_callbacks=amaze_queens \
  train.num_steps=${THINKER_STEPS} \
  train.max_seconds=${THINKER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Queens thinker (${SIZE}) complete → runs/${RUN_NAME}/checkpoint_final.pt"

# ── Stage 3: paper metrics (n=4..10), logged into the same wandb run ─────────
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  srun python experiments/sample_amaze_metrics.py \
    experiment=${EXP} \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task=queens \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    data.cell_size=${CELL} thinker.seq_len=${SEQ} translator.grid=${GRID} \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoints are safe in runs/${RUN_NAME}."
  echo "Metrics (queens, n=4..10) done — logged into wandb run ${RUN_NAME}."
fi
