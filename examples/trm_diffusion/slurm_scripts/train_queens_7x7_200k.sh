#!/bin/bash -l
#SBATCH --job-name=amaze_train_queens_200k
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

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"
TASK="queens"
N=7
# queens n=7 -> 7x7 board -> GRID=7, SEQ_LEN=49; CELL_SIZE=20 gives 144//20=7.
CELL_SIZE=20
SEQ_LEN=49
GRID=7
WANDB_PROJECT="${WANDB_PROJECT:-${1:-amaze}}"
RUN_NAME="${RUN_NAME:-${2:-train_queens_n${N}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs

export PYTHONUNBUFFERED=1

AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train "${TASK}" n=${N}
DATA_DIR="${PROJECT_ROOT}/data/amaze/train_queens_n${N}"

# ── Stage 1: standalone painter ──────────────────────────────────────────────
srun python train_trm.py experiment=amaze_unet_painter \
  data.amaze_root="${DATA_DIR}" \
  eval_callbacks=amaze_queens \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}_painter"

# ── Stage 2: thinker (TRM) with frozen painter ───────────────────────────────
srun python train_trm.py experiment=amaze_thinker_v1_controlnet \
  data.amaze_root="${DATA_DIR}" \
  painter.checkpoint="runs/${RUN_NAME}_painter/checkpoint_final.pt" \
  data.cell_size=${CELL_SIZE} \
  thinker.seq_len=${SEQ_LEN} \
  translator.grid=${GRID} \
  eval_callbacks=amaze_queens \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}_thinker"

echo "Train queens n=${N} — painter + thinker complete."

# ── Stage 3: paper metrics → logged into the SAME wandb run as the thinker
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_thinker_v1_controlnet \
    painter.checkpoint="runs/${RUN_NAME}_painter/checkpoint_final.pt" \
    +checkpoint="runs/${RUN_NAME}_thinker/checkpoint_final.pt" \
    +task="${TASK}" \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    data.cell_size=${CELL_SIZE} \
    thinker.seq_len=${SEQ_LEN} \
    translator.grid=${GRID} \
    || echo "WARN: metrics eval failed — training checkpoints are safe in runs/${RUN_NAME}_thinker."
  echo "Metrics eval (${TASK}) done — logged into wandb run ${RUN_NAME}_thinker."
fi
