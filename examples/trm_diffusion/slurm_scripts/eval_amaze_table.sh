#!/bin/bash -l
#SBATCH --job-name=amaze_eval_table
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

usage() {
  echo "usage: sbatch slurm_scripts/eval_amaze_table.sh <maze|queens> <painter.pt> <thinker.pt> [CELL_SIZE SEQ_LEN GRID]" >&2
  exit 1
}

# sbatch slurm_scripts/eval_amaze_table.sh queens runs/queens_7x7_painter/checkpoint_final.pt runs/queens_7x7_thinker/checkpoint_final.pt
# sbatch slurm_scripts/eval_amaze_table.sh maze runs/maze_hex_8x8_painter/checkpoint_final.pt runs/maze_hex_8x8_thinker/checkpoint_final.pt
# sbatch slurm_scripts/eval_amaze_table.sh maze runs/maze_square_8x8_painter/checkpoint_final.pt runs/maze_square_8x8_thinker/checkpoint_final.pt


TASK="${TASK:-${1:-}}"
PAINTER_CKPT="${PAINTER_CKPT:-${2:-}}"
THINKER_CKPT="${THINKER_CKPT:-${3:-}}"
[[ -z "${TASK}" || -z "${PAINTER_CKPT}" || -z "${THINKER_CKPT}" ]] && usage
[[ "${TASK}" != "maze" && "${TASK}" != "queens" ]] && { echo "TASK must be maze|queens" >&2; usage; }

# Thinker grid — must match training. Task-specific defaults.
if [[ "${TASK}" == "queens" ]]; then
  CELL_SIZE="${CELL_SIZE:-${4:-20}}"; SEQ_LEN="${SEQ_LEN:-${5:-49}}"; GRID="${GRID:-${6:-7}}"
else
  CELL_SIZE="${CELL_SIZE:-${4:-18}}"; SEQ_LEN="${SEQ_LEN:-${5:-64}}"; GRID="${GRID:-${6:-8}}"
fi
SAMPLES="${SAMPLES:-5}"     # Pass@5

# wandb: eval metrics are logged into the SAME run as training. The run id is read. Set WANDB_PROJECT='' to disable wandb logging (the JSON output is always written).
WANDB_PROJECT="${WANDB_PROJECT:-amaze}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

export PYTHONUNBUFFERED=1

# Ensure the canonical test set exists (idempotent; maze needs Node.js).
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py test "${TASK}"

srun python experiments/sample_amaze_metrics.py \
  experiment=amaze_thinker_v1_controlnet \
  painter.checkpoint="${PAINTER_CKPT}" \
  +checkpoint="${THINKER_CKPT}" \
  +task="${TASK}" \
  +data_root="${PROJECT_ROOT}/data/amaze" \
  +samples_per_puzzle="${SAMPLES}" \
  run.wandb_project="${WANDB_PROJECT}" \
  data.cell_size=${CELL_SIZE} \
  thinker.seq_len=${SEQ_LEN} \
  translator.grid=${GRID}

echo "Metrics eval (${TASK}) complete — results saved next to ${THINKER_CKPT}."
