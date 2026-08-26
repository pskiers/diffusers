#!/bin/bash -l
#SBATCH --job-name=amaze_eval_ft
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

# Generate K solution images with an FT model (or the dummy backend) and score
# them with AmazeMetrics into the amaze_final wandb project.
#
# Usage: sbatch slurm_scripts/eval_ft.sh <maze|queens>
# Env:
#   BACKEND        dummy | bagel | janus   (default dummy)
#   CHECKPOINT     FT checkpoint dir/path  (required for bagel|janus)
#   RUN_NAME       wandb run name          (default ft_<backend>_<task>)
#   GEN_DIR        image output dir        (default runs/<run_name>/generated)
#   WANDB_PROJECT  (default amaze_final)
#   SAMPLES        images per puzzle       (default 5)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

TASK="${1:?usage: sbatch eval_ft.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

BACKEND="${BACKEND:-dummy}"
SAMPLES="${SAMPLES:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_${BACKEND}_${TASK}}"
GEN_DIR="${GEN_DIR:-runs/${RUN_NAME}/generated}"

GEN_ARGS=( "${TASK}" --backend "${BACKEND}" --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" )
if [[ "${BACKEND}" != "dummy" ]]; then
  : "${CHECKPOINT:?set CHECKPOINT to the FT model dir for backend ${BACKEND}}"
  GEN_ARGS+=( --checkpoint "${CHECKPOINT}" )
fi
if [[ "${BACKEND}" == "bagel" ]]; then
  : "${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to the base BAGEL-7B-MoT snapshot dir}"
  GEN_ARGS+=( --bagel-model-path "${BAGEL_MODEL_PATH}" )
fi

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

export PYTHONUNBUFFERED=1

python experiments/generate_amaze_ft.py "${GEN_ARGS[@]}"

python experiments/score_amaze_images.py "${TASK}" \
  --gen-dir "${GEN_DIR}" \
  --samples-per-puzzle "${SAMPLES}" \
  --wandb-project "${WANDB_PROJECT}" \
  --run-name "${RUN_NAME}"
