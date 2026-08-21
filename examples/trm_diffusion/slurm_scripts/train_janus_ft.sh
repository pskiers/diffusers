#!/bin/bash -l
#SBATCH --job-name=amaze_janus_ft
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Full fine-tune Janus-Pro-7B on the amaze FT dataset, reusing the vendored AMAZE
# trainer third_party/amaze/sft/janus/sft.py (image->image via VQ-token injection).
# Data comes from scripts/export_amaze_for_ft.py -> data/amaze/ft/<task>.
#
# Usage (from trm_diffusion):  sbatch slurm_scripts/train_janus_ft.sh <maze|queens>
# Env: JANUS_MODEL_PATH (local Janus-Pro-7B snapshot, REQUIRED), N_EPOCHS (8),
#      LR (5e-6), GRAD_ACCUM (16), WANDB_PROJECT (amaze_final), RUN_NAME.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

TASK="${1:?usage: sbatch train_janus_ft.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
JANUS_SFT="${AMAZE_DIR}/sft/janus"
JANUS_BASE="${JANUS_SFT}/Janus"
DATA_DIR="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

N_EPOCHS="${N_EPOCHS:-8}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_janus_${TASK}}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:?set JANUS_MODEL_PATH to a local Janus-Pro-7B snapshot (huggingface-cli download deepseek-ai/Janus-Pro-7B)}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

[[ -d "${JANUS_BASE}" ]] || { echo "ERROR: ${JANUS_BASE} missing. Run on a login node: bash ${AMAZE_DIR}/setup_ft_code.sh (without SKIP_BASE)." >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/export_amaze_for_ft.py ${TASK}." >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}/${JANUS_BASE}:${PROJECT_ROOT}/${AMAZE_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')"

srun accelerate launch --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
  "${JANUS_SFT}/sft.py" \
  --model_path "${JANUS_MODEL_PATH}" \
  --data_path "${DATA_DIR}" \
  --output_dir "${PROJECT_ROOT}/runs/${RUN_NAME}" \
  --experiment_name "${RUN_NAME}" \
  --run_name "${RUN_NAME}" \
  --train_bsz_per_gpu 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --n_epochs "${N_EPOCHS}" \
  --warmup_rates 0.05 \
  --min_lr_ratio 0.15 \
  --max_grad_norm 1.0 \
  --weight_decay 0.1 \
  --max_ckpts 10 \
  --log_dir "${PROJECT_ROOT}/runs/${RUN_NAME}/logs" \
  --seed 42

echo "Janus FT (${TASK}) complete -> runs/${RUN_NAME}"
