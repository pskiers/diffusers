#!/bin/bash -l
#SBATCH --job-name=amaze_janus_ft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Helios (GH200 / aarch64) variant of train_janus_ft.sh. A single GH200 (120GB HBM +
# large Grace host RAM) full-fine-tunes Janus-Pro-7B via the vendored AMAZE trainer
# third_party/amaze/sft/janus/sft.py (image->image VQ-token injection). Only
# account/partition/modules/venv/GPU-count differ from the Athena (4x A100) script.
#
# Usage (from trm_diffusion):  sbatch slurm_scripts/train_janus_ft_helios.sh <maze|queens>
# Env: JANUS_MODEL_PATH (local Janus-Pro-7B snapshot, REQUIRED), N_EPOCHS (8),
#      LR (5e-6), GRAD_ACCUM (16), WANDB_PROJECT (amaze_final), RUN_NAME,
#      VENV (default $SCRATCH/trm_helios_venv).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_janus_ft_helios.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
JANUS_SFT="${AMAZE_DIR}/sft/janus"
JANUS_BASE="${JANUS_SFT}/Janus"
DATA_DIR="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

# Reference AMAZE config (sft.py): 8 epochs, lr 5e-6, grad_accum 16. Effective batch =
# 1 gpu x train_bsz(1) x grad_accum(16) = 16 samples/opt-step -> 1 epoch ~= 1875 steps,
# 8 epochs ~= 15000. The trainer checkpoints once per epoch (checkpoint-<epoch>-<step>/
# tfmr) with gradient checkpointing on; 8 epochs overruns one 48 h job, so resubmit with
# --resume_from_checkpoint runs/<RUN_NAME>/checkpoint-<epoch>-<step> to finish.
N_EPOCHS="${N_EPOCHS:-8}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_janus_${TASK}}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:?set JANUS_MODEL_PATH to a local Janus-Pro-7B snapshot (huggingface-cli download deepseek-ai/Janus-Pro-7B)}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
# newer libstdc++ for compiled extensions (insurance; harmless if unused)
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${JANUS_BASE}" ]] || { echo "ERROR: ${JANUS_BASE} missing. Run on a login node: bash ${AMAZE_DIR}/setup_ft_code.sh (without SKIP_BASE)." >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/export_amaze_for_ft.py ${TASK}." >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}/${JANUS_BASE}:${PROJECT_ROOT}/${AMAZE_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Single GH200 -> one training process.
RESUME_ARGS=()
[[ -n "${RESUME_FROM:-}" ]] && RESUME_ARGS=( --resume_from_checkpoint "${RESUME_FROM}" )

srun accelerate launch --num_processes 1 --mixed_precision bf16 \
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
  --seed 42 \
  "${RESUME_ARGS[@]}"

echo "Janus FT (${TASK}) complete -> runs/${RUN_NAME}"
