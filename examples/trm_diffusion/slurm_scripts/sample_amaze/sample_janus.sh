#!/bin/bash -l
#SBATCH --job-name=infer_janus
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage: sbatch slurm_scripts/infer_janus.sh <maze|queens|maze3_800|queens_n4_800> [checkpoint_path]
set -euo pipefail

TASK="${1:?Add subdirectory with data (maze|queens|maze3_800|queens_n4_800)}"
CHECKPOINT_OVERRIDE="${2:-}"

PROJECT_ROOT="/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion"
VENV="${SCRATCH}/trm_helios_venv"
SFT_DIR="${PROJECT_ROOT}/third_party/ear-amaze/sft/janus"
DATA_PATH="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${PROJECT_ROOT}/slurm_outputs"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"
cd "${SFT_DIR}"
export PYTHONPATH="${SFT_DIR}/Janus:${PWD}:${PYTHONPATH:-}"

OUTPUT_DIR="${SFT_DIR}/inference_results/${TASK}"
mkdir -p "${OUTPUT_DIR}"

# Auto-discover the most recently-modified checkpoint for this task
# (checkpoint dirs get rotated by max_ckpts, so newest-by-mtime == latest step)
if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
    CHECKPOINT_PATH="${CHECKPOINT_OVERRIDE}"
else
    CKPT_ROOT="${SFT_DIR}/outputs/${TASK}/janus_train_${TASK}"
    LATEST_CKPT_DIR=$(find "${CKPT_ROOT}" -maxdepth 3 -type d -name "checkpoint-*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-)
    if [ -z "${LATEST_CKPT_DIR}" ]; then
        echo "No checkpoint found under ${CKPT_ROOT}" >&2
        exit 1
    fi
    CHECKPOINT_PATH="${LATEST_CKPT_DIR}/tfmr"
fi

echo "============================================="
echo "Janus inference"
echo "Task: ${TASK}"
echo "Data: ${DATA_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================="

srun python infer_janus.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --split test \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 16 \
    --temperature 1.0 \
    --num_attempts 5

echo "Finished"
