#!/bin/bash -l
#SBATCH --job-name=janus_sft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage: sbatch slurm_scripts/train_janus.sh <maze|queens|maze3_800|queens_n4_800>

set -euo pipefail

TASK="${1:?Add subdirectory with data}"

PROJECT_ROOT="/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion"
VENV="${SCRATCH}/trm_helios_venv"
MAZE_DATASET_PATH="${PROJECT_ROOT}/data/amaze/ft/${TASK}"
SFT_DIR="${PROJECT_ROOT}/third_party/ear-amaze/sft/janus"

MODEL_PATH="deepseek-ai/Janus-Pro-7B"

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/slurm_outputs"

export PYTHONPATH="${SFT_DIR}/Janus:${PWD}:${PYTHONPATH:-}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"
cd "${SFT_DIR}"

OUTPUT_DIR="${SFT_DIR}/outputs/${TASK}"
LOG_DIR="${SFT_DIR}/train_logs"
EXPERIMENT_NAME="janus_train_${TASK}"
RUN_NAME="${TASK}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export OMP_NUM_THREADS=1

echo "============================================="
echo "Janus SFT Training"
echo "Task: $TASK"
echo "Data: $MAZE_DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Default sft.py arguments"
echo "============================================="

srun accelerate launch -m sft.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${MAZE_DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "${RUN_NAME}" \
    --log_dir "${LOG_DIR}"

echo "Finished"
