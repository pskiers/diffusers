#!/bin/bash -l
#SBATCH --job-name=janus_sft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage: sbatch slurm_scripts/train_amaze/train_janus.sh <maze|queens|maze3_800|queens_n4_800>

set -euo pipefail

TASK="${1:?usage: sbatch slurm_scripts/train_amaze/train_janus.sh <maze|queens|maze3_800|queens_n4_800>}"

: "${SCRATCH:?SCRATCH is not set - run this under sbatch on Helios, or export SCRATCH yourself}"

PROJECT_ROOT="/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion"
VENV="${SCRATCH}/trm_helios_venv"
EAR_AMAZE_ROOT="${PROJECT_ROOT}/third_party/ear-amaze"
SFT_DIR="${EAR_AMAZE_ROOT}/sft/janus"
MAZE_DATASET_PATH="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

MODEL_PATH="deepseek-ai/Janus-Pro-7B"

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/slurm_outputs"

# sft.py imports `janus` and `data.maze_dataset`.
export PYTHONPATH="${SFT_DIR}/Janus:${EAR_AMAZE_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${VENV}" ]] || { echo "ERROR: venv not found: ${VENV}" >&2; exit 1; }
source "${VENV}/bin/activate"

[[ -f "${MAZE_DATASET_PATH}/maze_dataset_train.parquet" ]] || {
    echo "ERROR: ${MAZE_DATASET_PATH}/maze_dataset_train.parquet missing." >&2
    echo "       Run: python ${PROJECT_ROOT}/scripts/gen_amaze.py ft ${TASK}" >&2; exit 1; }

python -c "import torch, wandb, transformers
from janus.models import VLChatProcessor
from data.maze_dataset import MazeDataset" || {
    echo "ERROR: imports failed. If it is attrdict: pip install attrdict3" >&2
    echo "       (attrdict 2.0.1 cannot import on Python 3.11)" >&2; exit 1; }

cd "${SFT_DIR}"

OUTPUT_DIR="${SFT_DIR}/outputs/${TASK}"
LOG_DIR="${SFT_DIR}/train_logs"
EXPERIMENT_NAME="janus_train_${TASK}"
RUN_NAME="${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# The authors' recipe, from the example at the top of sft.py. argparse's defaults
# are much weaker (n_epochs 8, lr 5e-6, accum 16) -- on 800 samples that is 400
# optimizer updates against their 20,000. Lower N_EPOCHS for the big datasets.
N_EPOCHS="${N_EPOCHS:-200}"
LR="${LR:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
MAX_CKPTS="${MAX_CKPTS:-10}"

echo "============================================="
echo "Janus SFT Training"
echo "Task: $TASK"
echo "Data: $MAZE_DATASET_PATH"
echo "Output: $OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME"
echo "Epochs: $N_EPOCHS | lr: $LR | grad_accum: $GRAD_ACCUM"
echo "============================================="

# sft.py calls dist.get_world_size() without an is_initialized() guard, but
# accelerate creates no process group for a single process. These make it
# initialise a 1-rank group instead of raising.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    sft.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${MAZE_DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "${RUN_NAME}" \
    --log_dir "${LOG_DIR}" \
    --n_epochs "${N_EPOCHS}" \
    --learning_rate "${LR}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --min_lr_ratio "${MIN_LR_RATIO}" \
    --max_ckpts "${MAX_CKPTS}"

echo "Finished -> ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
echo "Score it with: sbatch slurm_scripts/sample_amaze/eval_janus.sh ${TASK} <shape-if-maze>"
