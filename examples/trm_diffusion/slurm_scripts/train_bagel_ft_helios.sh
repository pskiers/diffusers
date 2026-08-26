#!/bin/bash -l
#SBATCH --job-name=amaze_bagel_ft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Helios (GH200 / aarch64) variant of train_bagel_ft.sh. A single GH200 (120GB HBM +
# large Grace host RAM) fits the 14.6B model with FSDP NO_SHARD + cpu_offload (optimizer
# states offloaded to host). Only account/partition/modules/venv/sharding differ from
# the Athena (4x A100) script.
#
# Usage: sbatch slurm_scripts/train_bagel_ft_helios.sh <maze|queens>
# Env: BAGEL_MODEL_PATH (local BAGEL-7B-MoT snapshot, REQUIRED), TOTAL_STEPS (5000),
#      LR (1e-5), WANDB_PROJECT (amaze_final), RUN_NAME, VENV (default $SCRATCH/trm_helios_venv).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_bagel_ft_helios.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
BAGEL_SFT="${AMAZE_DIR}/sft/bagel"
BAGEL_BASE="${BAGEL_SFT}/Bagel"
DATA_DIR="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

TOTAL_STEPS="${TOTAL_STEPS:-5000}"
LR="${LR:-1e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_bagel_${TASK}}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to a local BAGEL-7B-MoT snapshot}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
# newer libstdc++ for compiled extensions (insurance; harmless if unused)
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${BAGEL_BASE}" ]] || { echo "ERROR: ${BAGEL_BASE} missing. Run: bash ${AMAZE_DIR}/setup_ft_code.sh" >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/export_amaze_for_ft.py ${TASK}." >&2; exit 1; }

# ── Assemble: overlay AMAZE sft.py + maze data files into the base Bagel repo ──
cp "${BAGEL_SFT}/sft.py" "${BAGEL_BASE}/sft.py"
for f in maze_dataset.py maze_packed_dataset.py; do
  [[ -f "${AMAZE_DIR}/infer/bagel/data/${f}" ]] && cp "${AMAZE_DIR}/infer/bagel/data/${f}" "${BAGEL_BASE}/data/${f}"
done

cd "${BAGEL_BASE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Single GH200: NO_SHARD keeps the full model on one GPU; cpu_offload puts the
# (fp32) optimizer states on host RAM so 14.6B fits in 120GB HBM.
srun torchrun --standalone --nproc_per_node=1 sft.py \
  --maze_dataset_path "${DATA_DIR}" \
  --model_path "${BAGEL_MODEL_PATH}" \
  --resume_from "${BAGEL_MODEL_PATH}" \
  --results_dir "${PROJECT_ROOT}/runs/${RUN_NAME}" \
  --checkpoint_dir "${PROJECT_ROOT}/runs/${RUN_NAME}/checkpoints" \
  --wandb_project "${WANDB_PROJECT}" --wandb_name "${RUN_NAME}" --wandb_offline false \
  --visual_gen true --visual_und true \
  --finetune_from_ema true --resume_model_only true --finetune_from_hf true --auto_resume true \
  --total_steps "${TOTAL_STEPS}" --save_every 100 --log_every 1 --eval_every 50 --eval_samples 8 \
  --warmup_steps 10 --lr "${LR}" --lr_scheduler cosine --min_lr 1e-7 \
  --expected_num_tokens 5000 --max_num_tokens 5000 --max_num_tokens_per_sample 5000 \
  --gradient_accumulation_steps 8 \
  --freeze_llm false --freeze_vit true --freeze_vae true \
  --num_workers 1 --prefetch_factor 1 --max_buffer_size 1 --prefer_buffer_before 10000 \
  --max_grad_norm 1.0 --beta1 0.9 --beta2 0.95 --eps 1e-15 \
  --ce_weight 0.000001 --mse_weight 1.0 --timestep_shift 1.0 \
  --max_latent_size 64 --latent_patch_size 2 --vit_patch_size 14 --vit_max_num_patch_per_side 70 \
  --text_cond_dropout_prob 0 --vae_cond_dropout_prob 0 --vit_cond_dropout_prob 0 \
  --connector_act gelu_pytorch_tanh --interpolate_pos false --vit_rope false \
  --llm_qk_norm true --tie_word_embeddings false --layer_module Qwen2MoTDecoderLayer \
  --copy_init_moe true --use_flex false --global_seed 4396 \
  --sharding_strategy NO_SHARD --backward_prefetch BACKWARD_PRE \
  --num_replicate 1 --num_shard 1 --cpu_offload true --use_lora false

echo "Bagel FT (${TASK}) complete -> runs/${RUN_NAME}"
