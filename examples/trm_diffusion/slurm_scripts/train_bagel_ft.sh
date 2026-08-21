#!/bin/bash -l
#SBATCH --job-name=amaze_bagel_ft
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

# Full fine-tune BAGEL-7B-MoT on the amaze FT dataset, reusing the vendored AMAZE
# trainer third_party/amaze/sft/bagel/{run_sft.sh,sft.py} with the paper's flags.
#
# Bagel's sft.py imports `data.* / modeling.* / train.*` from the base Bagel repo,
# and the AMAZE maze data files (maze_dataset.py, maze_packed_dataset.py) must be
# overlaid into that repo's data/. This script assembles that (overlay into the
# gitignored base repo) and launches torchrun.
#
# FIRST-RUN NOTE: the base-repo assembly / PYTHONPATH may need one debug pass on
# the cluster (the AMAZE README hides the overlay). Do a short smoke run first
# (e.g. TOTAL_STEPS=20) before the full budget.
#
# Usage: sbatch slurm_scripts/train_bagel_ft.sh <maze|queens>
# Env: BAGEL_MODEL_PATH (local BAGEL-7B-MoT snapshot, REQUIRED), TOTAL_STEPS (5000),
#      LR (1e-5), WANDB_PROJECT (amaze_final), RUN_NAME.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

TASK="${1:?usage: sbatch train_bagel_ft.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
BAGEL_SFT="${AMAZE_DIR}/sft/bagel"
BAGEL_BASE="${BAGEL_SFT}/Bagel"
DATA_DIR="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

TOTAL_STEPS="${TOTAL_STEPS:-5000}"
LR="${LR:-1e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_bagel_${TASK}}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to a local BAGEL-7B-MoT snapshot (huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT)}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

[[ -d "${BAGEL_BASE}" ]] || { echo "ERROR: ${BAGEL_BASE} missing. Run on a login node: bash ${AMAZE_DIR}/setup_ft_code.sh (without SKIP_BASE)." >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/export_amaze_for_ft.py ${TASK}." >&2; exit 1; }

# ── Assemble: overlay AMAZE sft.py + maze data files into the base Bagel repo ──
cp "${BAGEL_SFT}/sft.py" "${BAGEL_BASE}/sft.py"
for f in maze_dataset.py maze_packed_dataset.py; do
  [[ -f "${AMAZE_DIR}/infer/bagel/data/${f}" ]] && cp "${AMAZE_DIR}/infer/bagel/data/${f}" "${BAGEL_BASE}/data/${f}"
done

cd "${BAGEL_BASE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')"

srun torchrun --standalone --nproc_per_node="${NUM_GPUS}" sft.py \
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
  --sharding_strategy HYBRID_SHARD --backward_prefetch BACKWARD_PRE \
  --num_replicate 1 --num_shard "${NUM_GPUS}" --cpu_offload true --use_lora false

echo "Bagel FT (${TASK}) complete -> runs/${RUN_NAME}"
