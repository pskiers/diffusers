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

# Usage (from trm_diffusion):  sbatch slurm_scripts/train_janus_ft_helios.sh <maze|queens>
# Env: JANUS_MODEL_PATH (local Janus-Pro-7B snapshot, REQUIRED), N_EPOCHS (8),
#      LR (5e-6), GRAD_ACCUM (16), WANDB_PROJECT (amaze_final), WANDB_MODE (online;
#      set offline on no-internet nodes then `wandb sync`), VAL_EVERY_EPOCHS (1),
#      RUN_NAME, VENV (default $SCRATCH/trm_helios_venv).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_janus_ft_helios.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
JANUS_SFT="${AMAZE_DIR}/sft/janus"
JANUS_BASE="${JANUS_SFT}/Janus"
DATA_DIR="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

N_EPOCHS="${N_EPOCHS:-8}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
WANDB_MODE="${WANDB_MODE:-online}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-200}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-16}"
SAMPLE="${SAMPLE:-true}"
SAMPLES="${SAMPLES:-5}"
SELECT="${SELECT:-$([[ "${TASK}" == "queens" ]] && echo all || echo val)}"  # queens: score all epochs; maze: pick min-val
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
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/gen_amaze.py ft ${TASK}." >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}/${JANUS_BASE}:${PROJECT_ROOT}/${AMAZE_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE}"

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
  --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" \
  --val_every_steps "${VAL_EVERY_STEPS}" --val_max_batches "${VAL_MAX_BATCHES}" \
  --seed 42 \
  "${RESUME_ARGS[@]}"

echo "Janus FT (${TASK}) complete -> runs/${RUN_NAME}"

# ── Checkpoint selection + scoring on the AMAZE metrics (one wandb run per scored checkpoint) ──
# SELECT=val  -> pick the checkpoint with the lowest validation MSE (recorded in each checkpoint's
#                training_state.json) and score ONLY that one on the full test set. (default for maze)
# SELECT=all  -> score EVERY epoch-checkpoint (trajectory; also the fallback for a run with no recorded
#                validation, e.g. the pre-val Janus-queens run). (default for queens)
# COST of ONE checkpoint = N_test_puzzles x SAMPLES image-gens (queens ~450 puzzles, maze ~3200).
# Resumable: already-scored checkpoints are skipped (.scored marker).
if [[ "${SAMPLE}" != "false" ]]; then
  mapfile -t CKPTS < <(find "${PROJECT_ROOT}/runs/${RUN_NAME}" -type d -name tfmr 2>/dev/null | sort -V)
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "WARN: no checkpoint-*/tfmr found under runs/${RUN_NAME} -> skipping auto-sampling." >&2
  else
    if [[ "${SELECT}" == "val" ]]; then
      BEST=$(python experiments/select_best_ckpt.py janus "${PROJECT_ROOT}/runs/${RUN_NAME}")
      if [[ -n "${BEST}" ]]; then
        echo ">> SELECT=val -> best Janus checkpoint by validation MSE: ${BEST}"
        CKPTS=("${BEST}")
      else
        echo "WARN: SELECT=val but no validation metrics recorded -> scoring the LAST checkpoint only." >&2
        CKPTS=("${CKPTS[-1]}")
      fi
    fi
    AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" python scripts/gen_amaze.py test "${TASK}"   # build the test set ONCE
    echo ">> scoring ${#CKPTS[@]} Janus checkpoint(s) x ${SAMPLES} samples (SELECT=${SELECT})."
    for CKPT in "${CKPTS[@]}"; do
      TAG=$(basename "$(dirname "${CKPT}")")           # checkpoint-<epoch>-<step>
      GEN_DIR="runs/${RUN_NAME}/generated/${TAG}"
      if [[ -f "${GEN_DIR}/.scored" ]]; then echo ">> [${TAG}] already scored -> skip."; continue; fi
      echo ">> [${TAG}] sampling: ${CKPT}"
      python experiments/generate_amaze_ft.py "${TASK}" --backend janus \
        --checkpoint "${CKPT}" --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
      && python experiments/score_amaze_images.py "${TASK}" \
        --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
        --wandb-project "${WANDB_PROJECT}" --run-name "${RUN_NAME}-${TAG}" \
      && touch "${GEN_DIR}/.scored" \
      || echo "WARN: [${TAG}] sampling/scoring failed -- continuing with the next checkpoint."
    done
    echo "Auto-sampling (janus ${TASK}, ${#CKPTS[@]} ckpt(s), SELECT=${SELECT}) done -> wandb ${WANDB_PROJECT}."
  fi
fi
