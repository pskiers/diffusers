#!/bin/bash -l
#SBATCH --job-name=amaze_bagel_ft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Bagel FULL fine-tune on ONE Helios node with 4x GH200 (NVLink) -- the authors' recipe
# (run_sft.sh uses 4 GPUs on one node). FSDP FULL_SHARD across the 4 GPUs shards the
# 14.6B model (~57GB/GPU) so it fits WITHOUT cpu_offload and the sharded checkpoint save
# fits too; NVLink (not the Slingshot network) keeps all-gather fast.
#
# Usage: sbatch slurm_scripts/train_bagel_ft.sh <maze|queens>
# Env: BAGEL_MODEL_PATH (REQUIRED), TOTAL_STEPS (5000), SAVE_EVERY (500), LR (1e-5),
#      WANDB_PROJECT (amaze_final), RUN_NAME, CPU_OFFLOAD (false), VENV (default $SCRATCH/trm_helios_venv).
# NB: each checkpoint = full model (~29GB) + optimizer state (~100GB+), and fsdp_save_ckpt does NOT
#     rotate/prune. SAVE_EVERY 500 over 5000 steps = 10 ckpts (~1.5TB). eval_every=50 keeps the wandb
#     val curve fine-grained so you can locate + pick the best checkpoint by val loss / Pass@K.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_bagel_ft.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

AMAZE_DIR="third_party/amaze"
BAGEL_SFT="${AMAZE_DIR}/sft/bagel"
BAGEL_BASE="${BAGEL_SFT}/Bagel"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/amaze/ft/${TASK}}"

TOTAL_STEPS="${TOTAL_STEPS:-5000}"   # authors' run_sft.sh recipe (optimizer updates; dataset-size-independent)
SAVE_EVERY="${SAVE_EVERY:-500}"      # ckpt cadence only (no effect on trained model). 5000/500=10 ckpts ~1.5TB; each ~150GB, NOT rotated -> lower only if scratch allows
LR="${LR:-1e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-ft_bagel_${TASK}}"
CPU_OFFLOAD="${CPU_OFFLOAD:-false}"
NPROC="${NPROC:-4}"
SAMPLE="${SAMPLE:-true}"
SAMPLES="${SAMPLES:-5}"
SELECT="${SELECT:-val}"   # val: score only the lowest-val-MSE checkpoint; all: score every checkpoint
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to a local BAGEL-7B-MoT snapshot}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${BAGEL_BASE}" ]] || { echo "ERROR: ${BAGEL_BASE} missing. Run: bash ${AMAZE_DIR}/setup_ft_code.sh" >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/gen_amaze.py ft ${TASK}." >&2; exit 1; }

# ── Assemble: overlay AMAZE sft.py + maze data + tolerate LoRA kwargs / use_orig_params ──
cp "${BAGEL_SFT}/sft.py" "${BAGEL_BASE}/sft.py"
for f in maze_dataset.py maze_packed_dataset.py; do
  [[ -f "${AMAZE_DIR}/infer/bagel/data/${f}" ]] && cp "${AMAZE_DIR}/infer/bagel/data/${f}" "${BAGEL_BASE}/data/${f}"
done
FSDP_UTILS="${BAGEL_BASE}/train/fsdp_utils.py"
python - "${FSDP_UTILS}" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
def tolerate(src, fn):
    m = re.search(r"def %s\(" % fn, src)
    if not m:
        return src
    i, depth = m.end(), 1
    while depth:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    close = i - 1
    sig = src[m.end():close]
    if "**" in sig:
        return src
    stripped = sig.rstrip()
    sep = " **_lora_kwargs" if stripped.endswith(",") else ", **_lora_kwargs"
    insert_at = m.end() + len(stripped)
    return src[:insert_at] + sep + src[insert_at:]
for fn in ("fsdp_save_ckpt", "try_load_ckpt"):
    s = tolerate(s, fn)
if "use_orig_params" not in s:
    s = re.sub(r"(return FSDP\(\s*\n\s*original_model,)",
               r"\1\n        use_orig_params=True,", s, count=1)
open(p, "w").write(s)
print("patched fsdp_utils.py (tolerate LoRA kwargs + use_orig_params)")
PY

cd "${BAGEL_BASE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Single node, ${NPROC} GPUs over NVLink -> torchrun --standalone (no cross-node rendezvous).
# FULL_SHARD across ${NPROC} ranks shards params/optimizer so full FT fits without offload.
srun torchrun --standalone --nproc_per_node="${NPROC}" sft.py \
  --maze_dataset_path "${DATA_DIR}" \
  --model_path "${BAGEL_MODEL_PATH}" \
  --resume_from "${BAGEL_MODEL_PATH}" \
  --results_dir "${PROJECT_ROOT}/runs/${RUN_NAME}" \
  --checkpoint_dir "${PROJECT_ROOT}/runs/${RUN_NAME}/checkpoints" \
  --wandb_project "${WANDB_PROJECT}" --wandb_name "${RUN_NAME}" --wandb_offline false \
  --visual_gen true --visual_und true \
  --finetune_from_ema true --resume_model_only true --finetune_from_hf true --auto_resume true \
  --total_steps "${TOTAL_STEPS}" --save_every "${SAVE_EVERY}" --log_every 1 --eval_every 50 --eval_samples 8 \
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
  --sharding_strategy FULL_SHARD --backward_prefetch BACKWARD_PRE \
  --num_replicate 1 --num_shard "${NPROC}" --cpu_offload "${CPU_OFFLOAD}" --use_lora false

echo "Bagel FULL FT (${TASK}, 1 node x ${NPROC} GPU) complete -> runs/${RUN_NAME}"

# ── Checkpoint selection + scoring on the AMAZE metrics (one wandb run per scored checkpoint) ──
# SELECT=val (default) -> pick the checkpoint with the lowest validation MSE (recorded in each step's
#   val.json) and score ONLY that one on the full test set. SELECT=all -> score EVERY step-checkpoint.
# COST of ONE checkpoint = N_test_puzzles x SAMPLES image-gens (queens ~450 puzzles, maze ~3200; Bagel also
# does 50 diffusion steps/img + a base-model reload). Resumable: scored checkpoints are skipped (.scored).
if [[ "${SAMPLE}" != "false" ]]; then
  cd "${PROJECT_ROOT}"
  CKPT_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}/checkpoints"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "WARN: ${CKPT_DIR} not found -> skipping auto-sampling." >&2
  else
    mapfile -t CKPTS < <(find "${CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort -V)
    if [[ ${#CKPTS[@]} -eq 0 ]]; then
      echo "WARN: no numeric step dirs under ${CKPT_DIR} -> skipping auto-sampling." >&2
    else
      if [[ "${SELECT}" == "val" ]]; then
        BEST=$(python experiments/select_best_ckpt.py bagel "${CKPT_DIR}")
        if [[ -n "${BEST}" ]]; then
          echo ">> SELECT=val -> best Bagel checkpoint by validation MSE: ${BEST}"
          CKPTS=("${BEST}")
        else
          echo "WARN: SELECT=val but no validation metrics recorded -> scoring the LAST checkpoint only." >&2
          CKPTS=("${CKPTS[-1]}")
        fi
      fi
      AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" python scripts/gen_amaze.py test "${TASK}"   # build the test set ONCE
      echo ">> scoring ${#CKPTS[@]} Bagel checkpoint(s) x ${SAMPLES} samples (SELECT=${SELECT})."
      for CKPT in "${CKPTS[@]}"; do
        TAG=$(basename "${CKPT}")                       # <step> (zero-padded)
        GEN_DIR="runs/${RUN_NAME}/generated/${TAG}"
        if [[ -f "${GEN_DIR}/.scored" ]]; then echo ">> [step ${TAG}] already scored -> skip."; continue; fi
        echo ">> [step ${TAG}] sampling: ${CKPT}"
        python experiments/generate_amaze_ft.py "${TASK}" --backend bagel \
          --checkpoint "${CKPT}" --bagel-model-path "${BAGEL_MODEL_PATH}" \
          --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
        && python experiments/score_amaze_images.py "${TASK}" \
          --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
          --wandb-project "${WANDB_PROJECT}" --run-name "${RUN_NAME}-${TAG}" \
        && touch "${GEN_DIR}/.scored" \
        || echo "WARN: [step ${TAG}] sampling/scoring failed -- continuing with the next checkpoint."
      done
      echo "Auto-sampling (bagel ${TASK}, ${#CKPTS[@]} ckpt(s), SELECT=${SELECT}) done -> wandb ${WANDB_PROJECT}."
    fi
  fi
fi
