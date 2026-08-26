#!/bin/bash -l
#SBATCH --job-name=amaze_bagel_ft
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

# Reference AMAZE config (run_sft.sh): 5000 opt-steps. At ~35 s/step on one GH200
# (cpu_offload) that is ~48.6 h > the 48 h wall, so a single job lands ~4900 steps;
# resubmit the same RUN_NAME to finish (auto_resume=true + save_every=100 chain from
# the last checkpoint, losing <=100 steps).
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
LR="${LR:-1e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
# LoRA is the practical single-GPU path for 14.6B (full FT can't save on one GH200 and
# is network-bound on multi-node). USE_LORA=true -> tiny trainable set, fits+saves on
# one GPU, saves only the adapter. cpu_offload not needed for LoRA.
USE_LORA="${USE_LORA:-false}"
if [[ "${USE_LORA}" == "true" ]]; then CPU_OFFLOAD="${CPU_OFFLOAD:-false}"; else CPU_OFFLOAD="${CPU_OFFLOAD:-true}"; fi
RUN_NAME="${RUN_NAME:-ft_bagel_${TASK}}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to a local BAGEL-7B-MoT snapshot}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
# reduce allocator fragmentation so the checkpoint full_state_dict gather has headroom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# newer libstdc++ for compiled extensions (insurance; harmless if unused)
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${BAGEL_BASE}" ]] || { echo "ERROR: ${BAGEL_BASE} missing. Run: bash ${AMAZE_DIR}/setup_ft_code.sh" >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || { echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing. Run: python scripts/export_amaze_for_ft.py ${TASK}." >&2; exit 1; }

# ── Assemble: overlay AMAZE sft.py + maze data files into the base Bagel repo ──
cp "${BAGEL_SFT}/sft.py" "${BAGEL_BASE}/sft.py"
for f in maze_dataset.py maze_packed_dataset.py; do
  [[ -f "${AMAZE_DIR}/infer/bagel/data/${f}" ]] && cp "${AMAZE_DIR}/infer/bagel/data/${f}" "${BAGEL_BASE}/data/${f}"
done

# AMAZE sft.py calls try_load_ckpt(..., use_lora=, use_lora_checkpoint=) and
# fsdp_save_ckpt(..., use_lora=, save_lora_only=); upstream Bagel's fsdp_utils.py
# signatures lack those. We don't use LoRA, so make both tolerate (ignore) extra kwargs.
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
    close = i - 1  # index of the signature's closing ")"
    sig = src[m.end():close]
    if "**" in sig:
        return src  # already accepts **kwargs
    # insert after the last real char; respect a trailing comma to avoid ", , **kwargs"
    stripped = sig.rstrip()
    sep = " **_lora_kwargs" if stripped.endswith(",") else ", **_lora_kwargs"
    insert_at = m.end() + len(stripped)
    return src[:insert_at] + sep + src[insert_at:]
for fn in ("fsdp_save_ckpt", "try_load_ckpt"):
    s = tolerate(s, fn)
# FSDP + LoRA: a layer mixes frozen base and trainable adapter params, so FSDP needs
# use_orig_params=True (else "uniform requires_grad" error). Harmless for full FT.
if "use_orig_params" not in s:
    s = re.sub(r"(return FSDP\(\s*\n\s*original_model,)",
               r"\1\n        use_orig_params=True,", s, count=1)
open(p, "w").write(s)
print("patched fsdp_utils.py (tolerate LoRA kwargs + use_orig_params)")
PY

cd "${BAGEL_BASE}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Single 96GB GH200: FULL_SHARD + cpu_offload keeps only the active layer's params on
# GPU (rest on host RAM), so the 14.6B model AND the checkpoint full_state_dict gather
# fit. (NO_SHARD keeps the whole fp32 master + bf16 copy resident ~93/95GB -> save OOMs.)
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
  --sharding_strategy FULL_SHARD --backward_prefetch BACKWARD_PRE \
  --num_replicate 1 --num_shard 1 --cpu_offload "${CPU_OFFLOAD}" --use_lora "${USE_LORA}" --save_lora_only "${USE_LORA}"

echo "Bagel FT (${TASK}) complete -> runs/${RUN_NAME}"
