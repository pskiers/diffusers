#!/bin/bash
# Round-2 launcher for UNCONDITIONAL Sokoban bit-diffusion.
#
# Follow-up sweep after the gradient-accumulation + isolate_transform changes:
#   - Embedded-TRM capacity/data sweep (shared_stack on/off, 4 vs 6 DiT layers,
#     1000 vs 64000 boards, batch 64, grad accumulation 1 vs 4).
#   - Diffusion-TRM self-conditioning / carry-recycling sweep (small 1000-board regime).
#
# Reuses _ablation_job.sh. Unlike ablations_training.sh (which takes data size / batch /
# epochs from each method's BASE CONFIG), this round overrides batch_size,
# dataset.total_train_size and gradient_accumulation_steps EXPLICITLY per run.
#
# Usage:
#   DRY_RUN=1 ./ablations_round2.sh     # preview the exact overrides, submit nothing
#   ./ablations_round2.sh               # sbatch all runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/_ablation_job.sh"

WANDB_PROJECT="Sokoban-Ablation-Round2"
OUTPUT_ROOT="/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/round2"

# 64000-board EMB runs with TRM recursion are slow; 8h (the job default) is not enough.
# Set a generous wall-clock here (adjust to your QOS max). 1000-board runs finish early.
TIME="24:00:00"

BOT_REMOVAL=0.75
EVAL_EVERY=50
NUM_EVAL_SAMPLES=128
INFERENCE_STEPS=400

# Applied to EVERY run. Data size / batch / accumulation are set per-run below.
COMMON=(
  conditioning=unconditional
  num_classes=0
  wandb_project="$WANDB_PROJECT"
  dataset.bot_removal_prob="$BOT_REMOVAL"
  eval_every_n_epochs="$EVAL_EVERY"
  num_eval_samples="$NUM_EVAL_SAMPLES"
  inference_steps="$INFERENCE_STEPS"
)

TRM="sokoban/bit_diffusion/train_trm.py"
EMB="sokoban/bit_diffusion/train_trm_embedded.py"
STD="sokoban/bit_diffusion/train_std.py"

# submit <script> <group> <run_name> [extra hydra overrides...]
submit () {
  local script="$1" group="$2" name="$3"; shift 3
  local args=("${COMMON[@]}" "output_dir=$OUTPUT_ROOT/$name" "$@")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY  %-32s %s\n' "$name" "${args[*]}"
  else
    sbatch --job-name="$name" --time="$TIME" "$JOB" "$script" "$name" "$group" "${args[@]}"
  fi
}

echo

# ============================================================================
# 1) EMBEDDED-TRM — shared_stack, self-cond, 4-layer, aux=0.5
#    Variables swept: data size (1k/64k), grad accumulation (1/4).
#    aux_loss_weight held at 0.5 across ALL emb runs so shared_stack / layers /
#    data / accumulation are the only moving parts.
# ============================================================================
submit "$EMB" embedded emb_shared_4L_aux05_1k_bs64 \
  trm.shared_stack=true self_cond=true model.num_layers=4 aux_loss_weight=0.5 \
  dataset.total_train_size=1000 batch_size=64 gradient_accumulation_steps=1

submit "$EMB" embedded emb_shared_4L_aux05_64k_bs64 \
  trm.shared_stack=true self_cond=true model.num_layers=4 aux_loss_weight=0.5 \
  dataset.total_train_size=64000 batch_size=64 gradient_accumulation_steps=1

submit "$EMB" embedded emb_shared_4L_aux05_64k_bs64_acc4 \
  trm.shared_stack=true self_cond=true model.num_layers=4 aux_loss_weight=0.5 \
  dataset.total_train_size=64000 batch_size=64 gradient_accumulation_steps=4

# ----------------------------------------------------------------------------
#    STANDARD
# ----------------------------------------------------------------------------
submit "$STD" standard std_4L_64k_bs64_acc4 \
  self_cond=true model.num_layers=4 weight_decay=1e-6 \
  dataset.total_train_size=64000 batch_size=64 gradient_accumulation_steps=4

submit "$STD" standard std_6L_64k_bs64_acc4 \
  self_cond=true model.num_layers=6 weight_decay=1e-6 \
  dataset.total_train_size=64000 batch_size=64 gradient_accumulation_steps=4

# ============================================================================
# 2) DIFFUSION-TRM — self-cond / carry-recycle sweep (small 1000-board)
# ============================================================================
submit "$TRM" trm trm_no_selfcond_no_recycle  self_cond=false use_carry_recycling=false
submit "$TRM" trm trm_selfcond                self_cond=true  use_carry_recycling=false
submit "$TRM" trm trm_carry_recycle           self_cond=true  use_carry_recycling=true

# ============================================================================
# 3) RESUME — continue the best shared-stack EMB run for 300 more epochs (-> 600).----------------------------------------------------------------------------
RESUME_CKPT="/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/ablation_uncond/emb_shared_stack/checkpoints/last-v1.ckpt"
submit "$EMB" embedded emb_shared_resume_600 \
  trm.shared_stack=true self_cond=true model.num_layers=2 aux_loss_weight=0.3 \
  dataset.total_train_size=1000 batch_size=32 num_epochs=600 \
  resume_from_checkpoint="$RESUME_CKPT"

echo
echo "submitted. monitor: squeue --me   |   logs: trm_sokoban/stdout & stderr"
