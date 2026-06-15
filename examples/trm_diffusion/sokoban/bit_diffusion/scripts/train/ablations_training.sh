#!/bin/bash
# Ablation launcher for UNCONDITIONAL Sokoban bit-diffusion.
#
# Submits one SLURM job per configuration (via _ablation_job.sh). Every run logs
# to a SINGLE W&B project, grouped by method.
#
#   standard   train_std.py           plain DiT bit-diffusion          (group: standard)
#   trm        train_trm.py           TRM recursion wrapping a DiT      (group: trm)
#   embedded   train_trm_embedded.py  TRM refiner inside every DiT layer (group: embedded)
#
# Data size, epochs, batch size, precision and warmup come from each method's BASE
# CONFIG (intentionally different): standard trains on the full 64000-board set,
# trm/embedded on 1000 — a data-efficiency comparison, not an equal-data control.
# Shared across all runs (via COMMON): bot_removal_prob, eval budget, inference steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/_ablation_job.sh"

WANDB_PROJECT="Sokoban-Ablation-Uncond"
OUTPUT_ROOT="/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/ablation_uncond"

BOT_REMOVAL=0.75
EVAL_EVERY=50
NUM_EVAL_SAMPLES=128
INFERENCE_STEPS=400

# Overrides applied to EVERY job (data size + epochs are set per-method below).
COMMON=(
  conditioning=unconditional
  num_classes=0
  wandb_project="$WANDB_PROJECT"
  dataset.bot_removal_prob="$BOT_REMOVAL"
  eval_every_n_epochs="$EVAL_EVERY"
  num_eval_samples="$NUM_EVAL_SAMPLES"
  inference_steps="$INFERENCE_STEPS"
)

STD="sokoban/bit_diffusion/train_std.py"
TRM="sokoban/bit_diffusion/train_trm.py"
EMB="sokoban/bit_diffusion/train_trm_embedded.py"

# submit <script> <group> <run_name> [extra hydra overrides...]
submit () {
  local script="$1" group="$2" name="$3"; shift 3
  local args=("${COMMON[@]}" "output_dir=$OUTPUT_ROOT/$name" "$@")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY  %-22s %s\n' "$name" "${args[*]}"
  else
    sbatch --job-name="$name" "$JOB" "$script" "$name" "$group" "${args[@]}"
  fi
}

echo

# ============================================================================
# 1) STANDARD DIFFUSION  (group: standard) — FULL dataset (64000)
# ============================================================================
submit "$STD" standard std_6L_baseline  \
  model.num_layers=6 model.num_attention_heads=4 model.attention_head_dim=64 \
  self_cond=false weight_decay=1e-6 gradient_accumulation_steps=1
submit "$STD" standard std_6L_selfcond  \
  model.num_layers=6 model.num_attention_heads=4 model.attention_head_dim=64 \
  self_cond=true weight_decay=1e-6 gradient_accumulation_steps=1
submit "$STD" standard std_6L_gradaccum  \
  model.num_layers=6 model.num_attention_heads=4 model.attention_head_dim=64 \
  self_cond=false weight_decay=1e-6 gradient_accumulation_steps=4
submit "$STD" standard std_12L_bigger  \
  model.num_layers=12 model.num_attention_heads=6 model.attention_head_dim=64 \
  self_cond=false weight_decay=1e-4 gradient_accumulation_steps=1

# ============================================================================
# 2) DIFFUSION-TRM  (group: trm) — SMALL dataset (1000). Tiny DiT (2 layers, width 256) wrapped in TRM recursion.
# Reference point: config defaults (self_cond=true, n=6, T=3, n_sup 2..6, no recycle).
# ============================================================================
submit "$TRM" trm trm_base           
submit "$TRM" trm trm_no_selfcond     self_cond=false
submit "$TRM" trm trm_carry_recycle   use_carry_recycling=true
submit "$TRM" trm trm_n_4_T_2         use_carry_recycling=true trm.n=4 trm.T=2

# ============================================================================
# 3) EMBEDDED-TRM  (group: embedded) — SMALL dataset (1000). A TRM refiner runs inside every DiT layer.
# ============================================================================
submit "$EMB" embedded emb_base         
submit "$EMB" embedded emb_aux_high      aux_loss_weight=0.5
submit "$EMB" embedded emb_steps4        model.num_layers=4
submit "$EMB" embedded emb_weight_tied   trm.weight_tied=true model.num_layers=4
submit "$EMB" embedded emb_shared_stack  trm.shared_stack=true
# submit "$EMB" embedded emb_more_reason   trm.n_inner=8 trm.T=4

echo
echo "submitted. monitor: squeue --me   |   logs: trm_sokoban/stdout & stderr"
