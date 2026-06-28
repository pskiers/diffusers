#!/bin/bash
# Submits one SLURM job per configuration (via _ablation_job.sh).
#   standard   train_std.py           plain DiT bit-diffusion             (group: standard)
#   trm        train_trm.py           TRM recursion wrapping a DiT        (group: trm)
#   embedded   train_trm_embedded.py  TRM refiner inside every DiT layer  (group: embedded)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/_ablation_job.sh"

WANDB_PROJECT="Sokoban-Ablation-Uncond"
OUTPUT_ROOT="/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/ablation_uncond"

BOT_REMOVAL=0.75
EVAL_EVERY=50
NUM_EVAL_SAMPLES=128
INFERENCE_STEPS=400
EPOCHS=600
BATCH_SIZE=64
TRAIN_SIZE=65536

# Overrides applied to EVERY job (data size + epochs are set per-method below).
COMMON=(
  conditioning=unconditional
  num_classes=0
  wandb_project="$WANDB_PROJECT"
  dataset.bot_removal_prob="$BOT_REMOVAL"
  eval_every_n_epochs="$EVAL_EVERY"
  num_eval_samples="$NUM_EVAL_SAMPLES"
  inference_steps="$INFERENCE_STEPS"
  num_epochs="$EPOCHS"
  batch_size="$BATCH_SIZE"
  dataset.total_train_size="$TRAIN_SIZE"
)

STD="sokoban/bit_diffusion/train_std.py"
TRM="sokoban/bit_diffusion/train_trm.py"
EMB="sokoban/bit_diffusion/train_trm_embedded.py"

# submit <script> <group> <run_name> [extra hydra overrides...]
submit () {
  local script="$1" group="$2" name="$3"; shift 3
  local args=("${COMMON[@]}" "output_dir=$OUTPUT_ROOT/$name" "$@")
  sbatch --job-name="$name" "$JOB" "$script" "$name" "$group" "${args[@]}"
}

echo

# 1) STANDARD DIFFUSION
# submit "$STD" standard std_6L_baseline  \
#   model.num_layers=6 model.num_attention_heads=4 model.attention_head_dim=64 \
#   self_cond=true weight_decay=1e-6 gradient_accumulation_steps=4


# 2) DIFFUSION-TRM
submit "$TRM" trm trm        use_carry_recycling=true use_carry_persistence=true
# submit "$TRM" trm trm_cr_75        use_carry_recycling=true carry_recycle_prob=0.75
# submit "$TRM" trm trm_cr_50        use_carry_recycling=true carry_recycle_prob=0.5
# submit "$TRM" trm trm_cr_25        use_carry_recycling=true carry_recycle_prob=0.25

# submit "$TRM" trm trm_grad_acc     use_carry_recycling=true gradient_accumulation_steps=4
# submit "$TRM" trm trm_grad_acc     use_carry_recycling=true model.num_layers=4
# submit "$TRM" trm trm_n_6          use_carry_recycling=true carry_recycle_prob=0.5 trm.n=6

# 3) EMBEDDED-TRM  (group: embedded)
# submit "$EMB" embedded emb_isolate_transform \
#   trm.shared_stack=true self_cond=true model.num_layers=4 aux_loss_weight=0.5 \
#   gradient_accumulation_steps=4 trm.isolate_transform=true

# submit "$EMB" embedded emb_big \
#   trm.num_inner_layers=2 model.num_layers=6 trm.shared_stack=true self_cond=true resume_from_checkpoint=/net/tscratch/people/plgmgrzanka/trm_sokoban/outputs/ablation_uncond/emb_big/checkpoints/best-epoch=58-step=60416.ckpt

# submit "$EMB" embedded emb_big_no_self_cond \
#   trm.num_inner_layers=2 model.num_layers=6 trm.shared_stack=true self_cond=false

echo
echo "submitted. monitor: squeue --me   |   logs: trm_sokoban/stdout & stderr"
