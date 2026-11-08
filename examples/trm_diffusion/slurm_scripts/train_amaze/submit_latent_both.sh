#!/bin/bash
# Run FROM YOUR LAPTOP (needs ssh aliases `helios` and `athena`): submit the LATENT
# painter-thinker pipeline (train_latent_painter_thinker.sh) for each task to BOTH
# clusters, then race them: whichever cluster starts a task's job first keeps it and
# its twin on the other cluster is cancelled (slurm_scripts/sample_amaze/flops_race.sh).
#
#   bash slurm_scripts/train_amaze/submit_latent_both.sh [maze] [queens]    (default: both)
# Env: HELIOS_DIR / ATHENA_DIR (trm_diffusion copies on the clusters), STATE (race state
#      file), PAINTER_STEPS, THINKER_STEPS, WANDB_PROJECT, NO_RACE=1 (submit only).
set -euo pipefail
HELIOS_DIR="${HELIOS_DIR:-/net/scratch/hscra/plgrid/plgmgrzanka/trm_latent}"
ATHENA_DIR="${ATHENA_DIR:-/net/tscratch/people/plgmgrzanka/trm_latent}"
TASKS=("$@"); [[ ${#TASKS[@]} -gt 0 ]] || TASKS=(maze queens)
FWD="PAINTER_STEPS=${PAINTER_STEPS:-40000} THINKER_STEPS=${THINKER_STEPS:-40000} WANDB_PROJECT=${WANDB_PROJECT:-amaze_latent}"
SCRIPT=slurm_scripts/train_amaze/train_latent_painter_thinker.sh
STATE="${STATE:-$PWD/latent_race_state.txt}"; : > "${STATE}"
for t in "${TASKS[@]}"; do
  hid=$(ssh helios "cd ${HELIOS_DIR} && mkdir -p slurm_outputs && ${FWD} sbatch --parsable --job-name=latent_${t} \
        --account=plgdiffusion3-gpu-gh200 --partition=plgrid-gpu-gh200 ${SCRIPT} ${t}")
  aid=$(ssh athena "cd ${ATHENA_DIR} && mkdir -p slurm_outputs && ${FWD} sbatch --parsable --job-name=latent_${t} \
        --account=plgdiffusion3-gpu-a100 --partition=plgrid-gpu-a100 ${SCRIPT} ${t}")
  echo "${t} ${hid%%;*} ${aid%%;*}" | tee -a "${STATE}"
done
echo "state -> ${STATE}"
[[ "${NO_RACE:-0}" == "1" ]] || exec bash "$(dirname "$0")/../sample_amaze/flops_race.sh" "${STATE}"
