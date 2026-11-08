#!/bin/bash
# Run FROM YOUR LAPTOP (needs ssh aliases `helios` and `athena`): submit the AMAZE FLOPs analysis for
# each task to BOTH clusters, then watch the queues and, as soon as one cluster starts a task's job,
# cancel that task's twin on the other cluster (see flops_race.sh).
#
#   bash slurm_scripts/sample_amaze/submit_flops_both.sh [maze] [queens]     (default: both tasks)
# Env: TAG (suffix for job name / state key), TIME (12:00:00), HELIOS_DIR / ATHENA_DIR (repo roots on the clusters), STAGES, PER_COMBO, WANDB_PROJECT, SMOKE
set -euo pipefail
HELIOS_DIR="${HELIOS_DIR:-/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion}"
ATHENA_DIR="${ATHENA_DIR:-/net/tscratch/people/plgmgrzanka/trm_flops}"
TASKS=("$@"); [[ ${#TASKS[@]} -gt 0 ]] || TASKS=(maze queens)
FWD="STAGES='${STAGES:-dm trm_flat profile cut_at_k converge carry report}' ${PER_COMBO:+PER_COMBO=$PER_COMBO} WANDB_PROJECT=${WANDB_PROJECT:-amaze_flops} SMOKE=${SMOKE:-0}"
STATE="${STATE:-$PWD/flops_race_state.txt}"; touch "${STATE}"   # appended to: several submissions (TAG=report / TAG=tune) can share one state file
for t in "${TASKS[@]}"; do
  hid=$(ssh helios "cd ${HELIOS_DIR} && mkdir -p slurm_outputs && ${FWD} sbatch --parsable --job-name=flops_${t}${TAG:+_$TAG} --time=${TIME:-12:00:00} \
        --account=plgdiffusion3-gpu-gh200 --partition=plgrid-gpu-gh200 slurm_scripts/sample_amaze/flops_analysis.sh ${t}")
  aid=$(ssh athena "cd ${ATHENA_DIR} && mkdir -p slurm_outputs && ${FWD} sbatch --parsable --job-name=flops_${t}${TAG:+_$TAG} --time=${TIME:-12:00:00} \
        --account=plgdiffusion3-gpu-a100 --partition=plgrid-gpu-a100 slurm_scripts/sample_amaze/flops_analysis.sh ${t}")
  echo "${t}${TAG:+_$TAG} ${hid%%;*} ${aid%%;*}" | tee -a "${STATE}"
done
echo "state -> ${STATE}; now: bash $(dirname "$0")/flops_race.sh ${STATE}"
