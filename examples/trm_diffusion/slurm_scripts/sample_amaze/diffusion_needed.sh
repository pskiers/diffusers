#!/bin/bash -l
#SBATCH --job-name=diffusion_needed
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# PaTh rebuttal, "is diffusion needed for single-solution boards": train + score ONE deterministic
# baseline (experiments/amaze_det_baselines.py; see its docstring for the exact recipe):
#   unet      a plain UNet, unsolved maze -> solved image in one pass (~PaTh's total parameter count)
#   trm_unet  PaTh's thinker + translator steering a TRAINABLE UNet that paints in one pass (no x_t, no diffusion)
# The third model, PaTh without x_t, is trained and evaluated on titan.
#
# Usage (from the trm_diffusion dir; account/partition/GPU from slurm_scripts/cluster_env.sh):
#   source slurm_scripts/cluster_env.sh && sbatch "${CLUSTER_GPU_SBATCH[@]}" slurm_scripts/sample_amaze/diffusion_needed.sh <unet|trm_unet>
# Env: STEPS (40000), MAX_MINUTES (2700: stop training in time to score; resumable), WANDB_PROJECT (PaTh-rebuttal),
#      MAZE_OOD_SCALES (10)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL="${1:?usage: diffusion_needed.sh <unet|trm_unet>}"
[[ "${MODEL}" == unet || "${MODEL}" == trm_unet ]] || { echo "MODEL must be unet|trm_unet" >&2; exit 1; }

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MAZE_OOD_SCALES="${MAZE_OOD_SCALES-10}"

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} > 2 ? ${SLURM_CPUS_PER_TASK:-8} - 2 : 1 ))
srun python experiments/amaze_det_baselines.py --model "${MODEL}" \
  --data_root "${PROJECT_ROOT}/data/amaze" --out_dir "runs/diffneed_${MODEL}" \
  --steps "${STEPS:-40000}" --max_minutes "${MAX_MINUTES:-2700}" --workers "${WORKERS}" \
  --wandb_project "${WANDB_PROJECT:-PaTh-rebuttal}" --wandb_name "diffneed_${MODEL}"
echo "diffneed_${MODEL} -> runs/diffneed_${MODEL}/amaze_metrics_maze.json"
