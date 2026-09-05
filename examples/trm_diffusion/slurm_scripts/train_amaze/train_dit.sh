#!/bin/bash -l
#SBATCH --job-name=amaze_dit
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Standalone (no-TRM) pixel-space concat-conditioned DiT baseline (Palette/SR3 style, CFG off).
#   maze   -> ALL shapes (square, hexagon, triangle, circle) x ALL sizes; every geometry in-distribution.
#   queens -> multi-scale mix n=4..10, scored across n=4..10.
# Maze data must already exist (generate with gen_amaze.sh on a Node-capable node); the
# gen_amaze.py calls below are idempotent no-ops when the parquet is present.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_dit.sh <maze|queens> [WANDB_PROJECT] [RUN_NAME]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_dit.sh <maze|queens> [WANDB_PROJECT] [RUN_NAME]}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

DIT_STEPS=${DIT_STEPS:-80000}
DIT_MAX_SECONDS=${DIT_MAX_SECONDS:-0}   # 0 = no wall-clock limit
WANDB_PROJECT="${WANDB_PROJECT:-${2:-amaze}}"
RUN_NAME="${RUN_NAME:-${3:-${TASK}_dit${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

if [[ "${TASK}" == "maze" ]]; then GEN_ARGS=(--shape all --size all); else GEN_ARGS=(--size all); fi

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train "${TASK}" "${GEN_ARGS[@]}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/train_${TASK}/all_train_size144"

srun python train_trm.py experiment="amaze_dit_${TASK}" \
  data.amaze_root="${DATA_DIR}" \
  train.num_steps=${DIT_STEPS} \
  train.max_seconds=${DIT_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "${TASK} DiT complete -> runs/${RUN_NAME}/checkpoint_final.pt"

if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py test "${TASK}"
  srun python experiments/sample_amaze_metrics.py \
    experiment="amaze_dit_${TASK}" \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task="${TASK}" \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed -- checkpoint is safe in runs/${RUN_NAME}."
  echo "Metrics (${TASK}) done -- logged into wandb run ${RUN_NAME}."
fi
