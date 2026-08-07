#!/bin/bash -l
#SBATCH --job-name=amaze_eval_dit
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/eval_dit.sh <queens|maze> <DIT_CHECKPOINT> [WANDB_PROJECT]
# Env: SAMPLES (Pass@K, default 5).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

TASK="${1:?usage: sbatch eval_dit.sh <queens|maze> <DIT_CHECKPOINT> [WANDB_PROJECT]}"
DIT_CKPT="${2:?need the DiT checkpoint path as arg 2}"
WANDB_PROJECT="${WANDB_PROJECT:-${3:-amaze}}"

case "${TASK}" in
  queens) EXP=amaze_dit_queens_v3 ;;
  maze)   EXP=amaze_dit_maze_v3 ;;
  *) echo "TASK must be 'queens' or 'maze', got '${TASK}'" >&2; exit 1 ;;
esac

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1

if [[ ! -f "${DIT_CKPT}" ]]; then
  echo "ERROR: DiT checkpoint not found: ${DIT_CKPT}" >&2
  exit 1
fi
if [[ "${TASK}" == "maze" && ! -d third_party/amaze/mazes-generator/node_modules ]]; then
  echo "ERROR: maze test generation needs third_party/amaze/mazes-generator/node_modules." >&2
  echo "Run ONCE on a login node: module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
  exit 1
fi

# Ensure the canonical test set exists (idempotent).
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py test "${TASK}"

srun python experiments/sample_amaze_metrics.py \
  experiment=${EXP} \
  +checkpoint="${DIT_CKPT}" \
  +task="${TASK}" \
  +data_root="${PROJECT_ROOT}/data/amaze" \
  +samples_per_puzzle="${SAMPLES:-5}" \
  run.wandb_project="${WANDB_PROJECT}"

echo "DiT (${TASK}) metrics done → ${DIT_CKPT}"
