#!/bin/bash -l
#SBATCH --job-name=amaze_gen
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage: sbatch slurm_scripts/gen_amaze.sh <test|train|ft> <maze|queens|both> [--shape all --size all --image-size 256]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

MODE="${1:-test}"
TASK="${2:-both}"
EXTRA="${*:3}"

[[ "${MODE}" != "test" && "${MODE}" != "train" && "${MODE}" != "ft" ]] && { echo "MODE must be test|train|ft" >&2; exit 1; }
[[ "${TASK}" != "maze" && "${TASK}" != "queens" && "${TASK}" != "both" ]] && { echo "TASK must be maze|queens|both" >&2; exit 1; }

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 GCCcore/14.3.0 nodejs/22.17.1

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

if [[ "${MODE}" != "ft" && ( "${TASK}" == "maze" || "${TASK}" == "both" ) ]]; then
  if [[ ! -d third_party/amaze/mazes-generator/node_modules ]]; then
    echo "ERROR: third_party/amaze/mazes-generator/node_modules is missing." >&2
    echo "Run ONCE on the LOGIN node (needs internet):" >&2
    echo "  module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
export AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze"

if [[ "${TASK}" == "both" ]]; then
  python scripts/gen_amaze.py "${MODE}" maze   ${EXTRA}
  python scripts/gen_amaze.py "${MODE}" queens ${EXTRA}
else
  python scripts/gen_amaze.py "${MODE}" "${TASK}" ${EXTRA}
fi

echo "Generation (${MODE} ${TASK} ${EXTRA}) complete -> ${AMAZE_OUT_ROOT}"
