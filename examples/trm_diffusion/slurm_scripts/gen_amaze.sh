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
#
# Generate the AMAZE datasets. WHAT gets generated (sizes, sample counts, output
# path, columns) lives in configs/data/amaze_generation.yaml — this script only
# picks the task and the stage.
#
# Usage: sbatch slurm_scripts/gen_amaze.sh [maze|queens|both] [all|test|train|verify|report] [extra args]
#   sbatch slurm_scripts/gen_amaze.sh                          # everything
#   sbatch slurm_scripts/gen_amaze.sh maze test                # maze test parquets only
#   sbatch slurm_scripts/gen_amaze.sh both all -o overwrite=true
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
CONFIG="${CONFIG:-configs/data/amaze_generation.yaml}"

TASK="${1:-both}"
STAGE="${2:-all}"
EXTRA=("${@:3}")

[[ "${TASK}" == "maze" || "${TASK}" == "queens" || "${TASK}" == "both" ]] \
  || { echo "TASK must be maze|queens|both" >&2; exit 1; }
[[ "${STAGE}" =~ ^(all|test|train|verify|report)$ ]] \
  || { echo "STAGE must be all|test|train|verify|report" >&2; exit 1; }

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 GCCcore/14.3.0 nodejs/22.17.1

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

# The maze generator is a node package
if [[ "${TASK}" != "queens" && "${STAGE}" != "verify" && "${STAGE}" != "report" ]]; then
  if [[ ! -d third_party/amaze/mazes-generator/node_modules ]]; then
    echo "ERROR: third_party/amaze/mazes-generator/node_modules is missing." >&2
    echo "Run ONCE on the LOGIN node (needs internet):" >&2
    echo "  module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1

ARGS=(--config "${CONFIG}" --task "${TASK}" --stage "${STAGE}")
[[ -n "${AMAZE_OUT_ROOT:-}" ]] && ARGS+=(--output-root "${AMAZE_OUT_ROOT}")

python scripts/gen_amaze.py "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
