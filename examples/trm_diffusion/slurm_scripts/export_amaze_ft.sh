#!/bin/bash -l
#SBATCH --job-name=amaze_export_ft
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Export the regenerated amaze parquet into the Bagel/Janus FT layout
# (data/amaze/ft/{maze,queens}/maze_dataset_{train,test}.parquet).
#
# Usage: sbatch slurm_scripts/export_amaze_ft.sh [maze|queens|both]

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

TASK="${1:-both}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" || "${TASK}" == "both" ]] || { echo "TASK must be maze|queens|both" >&2; exit 1; }

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

export PYTHONUNBUFFERED=1
python scripts/export_amaze_for_ft.py "${TASK}"
