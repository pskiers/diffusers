#!/bin/bash -l
#SBATCH --job-name=amaze_trajectory
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Capture the denoising trajectory (per-timestep x0 images + a labelled filmstrip)
# for a TRM or DiT checkpoint, on Helios (GH200).
#
# Usage:
#   sbatch slurm_scripts/trajectory_amaze.sh dit <maze|queens> <dit.pt>
#   sbatch slurm_scripts/trajectory_amaze.sh trm <maze|queens> <thinker.pt> <painter.pt>
# Env: COMBO PUZZLES(4) STEPS(8) CAPTURE_XT(false) SEED(0) WANDB_PROJECT(amaze) DATA_ROOT

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

usage() {
  echo "usage: sbatch slurm_scripts/trajectory_amaze.sh <trm|dit> <maze|queens> <checkpoint> [painter.pt]" >&2
  exit 1
}

MODEL="${1:-}"; TASK="${2:-}"; CKPT="${3:-}"; PAINTER_CKPT="${4:-}"
[[ -z "${MODEL}" || -z "${TASK}" || -z "${CKPT}" ]] && usage
[[ "${MODEL}" != "trm" && "${MODEL}" != "dit" ]] && { echo "MODEL must be trm|dit" >&2; usage; }
[[ "${TASK}" != "maze" && "${TASK}" != "queens" ]] && { echo "TASK must be maze|queens" >&2; usage; }
[[ "${MODEL}" == "trm" && -z "${PAINTER_CKPT}" ]] && { echo "trm needs a painter checkpoint (4th arg)" >&2; usage; }

COMBO="${COMBO:-}"
PUZZLES="${PUZZLES:-4}"
STEPS="${STEPS:-8}"
CAPTURE_XT="${CAPTURE_XT:-false}"
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/amaze}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

ARGS=(
  +checkpoint="${CKPT}"
  +task="${TASK}"
  +data_root="${DATA_ROOT}"
  +trajectory_puzzles="${PUZZLES}"
  +trajectory_num_steps="${STEPS}"
  +trajectory_capture_xt="${CAPTURE_XT}"
  +trajectory_seed="${SEED}"
  run.wandb_project="${WANDB_PROJECT}"
)
[[ -n "${COMBO}" ]] && ARGS+=( +trajectory_combo="${COMBO}" )

if [[ "${MODEL}" == "trm" ]]; then
  ARGS=( experiment=amaze_thinker_v2_controlnet painter.checkpoint="${PAINTER_CKPT}" "${ARGS[@]}" )
else
  ARGS=( experiment="amaze_dit_${TASK}" "${ARGS[@]}" )
fi

srun python experiments/sample_amaze_trajectory.py "${ARGS[@]}"

echo "Trajectory (${MODEL}/${TASK}) complete — PNGs under $(dirname "${CKPT}")/trajectory/${TASK}."
