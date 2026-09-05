#!/bin/bash -l
#SBATCH --job-name=amaze_painter_thinker
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

# Painter -> thinker pipeline, grid 12 (amaze_thinker_v2_controlnet). Stages toggle via flags
# (each defaults to true): painter=<bool> thinker=<bool> sample=<bool>.
#   maze   -> ALL shapes x ALL sizes, eval_callbacks=amaze
#   queens -> multi-scale mix n=4..10, eval_callbacks=amaze_queens
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_painter_thinker.sh <maze|queens> [painter=true thinker=true sample=true]
# Env: PAINTER_STEPS/THINKER_STEPS (40000), SAMPLES (Pass@K, 5), WANDB_PROJECT (amaze),
#      RUN_NAME, and PAINTER_CKPT/THINKER_CKPT to reuse checkpoints when a stage is skipped.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

TASK="${1:?usage: sbatch train_painter_thinker.sh <maze|queens> [painter= thinker= sample=]}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }
shift

PAINTER=true; THINKER=true; SAMPLE=true
for arg in "$@"; do
  case "${arg}" in
    painter=*) PAINTER="${arg#painter=}" ;;
    thinker=*) THINKER="${arg#thinker=}" ;;
    sample=*)  SAMPLE="${arg#sample=}" ;;
    *) echo "Unknown arg '${arg}'. Only painter=<bool> thinker=<bool> sample=<bool> are accepted." >&2; exit 1 ;;
  esac
done
is_true() { case "${1,,}" in true|1|yes|y|on) return 0 ;; *) return 1 ;; esac; }

PAINTER_STEPS=${PAINTER_STEPS:-40000}
THINKER_STEPS=${THINKER_STEPS:-40000}
PAINTER_MAX_SECONDS=${PAINTER_MAX_SECONDS:-0}
THINKER_MAX_SECONDS=${THINKER_MAX_SECONDS:-0}
WANDB_PROJECT="${WANDB_PROJECT:-amaze}"
RUN_NAME="${RUN_NAME:-${TASK}_all${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/${RUN_NAME}_painter/checkpoint_final.pt}"
THINKER_CKPT="${THINKER_CKPT:-runs/${RUN_NAME}_thinker/checkpoint_final.pt}"

if [[ "${TASK}" == "maze" ]]; then
  EVAL_CB=amaze; GEN_ARGS=(--shape all --size all)
else
  EVAL_CB=amaze_queens; GEN_ARGS=(--size all)
fi

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

DATA_DIR="${PROJECT_ROOT}/data/amaze/train_${TASK}/all_train_size144"

if is_true "${PAINTER}" || is_true "${THINKER}"; then
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py train "${TASK}" "${GEN_ARGS[@]}"
fi

# Stage 1: unconditional painter
if is_true "${PAINTER}"; then
  srun python train_trm.py experiment=amaze_unet_painter \
    data.amaze_root="${DATA_DIR}" \
    eval_callbacks="${EVAL_CB}" \
    train.num_steps=${PAINTER_STEPS} \
    train.max_seconds=${PAINTER_MAX_SECONDS} \
    run.wandb_project="${WANDB_PROJECT}" \
    run.output_dir="runs/${RUN_NAME}_painter"
  echo "${TASK} painter complete -> ${PAINTER_CKPT}"
else
  echo "Skipping painter (painter=${PAINTER})."
fi

# Stage 2: thinker on the frozen painter (cell/seq/grid come from the config)
if is_true "${THINKER}"; then
  if [[ ! -f "${PAINTER_CKPT}" ]]; then
    echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2
    echo "Run with painter=true, or point PAINTER_CKPT at an existing painter." >&2
    exit 1
  fi
  srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
    data.amaze_root="${DATA_DIR}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    eval_callbacks="${EVAL_CB}" \
    train.num_steps=${THINKER_STEPS} \
    train.max_seconds=${THINKER_MAX_SECONDS} \
    run.wandb_project="${WANDB_PROJECT}" \
    run.output_dir="runs/${RUN_NAME}_thinker"
  echo "${TASK} thinker complete -> ${THINKER_CKPT}"
else
  echo "Skipping thinker (thinker=${THINKER})."
fi

# Stage 3: paper metrics, logged into the thinker's wandb run
if is_true "${SAMPLE}"; then
  if [[ ! -f "${PAINTER_CKPT}" ]]; then
    echo "ERROR: painter checkpoint not found for eval: ${PAINTER_CKPT}" >&2; exit 1
  fi
  if [[ ! -f "${THINKER_CKPT}" ]]; then
    echo "ERROR: thinker checkpoint not found for eval: ${THINKER_CKPT}" >&2; exit 1
  fi
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py test "${TASK}"
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_thinker_v2_controlnet \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="${THINKER_CKPT}" \
    +task="${TASK}" \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed -- checkpoints are safe in runs/${RUN_NAME}_thinker."
  echo "Metrics (${TASK}) done -- logged into wandb run ${RUN_NAME}_thinker."
else
  echo "Skipping evaluation (sample=${SAMPLE})."
fi
