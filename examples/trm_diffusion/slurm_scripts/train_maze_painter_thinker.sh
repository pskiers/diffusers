#!/bin/bash -l
#SBATCH --job-name=amaze_maze_painter_thinker
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Maze painter -> thinker pipeline on ALL shapes (square, hexagon, triangle,
# circle) x ALL sizes, grid 12. Stages are toggled by flags (each defaults to true):
#   painter=<bool>   train the unconditional UNet painter
#   thinker=<bool>   train the TRM thinker on the frozen painter
#   sample=<bool>    run the paper metrics (4 geometries x scales)
# The reasoning grid (cell=12 / seq=144 / grid=12) is baked into the
# amaze_thinker_v2_controlnet config and is NEVER passed on the command line.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_maze_painter_thinker.sh painter=true thinker=true sample=true
# Env: PAINTER_STEPS/THINKER_STEPS (default 40000), SAMPLES (Pass@K, default 5),
#      WANDB_PROJECT (default amaze), RUN_NAME, and PAINTER_CKPT/THINKER_CKPT to
#      point at existing checkpoints when a stage is skipped.
# NOTE: maze gen needs the node generator (run once on a login node:
#       cd third_party/amaze/mazes-generator && npm install).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

# ── Flags: painter / thinker / sample (each defaults to true) ────────────────
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
RUN_NAME="${RUN_NAME:-maze_all${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/${RUN_NAME}_painter/checkpoint_final.pt}"
THINKER_CKPT="${THINKER_CKPT:-runs/${RUN_NAME}_thinker/checkpoint_final.pt}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

# Maze generation (train or test) requires the node maze generator.
if is_true "${PAINTER}" || is_true "${THINKER}" || is_true "${SAMPLE}"; then
  if [[ ! -d third_party/amaze/mazes-generator/node_modules ]]; then
    echo "ERROR: third_party/amaze/mazes-generator/node_modules is missing (maze gen needs it)." >&2
    echo "Run ONCE on a login node: module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
    exit 1
  fi
fi

DATA_DIR="${PROJECT_ROOT}/data/amaze/train_maze/all_train_size144"

# ── Data (only when a training stage runs; gen_amaze skips what already exists) ─
if is_true "${PAINTER}" || is_true "${THINKER}"; then
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py train maze --shape all --size all
fi

# ── Stage 1: unconditional painter ───────────────────────────────────────────
if is_true "${PAINTER}"; then
  srun python train_trm.py experiment=amaze_unet_painter \
    data.amaze_root="${DATA_DIR}" \
    eval_callbacks=amaze \
    train.num_steps=${PAINTER_STEPS} \
    train.max_seconds=${PAINTER_MAX_SECONDS} \
    run.wandb_project="${WANDB_PROJECT}" \
    run.output_dir="runs/${RUN_NAME}_painter"
  echo "Maze painter complete → ${PAINTER_CKPT}"
else
  echo "Skipping painter (painter=${PAINTER})."
fi

# ── Stage 2: thinker on the frozen painter (cell/seq/grid come from the config) ─
if is_true "${THINKER}"; then
  if [[ ! -f "${PAINTER_CKPT}" ]]; then
    echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2
    echo "Run with painter=true, or point PAINTER_CKPT at an existing painter." >&2
    exit 1
  fi
  srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
    data.amaze_root="${DATA_DIR}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    eval_callbacks=amaze \
    train.num_steps=${THINKER_STEPS} \
    train.max_seconds=${THINKER_MAX_SECONDS} \
    run.wandb_project="${WANDB_PROJECT}" \
    run.output_dir="runs/${RUN_NAME}_thinker"
  echo "Maze thinker complete → ${THINKER_CKPT}"
else
  echo "Skipping thinker (thinker=${THINKER})."
fi

# ── Stage 3: paper metrics (4 geometries x scales), logged into the thinker's run ─
if is_true "${SAMPLE}"; then
  if [[ ! -f "${PAINTER_CKPT}" ]]; then
    echo "ERROR: painter checkpoint not found for eval: ${PAINTER_CKPT}" >&2; exit 1
  fi
  if [[ ! -f "${THINKER_CKPT}" ]]; then
    echo "ERROR: thinker checkpoint not found for eval: ${THINKER_CKPT}" >&2; exit 1
  fi
  # Canonical test set (idempotent; sample_amaze_metrics.py never generates data).
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py test maze
  srun python experiments/sample_amaze_metrics.py \
    experiment=amaze_thinker_v2_controlnet \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="${THINKER_CKPT}" \
    +task=maze \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoints are safe in runs/${RUN_NAME}_thinker."
  echo "Metrics (maze, 4 geometries × scales) done — logged into wandb run ${RUN_NAME}_thinker."
else
  echo "Skipping evaluation (sample=${SAMPLE})."
fi
