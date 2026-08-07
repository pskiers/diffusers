#!/bin/bash -l
#SBATCH --job-name=amaze_maze_thinker
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

# Maze THINKER driver (TRM + frozen painter, then Stage-3 metrics for all 4 geometries).
#   square/hexagon 8  -> grid 8  (cell18/seq64). Single-shape 8x8: that shape is the
#                        in-distribution row; the other 3 geometries are OOD.
#   all     all       -> grid 12 (cell12/seq144). Every geometry in-distribution.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_maze_thinker.sh <shape|all> <N|all> <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]
# Env: THINKER_STEPS (default 40000), SAMPLES (Pass@K, default 5), RUN_METRICS (default 1).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="/net/tscratch/people/plgmgrzanka/trm_sokoban/venv"

SHAPE="${1:?usage: sbatch train_maze_thinker.sh <shape|all> <N|all> <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]}"
SIZE="${2:?usage: sbatch train_maze_thinker.sh <shape|all> <N|all> <PAINTER_CKPT> [WANDB_PROJECT] [RUN_NAME]}"
PAINTER_CKPT="${3:?need a frozen painter checkpoint as arg 3}"
THINKER_STEPS=${THINKER_STEPS:-40000}
THINKER_MAX_SECONDS=${THINKER_MAX_SECONDS:-0}
WANDB_PROJECT="${WANDB_PROJECT:-${4:-amaze}}"

# Reasoning grid: multi-size 'all' -> 12x12=144; single size N -> NxN (cell=144/N).
if [[ "${SIZE}" == "all" ]]; then
  EXP=amaze_thinker_v2_controlnet; CELL=12; SEQ=144; GRID=12
elif [[ "${SIZE}" =~ ^[0-9]+$ ]]; then
  EXP=amaze_thinker_v1_controlnet; GRID="${SIZE}"; CELL=$((144 / SIZE)); SEQ=$((SIZE * SIZE))
else
  echo "SIZE must be an integer (e.g. 8) or 'all', got '${SIZE}'" >&2; exit 1
fi

if [[ "${SHAPE}" == "all" ]]; then
  DATA_SUB="train_maze/all_train_size144"; TAG="all"
elif [[ "${SIZE}" == "all" ]]; then
  DATA_SUB="train_maze/${SHAPE}/all_${SHAPE}_train_size144"; TAG="${SHAPE}_all"
else
  DATA_SUB="train_maze/${SHAPE}/n${SIZE}_${SHAPE}_train_size144"; TAG="${SHAPE}_n${SIZE}"
fi
RUN_NAME="${RUN_NAME:-${5:-maze_thinker_${TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}}}"

module load CUDA/12.4.0
module load GCCcore/14.3.0 nodejs/22.17.1
module load Miniconda3/23.3.1-0

source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1

if [[ ! -d third_party/amaze/mazes-generator/node_modules ]]; then
  echo "ERROR: third_party/amaze/mazes-generator/node_modules is missing (maze gen needs it)." >&2
  echo "Run ONCE on a login node: module load GCCcore/14.3.0 nodejs/22.17.1 && (cd third_party/amaze/mazes-generator && npm install)" >&2
  exit 1
fi
if [[ ! -f "${PAINTER_CKPT}" ]]; then
  echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2
  echo "Train one first: sbatch slurm_scripts/train_maze_painter.sh ${SHAPE} ${SIZE}" >&2
  exit 1
fi

# ── Data (idempotent) ────────────────────────────────────────────────────────
AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
  python scripts/gen_amaze.py train maze --shape "${SHAPE}" --size "${SIZE}"
DATA_DIR="${PROJECT_ROOT}/data/amaze/${DATA_SUB}"

# ── Stage 2: thinker with the frozen painter ─────────────────────────────────
srun python train_trm.py experiment=${EXP} \
  data.amaze_root="${DATA_DIR}" \
  painter.checkpoint="${PAINTER_CKPT}" \
  data.cell_size=${CELL} thinker.seq_len=${SEQ} translator.grid=${GRID} \
  eval_callbacks=amaze \
  train.num_steps=${THINKER_STEPS} \
  train.max_seconds=${THINKER_MAX_SECONDS} \
  run.wandb_project="${WANDB_PROJECT}" \
  run.output_dir="runs/${RUN_NAME}"

echo "Maze thinker (${SHAPE} ${SIZE}) complete → runs/${RUN_NAME}/checkpoint_final.pt"

# ── Stage 3: paper metrics (all 4 geometries × scales) ───────────────────────
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" \
    python scripts/gen_amaze.py test maze
  srun python experiments/sample_amaze_metrics.py \
    experiment=${EXP} \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="runs/${RUN_NAME}/checkpoint_final.pt" \
    +task=maze \
    +data_root="${PROJECT_ROOT}/data/amaze" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    data.cell_size=${CELL} thinker.seq_len=${SEQ} translator.grid=${GRID} \
    run.wandb_project="${WANDB_PROJECT}" \
    || echo "WARN: metrics eval failed — checkpoints are safe in runs/${RUN_NAME}."
  echo "Metrics (maze, 4 geometries × scales) done — logged into wandb run ${RUN_NAME}."
fi
