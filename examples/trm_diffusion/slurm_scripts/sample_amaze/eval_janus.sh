#!/bin/bash -l
#SBATCH --job-name=infer_score_janus
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Usage:
#   maze:   sbatch slurm_scripts/sample_amaze/eval_janus.sh <maze|maze3_800> <square|triangle|hexagon|circle> [checkpoint_path]
#   queens: sbatch slurm_scripts/sample_amaze/eval_janus.sh <queens|queens_n4_800> [checkpoint_path]
#
# Maze runs one shape per job (infer_janus.py flat filenames don't encode shape).
# Queens has no shapes: infer_janus.py names every board "0×0_<id>_attempt..." and
# the scorer separates scales by puzzle `id`, so one job covers all queens scales.
set -euo pipefail

TASK="${1:?Add task (maze|maze3_800|queens|queens_n4_800)}"
case "${TASK}" in
    queens*) KIND="queens" ;;
    *)       KIND="maze" ;;
esac

if [ "${KIND}" = "maze" ]; then
    SHAPE="${2:?maze needs a shape (square|triangle|hexagon|circle)}"
    CHECKPOINT_OVERRIDE="${3:-}"
else
    SHAPE=""
    CHECKPOINT_OVERRIDE="${2:-}"
fi

PROJECT_ROOT="/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion"
VENV="${SCRATCH}/trm_helios_venv"
EAR_AMAZE_ROOT="${PROJECT_ROOT}/third_party/ear-amaze"
DATA_PATH="${PROJECT_ROOT}/data/amaze/ft/${TASK}"

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${PROJECT_ROOT}/slurm_outputs"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"
cd "${EAR_AMAZE_ROOT}"

# infer_janus.py lives at repo root's infer/, and needs both the Janus package
# and the repo's top-level data/ (shared MazeDataset loader) importable.
export PYTHONPATH="${EAR_AMAZE_ROOT}/sft/janus/Janus:${EAR_AMAZE_ROOT}:${PYTHONPATH:-}"

# Maze: shape gets its own output dir (flat filenames don't encode shape for
# non-circle mazes). Queens: one flat dir for the whole run (scale recovered by id).
if [ "${KIND}" = "maze" ]; then
    GEN_DIR="${EAR_AMAZE_ROOT}/inference_results/${TASK}/${SHAPE}"
else
    GEN_DIR="${EAR_AMAZE_ROOT}/inference_results/${TASK}"
fi
mkdir -p "${GEN_DIR}"

if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
    CHECKPOINT_PATH="${CHECKPOINT_OVERRIDE}"
else
    CKPT_ROOT="${EAR_AMAZE_ROOT}/sft/janus/outputs/${TASK}/janus_train_${TASK}"
    LATEST_CKPT_DIR=$(find "${CKPT_ROOT}" -maxdepth 3 -type d -name "checkpoint-*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-)
    if [ -z "${LATEST_CKPT_DIR}" ]; then
        echo "No checkpoint found under ${CKPT_ROOT}" >&2
        exit 1
    fi
    CHECKPOINT_PATH="${LATEST_CKPT_DIR}/tfmr"
fi

echo "============================================="
echo "Janus inference: task=${TASK} kind=${KIND} shape=${SHAPE:-<none>}"
echo "Data: ${DATA_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${GEN_DIR}"
echo "============================================="

# --- 1. Generate with the authors' own infer_janus.py ---
INFER_ARGS=(
    --checkpoint_path "${CHECKPOINT_PATH}"
    --data_path "${DATA_PATH}"
    --split test
    --output_dir "${GEN_DIR}"
    --batch_size 8
    --temperature 1.0
    --num_attempts 5
)
# Maze filters to one shape; queens has no shape filter (generate every board).
[ "${KIND}" = "maze" ] && INFER_ARGS+=( --filter_shape "${SHAPE}" )
srun python "${EAR_AMAZE_ROOT}/infer/infer_janus.py" "${INFER_ARGS[@]}"

echo "Generation finished. Scoring..."

# --- 2. Score with the adapted scorer (reads flat/id-keyed output above) ---
SCORE_ARGS=(
    "${KIND}"
    --gen-dir "${GEN_DIR}"
    --data-root "${PROJECT_ROOT}/data/amaze"
    --run-name "janus_${TASK}${SHAPE:+_${SHAPE}}"
    --wandb-project amaze_final
)
# Maze needs the shape; queens scores all scales in one pass (no --geometry).
[ "${KIND}" = "maze" ] && SCORE_ARGS+=( --geometry "${SHAPE}" )
python "${PROJECT_ROOT}/experiments/score_amaze_images.py" "${SCORE_ARGS[@]}"

echo "Done."
