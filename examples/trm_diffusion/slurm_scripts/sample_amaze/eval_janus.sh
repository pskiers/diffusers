#!/bin/bash -l
#SBATCH --job-name=infer_score_janus
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

if [[ -z "${EAR_AMAZE_ROOT:-}" ]]; then
    for _cand in "${PROJECT_ROOT}/third_party/ear_amaze"; do
        if [[ -f "${_cand}/infer/infer_janus.py" ]]; then EAR_AMAZE_ROOT="${_cand}"; break; fi
    done
fi
: "${EAR_AMAZE_ROOT:?no AMAZE checkout with infer/infer_janus.py under PROJECT_ROOT/third_party (tried ear_amaze). Set EAR_AMAZE_ROOT=, or run third_party/ear_amaze/setup_ft_code.sh}"

# Override DATA_PATH to sample a subset (e.g. a single size) instead of the full
# ft/ test split; it just needs a directory holding maze_dataset_test.parquet.
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/amaze/ft/${TASK}}"

# Janus/Bagel inference reads the MERGED ft/ test split, which holds in-distribution
# sizes only — OOD images are never generated. So OOD scoring defaults to OFF here;
# set MAZE_OOD_SCALES=10 / QUEEN_OOD_SCALES=12 explicitly only if you staged OOD data.
export MAZE_OOD_SCALES="${MAZE_OOD_SCALES-}"
export QUEEN_OOD_SCALES="${QUEEN_OOD_SCALES-}"

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${PROJECT_ROOT}/slurm_outputs"

module load "${PY_MODULE:-Python/3.11.5}" "${CUDA_MODULE:-CUDA/12.4.0}" "${CUDNN_MODULE:-cuDNN/9.2.1.18-CUDA-12.4.0}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"
cd "${EAR_AMAZE_ROOT}"

# infer_janus.py lives at repo root's infer/, and needs both the Janus package
# and the repo's top-level data/ (shared MazeDataset loader) importable.
export PYTHONPATH="${EAR_AMAZE_ROOT}/sft/janus/Janus:${EAR_AMAZE_ROOT}:${PYTHONPATH:-}"

# Maze: shape gets its own output dir (flat filenames don't encode shape for
# non-circle mazes). Queens: one flat dir for the whole run (scale recovered by id).
# Queens filenames are identical across models (same test set, same ids), so two
# runs sharing a GEN_DIR silently overwrite each other's images AND their metrics
# json. Override GEN_DIR when comparing two checkpoints of the same task.
# THINK=1 (CoT) writes to <task>_cot/... so it can never overwrite the plain
# generations that the non-CoT tables were scored from. An explicit GEN_DIR
# still wins, as before.
COT_SUFFIX=""
[ "${THINK:-0}" = "1" ] && COT_SUFFIX="_cot"
if [ -z "${GEN_DIR:-}" ]; then
    if [ "${KIND}" = "maze" ]; then
        GEN_DIR="${EAR_AMAZE_ROOT}/inference_results/${TASK}${COT_SUFFIX}/${SHAPE}"
    else
        GEN_DIR="${EAR_AMAZE_ROOT}/inference_results/${TASK}${COT_SUFFIX}"
    fi
fi
mkdir -p "${GEN_DIR}"
if [ -n "$(find "${GEN_DIR}" -maxdepth 1 -name '*_attempt*' -print -quit 2>/dev/null)" ]; then
    echo "WARNING: ${GEN_DIR} already holds generated images — they will be overwritten." >&2
    echo "         Set GEN_DIR= to keep runs separate." >&2
fi

if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
    CHECKPOINT_PATH="${CHECKPOINT_OVERRIDE}"
else
    # sft.py writes <task>/janus_train_<task>/<run_name>/checkpoint-<epoch>-<step>/tfmr,
    # and RUN_NAME is timestamped, so there is one run dir per training attempt. Take the
    # most recently WRITTEN checkpoint that actually holds weights — note that is the last
    # one, not the best by val loss. Pass a checkpoint explicitly to pin a specific one.
    CKPT_ROOT="${EAR_AMAZE_ROOT}/sft/janus/outputs/${TASK}/janus_train_${TASK}"
    LATEST_CKPT_DIR=""
    while IFS= read -r _cand; do
        if [ -d "${_cand}/tfmr" ]; then LATEST_CKPT_DIR="${_cand}"; break; fi
    done < <(find "${CKPT_ROOT}" -mindepth 1 -maxdepth 3 -type d -name "checkpoint-*" \
                -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
    if [ -z "${LATEST_CKPT_DIR}" ]; then
        echo "No checkpoint-*/tfmr found under ${CKPT_ROOT}" >&2
        echo "Candidates seen (newest first):" >&2
        find "${CKPT_ROOT}" -mindepth 1 -maxdepth 3 -type d -name "checkpoint-*" \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2- | head -5 >&2
        exit 1
    fi
    echo "Auto-selected checkpoint (newest written, NOT best-by-val): ${LATEST_CKPT_DIR}"
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
    --batch_size "${BATCH:-8}"
    --temperature "${TEMPERATURE:-1.0}"
    --num_attempts "${ATTEMPTS:-5}"
)
# Maze filters to one shape; queens has no shape filter (generate every board).
# optional: restrict the run for quick controlled comparisons
[ -n "${SAMPLES_PER_SIZE:-}" ] && INFER_ARGS+=( --samples_per_size "${SAMPLES_PER_SIZE}" )
[ "${KIND}" = "maze" ] && INFER_ARGS+=( --filter_shape "${SHAPE}" )
srun python "${EAR_AMAZE_ROOT}/infer/infer_janus.py" "${INFER_ARGS[@]}"

echo "Generation finished. Scoring..."

# --- 2. Score with the adapted scorer (reads flat/id-keyed output above) ---
SCORE_ARGS=(
    "${KIND}"
    --gen-dir "${GEN_DIR}"
    --data-root "${PROJECT_ROOT}/data/amaze"
    --run-name "janus_${TASK}${SHAPE:+_${SHAPE}}${COT_SUFFIX}"
    --wandb-project amaze_final
)
# Maze needs the shape; queens scores all scales in one pass (no --geometry).
[ "${KIND}" = "maze" ] && SCORE_ARGS+=( --geometry "${SHAPE}" )
python "${PROJECT_ROOT}/experiments/amaze_score_generated_images.py" "${SCORE_ARGS[@]}"

echo "Done."
