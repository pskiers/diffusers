#!/bin/bash -l
#SBATCH --job-name=infer_score_bagel
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

# Sample a fine-tuned Bagel with the AUTHORS' own infer/infer_bagel.py, then score
# the images with our AmazeMetrics (experiments/amaze_score_generated_images.py) —
# the same split of responsibilities as eval_janus.sh, so Bagel, Janus, DiT and TRM
# all land on one metric implementation.
#
# Usage:
#   maze:   sbatch slurm_scripts/sample_amaze/eval_bagel.sh maze <square|triangle|hexagon|circle> [checkpoint]
#   queens: sbatch slurm_scripts/sample_amaze/eval_bagel.sh queens [checkpoint]
#
# Maze runs one shape per job (infer_bagel.py filters with config.sample.filter_shape).
# Queens has no shapes: every board is named "0x0_<id>_attempt..." and the scorer
# separates scales by puzzle `id`, so one job covers all queens scales.
#
# Env: BAGEL_MODEL_PATH (REQUIRED, base snapshot), CHECKPOINT, CONFIG (config/maze.py),
#      ATTEMPTS (5), RESOLUTION (1024), BATCH (4), STEPS (50), LOGDIR, WANDB_PROJECT,
#      MAZE_OOD_SCALES / QUEEN_OOD_SCALES (must match the generated test sizes),
#      EAR_AMAZE_ROOT, VENV.
set -euo pipefail

TASK="${1:?Add task (maze|queens)}"
case "${TASK}" in
    queens*) KIND="queens" ;;
    maze*)   KIND="maze" ;;
    *)       echo "TASK must be maze|queens" >&2; exit 1 ;;
esac
if [ "${KIND}" = "maze" ]; then
    SHAPE="${2:?maze needs a shape (square|triangle|hexagon|circle)}"
    CHECKPOINT_OVERRIDE="${3:-${CHECKPOINT:-}}"
else
    SHAPE=""
    CHECKPOINT_OVERRIDE="${2:-${CHECKPOINT:-}}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

if [[ -z "${EAR_AMAZE_ROOT:-}" ]]; then
    for _cand in "${PROJECT_ROOT}/third_party/ear-amaze" "${PROJECT_ROOT}/third_party/amaze"; do
        if [[ -f "${_cand}/infer/infer_bagel.py" ]]; then EAR_AMAZE_ROOT="${_cand}"; break; fi
    done
fi
: "${EAR_AMAZE_ROOT:?no AMAZE checkout with infer/infer_bagel.py under PROJECT_ROOT/third_party. Set EAR_AMAZE_ROOT=, or run third_party/amaze/setup_ft_code.sh}"

# infer_bagel.py imports `dataset.maze_dataset` and is driven by an ml_collections
# config file; neither is needed by infer_janus.py, so an older checkout may lack them.
CONFIG="${CONFIG:-${EAR_AMAZE_ROOT}/config/maze.py}"
[[ -f "${EAR_AMAZE_ROOT}/dataset/maze_dataset.py" ]] || {
    echo "ERROR: ${EAR_AMAZE_ROOT}/dataset/maze_dataset.py missing — infer_bagel.py imports it." >&2
    echo "       Run: bash third_party/amaze/setup_ft_code.sh   (vendors dataset/ and config/)" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || {
    echo "ERROR: config file ${CONFIG} missing — infer_bagel.py is driven by ml_collections." >&2
    echo "       Run: bash third_party/amaze/setup_ft_code.sh, or set CONFIG=<path to a config .py>" >&2; exit 1; }

: "${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to a local BAGEL-7B-MoT snapshot (the base weights)}"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/amaze/ft/${TASK}}"
[[ -f "${DATA_PATH}/maze_dataset_test.parquet" ]] || {
    echo "ERROR: ${DATA_PATH}/maze_dataset_test.parquet missing." >&2
    echo "       Run: python scripts/gen_amaze.py --task ${TASK}   (ft_links: true symlinks data/amaze/ft/)" >&2; exit 1; }

ATTEMPTS="${ATTEMPTS:-5}"
RESOLUTION="${RESOLUTION:-1024}"
BATCH="${BATCH:-4}"
STEPS="${STEPS:-50}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/runs/bagel_infer}"
RUN_NAME="bagel_${TASK}${SHAPE:+_${SHAPE}}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"

# Resolve the fine-tuned checkpoint: explicit arg wins, else newest step dir.
if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
    CHECKPOINT_PATH="${CHECKPOINT_OVERRIDE}"
else
    # fsdp_save_ckpt writes <checkpoint_dir>/<7-digit step>/model.safetensors (+ optimizer
    # shards). Take the highest step that actually holds weights — the LAST checkpoint, not
    # the best by val. Pass a checkpoint explicitly to pin a specific one.
    CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/runs/ft_bagel_${TASK}/checkpoints}"
    CHECKPOINT_PATH=""
    while IFS= read -r _cand; do
        if [[ -f "${_cand}/model.safetensors" || -f "${_cand}/ema.safetensors" ]]; then
            CHECKPOINT_PATH="${_cand}"; break
        fi
    done < <(find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort -Vr)
    [[ -n "${CHECKPOINT_PATH}" ]] || {
        echo "ERROR: no checkpoint with model.safetensors under ${CKPT_ROOT}." >&2
        echo "Candidates seen (highest step first):" >&2
        find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort -Vr | head -5 >&2
        echo "Pass a checkpoint as the last argument or set CHECKPOINT=." >&2; exit 1; }
    echo "Auto-selected checkpoint (highest step, NOT best-by-val): ${CHECKPOINT_PATH}"
fi

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${PROJECT_ROOT}/slurm_outputs" "${LOGDIR}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"
cd "${EAR_AMAZE_ROOT}"

# infer_bagel.py lives at repo root's infer/ and imports `infer.bagel.*`,
# `dataset.maze_dataset` and the config module — all relative to the repo root.
export PYTHONPATH="${EAR_AMAZE_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Bagel inference: task=${TASK} kind=${KIND} shape=${SHAPE:-<none>}"
echo "Config:     ${CONFIG}"
echo "Data:       ${DATA_PATH}"
echo "Base model: ${BAGEL_MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Logdir:     ${LOGDIR}"
echo "============================================="

# --- 1. Generate with the authors' own infer_bagel.py (absl + ml_collections) ---
INFER_ARGS=(
    --config "${CONFIG}"
    --config.dataset="${DATA_PATH}"
    --config.dataset_split=test
    --config.pretrained.model="${BAGEL_MODEL_PATH}"
    --config.pretrained.checkpoint_path="${CHECKPOINT_PATH}"
    --config.logdir="${LOGDIR}"
    --config.run_name="${RUN_NAME}"
    --config.sample.num_attempts="${ATTEMPTS}"
    --config.sample.resolution="${RESOLUTION}"
    --config.sample.test_batch_size="${BATCH}"
    --config.sample.eval_num_steps="${STEPS}"
)
# Maze filters to one shape; circle boards are sized by layers, which the authors'
# code keys off is_circle. Queens generates every board (no shape filter).
if [ "${KIND}" = "maze" ]; then
    INFER_ARGS+=( --config.sample.filter_shape="${SHAPE}" )
    [ "${SHAPE}" = "circle" ] && INFER_ARGS+=( --config.is_circle=True )
fi

srun python "${EAR_AMAZE_ROOT}/infer/infer_bagel.py" "${INFER_ARGS[@]}"

# infer_bagel.py appends a timestamp to run_name and writes to
# <logdir>/<run_name>_<timestamp>_<eval_num_steps>/generated_images, so the exact
# directory is only known after the fact — take the newest match.
GEN_DIR=$(find "${LOGDIR}" -maxdepth 2 -type d -name generated_images -path "*${RUN_NAME}_*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-)
[[ -n "${GEN_DIR}" && -d "${GEN_DIR}" ]] || {
    echo "ERROR: no generated_images dir under ${LOGDIR} matching ${RUN_NAME}_* — did inference run?" >&2; exit 1; }

echo "Generation finished -> ${GEN_DIR}"
echo "Scoring..."

# --- 2. Score with OUR metrics (same scorer as Janus/DiT/TRM) ---
# The scorer accepts infer_bagel's "9x9_<id>_attempt001.jpg" as well as
# infer_janus's "9×9_<id>_attempt001.png".
SCORE_ARGS=(
    "${KIND}"
    --gen-dir "${GEN_DIR}"
    --data-root "${PROJECT_ROOT}/data/amaze"
    --samples-per-puzzle "${ATTEMPTS}"
    --run-name "bagel_${TASK}${SHAPE:+_${SHAPE}}"
    --wandb-project "${WANDB_PROJECT}"
)
[ "${KIND}" = "maze" ] && SCORE_ARGS+=( --geometry "${SHAPE}" )
python "${PROJECT_ROOT}/experiments/amaze_score_generated_images.py" "${SCORE_ARGS[@]}"

echo "Bagel eval (${TASK}${SHAPE:+/${SHAPE}}) done -> ${GEN_DIR}"
