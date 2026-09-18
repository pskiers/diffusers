#!/bin/bash -l
#SBATCH --job-name=amaze_score
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Score images that were ALREADY generated, without re-running inference. Use this
# after a scorer fix, or to redo a table with different scale settings.
#
# Usage:
#   maze:   sbatch slurm_scripts/sample_amaze/score_generated.sh maze <square|triangle|hexagon|circle>
#   queens: sbatch slurm_scripts/sample_amaze/score_generated.sh queens
#
# Env: GEN_DIR (default third_party/<amaze>/inference_results/<task>[/<shape>]),
#      SAMPLES (5), RUN_NAME, WANDB_PROJECT (amaze_final), IMAGE_SIZE,
#      MAZE_SCALES / MAZE_OOD_SCALES / QUEEN_SCALES / QUEEN_OOD_SCALES.
#
# NB: OOD defaults to EMPTY here on purpose. Janus/Bagel inference reads the merged
# ft/ test split, which contains in-distribution sizes only — no OOD images were ever
# generated, so asking for them would just produce a zero row.
set -euo pipefail

TASK="${1:?Add task (maze|queens)}"
case "${TASK}" in
    queens*) KIND="queens" ;;
    maze*)   KIND="maze" ;;
    *)       echo "TASK must be maze|queens" >&2; exit 1 ;;
esac
if [ "${KIND}" = "maze" ]; then
    SHAPE="${2:?maze needs a shape (square|triangle|hexagon|circle)}"
else
    SHAPE=""
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

if [[ -z "${EAR_AMAZE_ROOT:-}" ]]; then
    for _cand in "${PROJECT_ROOT}/third_party/ear_amaze"; do
        [[ -d "${_cand}/inference_results" ]] && { EAR_AMAZE_ROOT="${_cand}"; break; }
    done
fi
: "${EAR_AMAZE_ROOT:=${PROJECT_ROOT}/third_party/ear_amaze}"

if [ "${KIND}" = "maze" ]; then
    GEN_DIR="${GEN_DIR:-${EAR_AMAZE_ROOT}/inference_results/${TASK}/${SHAPE}}"
else
    GEN_DIR="${GEN_DIR:-${EAR_AMAZE_ROOT}/inference_results/${TASK}}"
fi
[[ -d "${GEN_DIR}" ]] || { echo "ERROR: gen dir not found: ${GEN_DIR}" >&2; exit 1; }
N_IMG=$(find "${GEN_DIR}" -maxdepth 1 -name '*_attempt*' \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
[[ "${N_IMG}" -gt 0 ]] || { echo "ERROR: no *_attempt* images in ${GEN_DIR}" >&2; exit 1; }

SAMPLES="${SAMPLES:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
RUN_NAME="${RUN_NAME:-janus_${TASK}${SHAPE:+_${SHAPE}}}"
export MAZE_OOD_SCALES="${MAZE_OOD_SCALES-}"
export QUEEN_OOD_SCALES="${QUEEN_OOD_SCALES-}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1

echo "============================================="
echo "Scoring only (no inference): task=${TASK} shape=${SHAPE:-<none>}"
echo "Images:   ${N_IMG} in ${GEN_DIR}"
echo "Scales:   MAZE_SCALES=${MAZE_SCALES:-<default>} MAZE_OOD_SCALES='${MAZE_OOD_SCALES}'"
echo "Run name: ${RUN_NAME}"
echo "============================================="

SCORE_ARGS=(
    "${KIND}"
    --gen-dir "${GEN_DIR}"
    --data-root "${PROJECT_ROOT}/data/amaze"
    --samples-per-puzzle "${SAMPLES}"
    --run-name "${RUN_NAME}"
    --wandb-project "${WANDB_PROJECT}"
)
[ "${KIND}" = "maze" ] && SCORE_ARGS+=( --geometry "${SHAPE}" )
[ -n "${IMAGE_SIZE:-}" ] && SCORE_ARGS+=( --image-size "${IMAGE_SIZE}" )

python experiments/amaze_score_generated_images.py "${SCORE_ARGS[@]}"
echo "Scoring (${TASK}${SHAPE:+/${SHAPE}}) done."

