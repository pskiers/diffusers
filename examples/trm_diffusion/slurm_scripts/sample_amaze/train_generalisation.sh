#!/bin/bash -l
#SBATCH --job-name=amaze_generalisation
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=30:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# Train ONE thinker on a generalisation subset against the frozen final PaTh
# painter, then score it on the FULL maze test sets (all shapes x sizes + size 10)
# with the same scorer as the final results.
#
#   KIND=trm          the PaTh TRM thinker, exactly as in the original run
#                     (amaze_thinker_v2_controlnet, 40k steps, eval_callbacks=amaze)
#   KIND=linear_head  the control: same encoder/translator/painter, the TRM replaced
#                     by a per-token linear head (configs/thinker/amaze_linear_head.yaml)
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/sample_amaze/train_generalisation.sh <subset> <trm|linear_head> [train=<bool>] [eval=<bool>]
# Env: PAINTER_CKPT (runs/pt_maze_final_painter/checkpoint_final.pt), THINKER_STEPS (40000),
#      SAMPLES (5), MAZE_OOD_SCALES (10), WANDB_PROJECT (amaze_generalisation), RUN_NAME, GEN_ROOT,
#      WANDB_ENTITY (workspace to log into; default: the key's default workspace),
#      WANDB_KEY_FILE (file holding a wandb API key, chmod 600; used instead of ~/.netrc for THIS job only)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/slurm_scripts/sample_amaze/generalisation_subsets.sh"

SUBSET="${1:?usage: sbatch train_generalisation.sh <subset> <trm|linear_head> [train=] [eval=]}"
KIND="${2:?need kind: trm|linear_head}"
shift 2
subset_shapes "${SUBSET}" > /dev/null

TRAIN=true; EVAL=true
for arg in "$@"; do
  case "${arg}" in
    train=*) TRAIN="${arg#train=}" ;;
    eval=*)  EVAL="${arg#eval=}" ;;
    *) echo "Unknown arg '${arg}'. Only train=<bool> eval=<bool> are accepted." >&2; exit 1 ;;
  esac
done
is_true() { case "${1,,}" in true|1|yes|y|on) return 0 ;; *) return 1 ;; esac; }

case "${KIND}" in
  trm)         THINKER_ARGS=() ;;
  linear_head) THINKER_ARGS=( thinker=amaze_linear_head ) ;;
  *) echo "KIND must be trm|linear_head" >&2; exit 1 ;;
esac

PAINTER_CKPT="${PAINTER_CKPT:-runs/pt_maze_final_painter/checkpoint_final.pt}"
THINKER_STEPS="${THINKER_STEPS:-40000}"
SAMPLES="${SAMPLES:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_generalisation}"
RUN_NAME="${RUN_NAME:-gen_${SUBSET}_${KIND}}"
THINKER_CKPT="runs/${RUN_NAME}/checkpoint_final.pt"
TRAIN_PARQUET="$(subset_train_parquet "${SUBSET}")"
DATA_ROOT="${PROJECT_ROOT}/data/amaze"          # full test sets (val during training, final scoring)

# Cluster detection from the scratch filesystem (same as flops_analysis.sh). On Athena
# submit with: sbatch --account=plgdiffusion3-gpu-a100 --partition=plgrid-gpu-a100 ...
# (or slurm_scripts/submit.sh, which swaps them by hostname).
case "${SCRATCH:-}" in
  /net/tscratch/*)   # Athena (x86_64, A100)
    CLUSTER=athena
    VENV="${VENV:-${SCRATCH}/trm_athena_venv}"
    module load "${PY_MODULE:-Python/3.10.4}" "${CUDA_MODULE:-CUDA/12.4.0}" "${CUDNN_MODULE:-cuDNN/9.2.1.18-CUDA-12.4.0}"
    # The A100s here have 40 GB. Training alone fits (~32 GB with expandable segments),
    # but after an in-training eval round the next training steps run out of memory
    # (tested: OOM right after the step-32 eval, at both eval batch 128 and 64). That
    # eval is monitoring only — it never feeds back into training — so it is switched
    # off here; the final evaluation below is a separate process and fits (~16 GB).
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    CLUSTER_TRAIN_ARGS=( eval.eval_every="${IN_TRAIN_EVAL_EVERY:-1000000000}" )
    ;;
  *)                 # Helios (aarch64, GH200)
    CLUSTER=helios
    VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
    module load "${PY_MODULE:-Python/3.11.5}" "${CUDA_MODULE:-CUDA/12.4.0}" "${CUDNN_MODULE:-cuDNN/9.2.1.18-CUDA-12.4.0}"
    export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
    CLUSTER_TRAIN_ARGS=()
    ;;
esac
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1
# Use the diffusers of THIS checkout (../../src), not whatever the venv's editable
# install points at — on Athena that is a different clone.
DIFFUSERS_SRC="$(cd "${PROJECT_ROOT}/../.." && pwd)/src"
[[ -d "${DIFFUSERS_SRC}/diffusers" ]] && export PYTHONPATH="${DIFFUSERS_SRC}:${PYTHONPATH:-}"
echo ">> cluster=${CLUSTER} host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(python -V 2>&1)" \
     "diffusers=$(python -c 'import diffusers; print(diffusers.__file__)')"

[[ -f "${PAINTER_CKPT}" ]] || { echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2; exit 1; }

# wandb account for this job only. The key is read from a file inside the job (never
# passed on the command line, where it would show up in the job record); wandb prefers
# WANDB_API_KEY over ~/.netrc, so other jobs keep using ~/.netrc.
if [[ -n "${WANDB_KEY_FILE:-}" ]]; then
  [[ -r "${WANDB_KEY_FILE}" ]] || { echo "ERROR: WANDB_KEY_FILE not readable: ${WANDB_KEY_FILE}" >&2; exit 1; }
  WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
  export WANDB_API_KEY
fi
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY
echo ">> wandb: project ${WANDB_PROJECT}, workspace ${WANDB_ENTITY:-<key default>}, key from ${WANDB_KEY_FILE:-~/.netrc}"

# Shared by training and scoring, so the two cannot drift apart.
MODEL_ARGS=( experiment=amaze_thinker_v2_controlnet ${THINKER_ARGS[@]+"${THINKER_ARGS[@]}"}
             data.amaze_root="${DATA_ROOT}/maze"
             painter.checkpoint="${PAINTER_CKPT}" )

echo ">> ${RUN_NAME}: subset ${SUBSET}, thinker ${KIND}, painter ${PAINTER_CKPT}"

if is_true "${TRAIN}"; then
  [[ -f "${TRAIN_PARQUET}" ]] || {
    echo "ERROR: subset not generated: ${TRAIN_PARQUET}" >&2
    echo "Run: sbatch slurm_scripts/sample_amaze/gen_generalisation_data.sh ${SUBSET}" >&2
    exit 1
  }
  srun python train_trm.py "${MODEL_ARGS[@]}" \
    data.train_dataset.dataset_path="${TRAIN_PARQUET}" \
    eval_callbacks=amaze \
    train.num_steps="${THINKER_STEPS}" \
    train.max_seconds=0 \
    run.wandb_project="${WANDB_PROJECT}" \
    run.output_dir="runs/${RUN_NAME}" \
    ${CLUSTER_TRAIN_ARGS[@]+"${CLUSTER_TRAIN_ARGS[@]}"}
  echo "${RUN_NAME} training complete -> ${THINKER_CKPT}"
else
  echo "Skipping training (train=${TRAIN})."
fi

if is_true "${EVAL}"; then
  [[ -f "${THINKER_CKPT}" ]] || { echo "ERROR: thinker checkpoint not found: ${THINKER_CKPT}" >&2; exit 1; }
  # Same scorer and test sets as final_results/maze/pt/metrics.json; size 10 (never
  # trained on by any model) is scored as the OOD table.
  export MAZE_OOD_SCALES="${MAZE_OOD_SCALES-10}"
  srun python experiments/amaze_generate_and_calculate_metrics.py "${MODEL_ARGS[@]}" \
    +checkpoint="${THINKER_CKPT}" \
    +task=maze \
    +data_root="${DATA_ROOT}" \
    +samples_per_puzzle="${SAMPLES}" \
    run.wandb_project="${WANDB_PROJECT}"
  echo "${RUN_NAME} metrics -> runs/${RUN_NAME}/amaze_metrics_maze.json"
else
  echo "Skipping evaluation (eval=${EVAL})."
fi
