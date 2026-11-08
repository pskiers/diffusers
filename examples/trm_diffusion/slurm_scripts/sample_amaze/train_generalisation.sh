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
#      SAMPLES (5), MAZE_OOD_SCALES (10), WANDB_PROJECT (PaTh-rebuttal), RUN_NAME, GEN_ROOT,
#      CLUSTER/VENV/MODULES, WANDB_ENTITY, WANDB_KEY_FILE (see slurm_scripts/cluster_env.sh).
# Submit through submit_generalisation.sh, which adds the cluster's account/partition/GPU type.
#
# wandb (project PaTh-rebuttal, group "generalisation"): one run per model, named
# gen_<subset>_<kind>; training curves and the final evaluation land in the SAME run.
# The final evaluation also logs gen/seen/* and gen/unseen/* (mean over the test cells
# inside vs outside the training subset) and one scalar per (shape, size) cell.
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
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
RUN_NAME="${RUN_NAME:-gen_${SUBSET}_${KIND}}"
THINKER_CKPT="runs/${RUN_NAME}/checkpoint_final.pt"
TRAIN_PARQUET="$(subset_train_parquet "${SUBSET}")"
DATA_ROOT="${PROJECT_ROOT}/data/amaze"          # full test sets (val during training, final scoring)

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
CLUSTER_TRAIN_ARGS=()
if [[ "${CLUSTER}" == athena ]]; then
  # The A100s here have 40 GB. Training alone fits (~32 GB with expandable segments),
  # but after an in-training eval round the next training steps run out of memory
  # (tested: OOM right after the step-32 eval, at both eval batch 128 and 64). That
  # eval is monitoring only — it never feeds back into training — so it is switched
  # off here; the final evaluation below is a separate process and fits (~16 GB).
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  CLUSTER_TRAIN_ARGS=( eval.eval_every="${IN_TRAIN_EVAL_EVERY:-1000000000}" )
fi
echo ">> diffusers=$(python -c 'import diffusers; print(diffusers.__file__)')"

[[ -f "${PAINTER_CKPT}" ]] || { echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2; exit 1; }

echo ">> wandb: ${WANDB_ENTITY}/${WANDB_PROJECT}, group generalisation, run ${RUN_NAME}"
SEEN_CELLS="$(subset_seen_cells "${SUBSET}")"

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
    run.wandb_group=generalisation \
    run.wandb_job_type=train \
    "run.wandb_tags=[generalisation,${SUBSET},${KIND},${CLUSTER}]" \
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
    "+seen_cells='${SEEN_CELLS}'" \
    run.wandb_project="${WANDB_PROJECT}"
  echo "${RUN_NAME} metrics -> runs/${RUN_NAME}/amaze_metrics_maze.json"
else
  echo "Skipping evaluation (eval=${EVAL})."
fi
