#!/bin/bash -l
#SBATCH --job-name=amaze_latent_pt
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# LATENT painter -> thinker pipeline (experiments amaze_latent_unet_painter +
# amaze_latent_thinker_xattn): frozen pretrained SD VAE, latent-diffusion painter,
# cross-attention thinker input, ControlNet steering on the latent grid.
#
# Stages (each toggled by <stage>=<bool>, all default true):
#   data     derive maze_dataset_train_288.parquet (gen_amaze.py, from the original parquet)
#   ceiling  VAE reconstruction ceiling on the test set + per-channel latent stats (JSON)
#   painter  stage 1: unconditional latent painter
#   thinker  stage 2: thinker steering the frozen latent painter
#   sample   paper metrics (Pass@K) on the test set
#
# Usage (from the trm_diffusion dir; works on Helios and Athena — cluster from $SCRATCH):
#   sbatch slurm_scripts/train_amaze/train_latent_painter_thinker.sh <maze|queens> [painter=false ...]
#   (Athena: add --account=plgdiffusion3-gpu-a100 --partition=plgrid-gpu-a100)
# Env: PAINTER_STEPS/THINKER_STEPS (40000), SAMPLES (Pass@K, 5), WANDB_PROJECT (PaTh-rebuttal, group latent),
#      RUN_NAME, VAE_PATH, STATS_JSON, PAINTER_CKPT/THINKER_CKPT to reuse earlier stages.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TASK="${1:?usage: train_latent_painter_thinker.sh <maze|queens> [data= ceiling= painter= thinker= sample=]}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }
shift

DATA=true; CEILING=true; PAINTER=true; THINKER=true; SAMPLE=true
for arg in "$@"; do
  case "${arg}" in
    data=*)    DATA="${arg#data=}" ;;
    ceiling=*) CEILING="${arg#ceiling=}" ;;
    painter=*) PAINTER="${arg#painter=}" ;;
    thinker=*) THINKER="${arg#thinker=}" ;;
    sample=*)  SAMPLE="${arg#sample=}" ;;
    *) echo "Unknown arg '${arg}'" >&2; exit 1 ;;
  esac
done
is_true() { case "${1,,}" in true|1|yes|y|on) return 0 ;; *) return 1 ;; esac; }

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg HF_HUB_OFFLINE=1
echo ">> cluster=${CLUSTER} host=$(hostname) job=${SLURM_JOB_ID:-none} task=${TASK} python=$(python -V 2>&1)"

PAINTER_STEPS=${PAINTER_STEPS:-40000}
THINKER_STEPS=${THINKER_STEPS:-40000}
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
RUN_NAME="${RUN_NAME:-latent_${TASK}}"   # fixed, so the painter and thinker stages can run as separate jobs
VAE_PATH="${VAE_PATH:-${CLUSTER_MODELS_DIR}/sd-vae-ft-mse}"   # local copy: compute nodes are offline
STATS_JSON="${STATS_JSON:-runs/${RUN_NAME}_vae_ceiling.json}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/${RUN_NAME}_painter/checkpoint_final.pt}"
THINKER_CKPT="${THINKER_CKPT:-runs/${RUN_NAME}_thinker/checkpoint_final.pt}"
DATA_ROOT="${PROJECT_ROOT}/data/amaze"
DATA_DIR="${DATA_ROOT}/${TASK}"
if [[ "${TASK}" == "maze" ]]; then EVAL_CB=amaze; else EVAL_CB=amaze_queens; fi

# Stage 0a: 288px train parquet, derived from maze_dataset_train_original.parquet.
if is_true "${DATA}"; then
  AMAZE_OUT_ROOT="${DATA_ROOT}" python scripts/gen_amaze.py --task "${TASK}" --stage train \
    -o 'scaled_image_sizes=[144,288]'
fi

# Stage 0b: VAE ceiling + latent statistics (shift/scale for the painter).
if is_true "${CEILING}"; then
  python experiments/amaze_vae_ceiling.py --task "${TASK}" --data_root "${DATA_ROOT}" \
    --vae "${VAE_PATH}" --sizes 144 288 --stats_size 288 --out "${STATS_JSON}"
fi
[[ -f "${STATS_JSON}" ]] || { echo "ERROR: latent stats not found: ${STATS_JSON} (run ceiling=true)" >&2; exit 1; }
read -r SHIFT SCALE < <(python -c "
import json; d = json.load(open('${STATS_JSON}'))
f = lambda v: '[' + ','.join(repr(x) for x in v) + ']'
print(f(d['latent_shift']), f(d['latent_scale']))")
echo ">> latent_shift=${SHIFT} latent_scale=${SCALE}"

COMMON=( data.amaze_root="${DATA_DIR}" eval_callbacks="${EVAL_CB}"
         vae.pretrained_model_name_or_path="${VAE_PATH}"
         "painter.latent_shift=${SHIFT}" "painter.latent_scale=${SCALE}"
         run.wandb_project="${WANDB_PROJECT}" run.wandb_group=latent )

# Stage 1: unconditional latent painter
if is_true "${PAINTER}"; then
  srun python train_trm.py experiment=amaze_latent_unet_painter "${COMMON[@]}" \
    train.num_steps=${PAINTER_STEPS} \
    run.wandb_job_type=train "run.wandb_tags=[latent,painter,${TASK},${CLUSTER}]" \
    run.output_dir="runs/${RUN_NAME}_painter"
  echo "${TASK} latent painter complete -> ${PAINTER_CKPT}"
fi

# Stage 2: thinker on the frozen latent painter
if is_true "${THINKER}"; then
  [[ -f "${PAINTER_CKPT}" ]] || { echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2; exit 1; }
  srun python train_trm.py experiment=amaze_latent_thinker_xattn "${COMMON[@]}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    train.num_steps=${THINKER_STEPS} \
    run.wandb_job_type=train "run.wandb_tags=[latent,thinker,${TASK},${CLUSTER}]" \
    run.output_dir="runs/${RUN_NAME}_thinker"
  echo "${TASK} latent thinker complete -> ${THINKER_CKPT}"
fi

# Stage 3: paper metrics, logged into the thinker's wandb run
if is_true "${SAMPLE}"; then
  [[ -f "${THINKER_CKPT}" ]] || { echo "ERROR: thinker checkpoint not found: ${THINKER_CKPT}" >&2; exit 1; }
  AMAZE_OUT_ROOT="${DATA_ROOT}" python scripts/gen_amaze.py --task "${TASK}" --stage test
  # Same test sets and OOD size (10) as the generalisation runs.
  export MAZE_OOD_SCALES="${MAZE_OOD_SCALES-10}"
  srun python experiments/amaze_generate_and_calculate_metrics.py \
    experiment=amaze_latent_thinker_xattn "${COMMON[@]}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="${THINKER_CKPT}" \
    +task="${TASK}" \
    +data_root="${DATA_ROOT}" \
    +samples_per_puzzle="${SAMPLES:-5}" \
    || echo "WARN: metrics eval failed -- checkpoints are safe in runs/${RUN_NAME}_thinker."
fi
echo ">> done (${RUN_NAME})"
