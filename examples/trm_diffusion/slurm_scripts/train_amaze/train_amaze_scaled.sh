#!/bin/bash -l
#SBATCH --job-name=amaze_scaled
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

# Painter -> thinker on SCALED/TRANSLATED Amaze targets (data=amaze_scaled): the solved
# puzzle is pasted at a random 80-90% scale and random offset on a black canvas while the
# condition stays untransformed (the Amaze counterpart of the scaled-Sudoku test).
# Generations are cropped back to their content box before scoring.
#
#   PIPELINE=latent  frozen SD VAE + latent painter + cross-attention thinker (288px)
#   PIPELINE=pixel   the old pixel pipeline as baseline: amaze_unet_painter +
#                    amaze_thinker_v2_controlnet (concat encoder, 144px)
#
# Stages (each <stage>=<bool>, default true): data ceiling painter thinker sample
#   data     latent only: derive maze_dataset_train_288.parquet
#   ceiling  crop-back check (transformed GT -> crop -> score) + VAE ceiling + latent stats
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/train_amaze/train_amaze_scaled.sh <queens> <latent|pixel> [painter=false ...]
# Env: PAINTER_STEPS/THINKER_STEPS (40000), SAMPLES (Pass@K, 5), WANDB_PROJECT (PaTh-rebuttal, group latent),
#      RUN_NAME, VAE_PATH, STATS_JSON, PAINTER_CKPT/THINKER_CKPT, THINKER_BS/THINKER_ACCUM,
#      VENV + MODULES (space-separated) on a cluster other than Helios/Athena.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TASK="${1:?usage: train_amaze_scaled.sh <queens> <latent|pixel> [stage=bool ...]}"
PIPELINE="${2:?usage: train_amaze_scaled.sh <queens> <latent|pixel> [stage=bool ...]}"
shift 2
# Mazes have a black background, so a black canvas would make the crop-back ambiguous;
# they need their own canvas colour + eval-callback config before they can run here.
[[ "${TASK}" == "queens" ]] || { echo "Only TASK=queens is set up (maze needs a non-black canvas)" >&2; exit 1; }
[[ "${PIPELINE}" == "latent" || "${PIPELINE}" == "pixel" ]] || { echo "PIPELINE must be latent|pixel" >&2; exit 1; }

DATA=true; CEILING=true; PAINTER=true; THINKER=true; SAMPLE=true
for arg in "$@"; do
  case "${arg}" in
    data=*) DATA="${arg#data=}" ;; ceiling=*) CEILING="${arg#ceiling=}" ;;
    painter=*) PAINTER="${arg#painter=}" ;; thinker=*) THINKER="${arg#thinker=}" ;;
    sample=*) SAMPLE="${arg#sample=}" ;;
    *) echo "Unknown arg '${arg}'" >&2; exit 1 ;;
  esac
done
is_true() { case "${1,,}" in true|1|yes|y|on) return 0 ;; *) return 1 ;; esac; }

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg HF_HUB_OFFLINE=1
echo ">> cluster=${CLUSTER} host=$(hostname) job=${SLURM_JOB_ID:-none} task=${TASK} pipeline=${PIPELINE}"

PAINTER_STEPS=${PAINTER_STEPS:-40000}
THINKER_STEPS=${THINKER_STEPS:-40000}
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
RUN_NAME="${RUN_NAME:-scaled_${PIPELINE}_${TASK}}"   # fixed, so stages can run as separate jobs
VAE_PATH="${VAE_PATH:-${CLUSTER_MODELS_DIR}/sd-vae-ft-mse}"
STATS_JSON="${STATS_JSON:-runs/${RUN_NAME}_ceiling.json}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/${RUN_NAME}_painter/checkpoint_final.pt}"
THINKER_CKPT="${THINKER_CKPT:-runs/${RUN_NAME}_thinker/checkpoint_final.pt}"
DATA_ROOT="${PROJECT_ROOT}/data/amaze"
DATA_DIR="${DATA_ROOT}/${TASK}"
EVAL_CB=amaze_queens_scaled

if [[ "${PIPELINE}" == "latent" ]]; then
  SIZE=288; PAINTER_EXP=amaze_latent_unet_painter; THINKER_EXP=amaze_latent_thinker_xattn
else
  SIZE=144; PAINTER_EXP=amaze_unet_painter; THINKER_EXP=amaze_thinker_v2_controlnet
fi

if [[ "${PIPELINE}" == "latent" ]] && is_true "${DATA}"; then
  AMAZE_OUT_ROOT="${DATA_ROOT}" python scripts/gen_amaze.py --task "${TASK}" --stage train \
    -o 'scaled_image_sizes=[144,288]'
fi

# Crop-back sanity (scaled_gt_cropped must be ~= resized_gt, else the scoring itself is
# lossy), VAE ceiling on scaled targets, and latent stats measured on scaled targets.
if is_true "${CEILING}"; then
  python experiments/amaze_vae_ceiling.py --task "${TASK}" --data_root "${DATA_ROOT}" \
    --vae "${VAE_PATH}" --sizes ${SIZE} --stats_size ${SIZE} --scaled --out "${STATS_JSON}"
fi

COMMON=( data=amaze_scaled data.amaze_root="${DATA_DIR}" eval_callbacks="${EVAL_CB}"
         run.wandb_project="${WANDB_PROJECT}" run.wandb_group=latent )
if [[ "${PIPELINE}" == "latent" ]]; then
  [[ -f "${STATS_JSON}" ]] || { echo "ERROR: latent stats not found: ${STATS_JSON} (run ceiling=true)" >&2; exit 1; }
  read -r SHIFT SCALE < <(python -c "
import json; d = json.load(open('${STATS_JSON}'))
f = lambda v: '[' + ','.join(repr(x) for x in v) + ']'
print(f(d['latent_shift']), f(d['latent_scale']))")
  echo ">> latent_shift=${SHIFT} latent_scale=${SCALE}"
  COMMON+=( vae.pretrained_model_name_or_path="${VAE_PATH}"
            "painter.latent_shift=${SHIFT}" "painter.latent_scale=${SCALE}" )
fi
THINKER_EXTRA=()
[[ -n "${THINKER_BS:-}" ]] && THINKER_EXTRA+=( train.batch_size=${THINKER_BS} )
[[ -n "${THINKER_ACCUM:-}" ]] && THINKER_EXTRA+=( train.gradient_accumulation_steps=${THINKER_ACCUM} )

if is_true "${PAINTER}"; then
  srun python train_trm.py experiment=${PAINTER_EXP} "${COMMON[@]}" \
    train.num_steps=${PAINTER_STEPS} run.output_dir="runs/${RUN_NAME}_painter" \
    run.wandb_job_type=train "run.wandb_tags=[latent,scaled,${PIPELINE},painter,${TASK},${CLUSTER}]"
  echo "painter complete -> ${PAINTER_CKPT}"
fi

if is_true "${THINKER}"; then
  [[ -f "${PAINTER_CKPT}" ]] || { echo "ERROR: painter checkpoint not found: ${PAINTER_CKPT}" >&2; exit 1; }
  srun python train_trm.py experiment=${THINKER_EXP} "${COMMON[@]}" "${THINKER_EXTRA[@]}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    train.num_steps=${THINKER_STEPS} run.output_dir="runs/${RUN_NAME}_thinker" \
    run.wandb_job_type=train "run.wandb_tags=[latent,scaled,${PIPELINE},thinker,${TASK},${CLUSTER}]"
  echo "thinker complete -> ${THINKER_CKPT}"
fi

if is_true "${SAMPLE}"; then
  [[ -f "${THINKER_CKPT}" ]] || { echo "ERROR: thinker checkpoint not found: ${THINKER_CKPT}" >&2; exit 1; }
  # Test sets on disk: queens n4..n10 (no OOD size staged), maze + size 10.
  export QUEEN_OOD_SCALES="${QUEEN_OOD_SCALES-}" MAZE_OOD_SCALES="${MAZE_OOD_SCALES-10}"
  srun python experiments/amaze_generate_and_calculate_metrics.py \
    experiment=${THINKER_EXP} "${COMMON[@]}" \
    painter.checkpoint="${PAINTER_CKPT}" \
    +checkpoint="${THINKER_CKPT}" +task="${TASK}" +data_root="${DATA_ROOT}" \
    +samples_per_puzzle="${SAMPLES:-5}" +crop_to_content=true \
    || echo "WARN: metrics eval failed -- checkpoints are safe in runs/${RUN_NAME}_thinker."
fi
echo ">> done (${RUN_NAME})"
