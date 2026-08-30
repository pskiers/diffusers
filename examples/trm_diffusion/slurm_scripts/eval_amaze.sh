#!/bin/bash -l
#SBATCH --job-name=amaze_eval
#SBATCH --account=plgdiffusion3-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Score any model on the AMAZE tables
#   bagel|janus|dummy : score_amaze_images.py
#   dit|trm           : sample_amaze_metrics.py
#
# Usage: sbatch slurm_scripts/eval_amaze.sh <dummy|bagel|janus|dit|trm> <maze|queens> [CKPT] [PAINTER_CKPT]
# Env: SAMPLES(5) WANDB_PROJECT RUN_NAME THINK(0);  bagel: BAGEL_MODEL_PATH;  trm: CELL_SIZE SEQ_LEN GRID

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_sokoban/venv}"

MODEL="${1:?usage: sbatch eval_amaze.sh <dummy|bagel|janus|dit|trm> <maze|queens> [CKPT] [PAINTER_CKPT]}"
TASK="${2:?need task: maze|queens}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }
SAMPLES="${SAMPLES:-5}"

# Athena: the venv (torch 2.4.1+cu124) bundles CUDA/cuDNN -> no Python/CUDA module needed.
# Set MODULES only if you actually need extra modules (e.g. MODULES="CUDA/12.4.0").
if [[ -n "${MODULES:-}" ]]; then module load ${MODULES}; fi
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

case "${MODEL}" in
  dummy|bagel|janus)
    WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
    THINK="${THINK:-0}"
    SUFFIX=""; [[ "${THINK}" == "1" ]] && SUFFIX="_cot"
    RUN_NAME="${RUN_NAME:-ft_${MODEL}_${TASK}}${SUFFIX}"
    GEN_DIR="${GEN_DIR:-runs/${RUN_NAME}/generated}"
    [[ "${MODEL}" == "janus" ]] && export PYTHONPATH="${PROJECT_ROOT}/third_party/amaze/sft/janus/Janus:${PYTHONPATH:-}"

    GEN_ARGS=( "${TASK}" --backend "${MODEL}" --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" )
    [[ "${THINK}" == "1" ]] && GEN_ARGS+=( --think )
    if [[ "${MODEL}" != "dummy" ]]; then
      CHECKPOINT="${CHECKPOINT:-${3:-}}"
      : "${CHECKPOINT:?set CHECKPOINT (or pass as arg 3) for ${MODEL}}"
      GEN_ARGS+=( --checkpoint "${CHECKPOINT}" )
    fi
    if [[ "${MODEL}" == "bagel" ]]; then
      : "${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH to the base BAGEL-7B-MoT snapshot dir}"
      GEN_ARGS+=( --bagel-model-path "${BAGEL_MODEL_PATH}" )
    fi

    python experiments/generate_amaze_ft.py "${GEN_ARGS[@]}"
    python experiments/score_amaze_images.py "${TASK}" \
      --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
      --wandb-project "${WANDB_PROJECT}" --run-name "${RUN_NAME}"
    ;;

  dit|trm)
    WANDB_PROJECT="${WANDB_PROJECT:-amaze}"
    CKPT="${CHECKPOINT:-${3:?need the checkpoint path as arg 3}}"
    AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" python scripts/gen_amaze.py test "${TASK}"

    ARGS=( +checkpoint="${CKPT}" +task="${TASK}" +data_root="${PROJECT_ROOT}/data/amaze"
           +samples_per_puzzle="${SAMPLES}" run.wandb_project="${WANDB_PROJECT}" )
    if [[ "${MODEL}" == "dit" ]]; then
      ARGS+=( experiment="amaze_dit_${TASK}" )
    else
      PAINTER_CKPT="${PAINTER_CKPT:-${4:?trm needs the painter checkpoint as arg 4}}"
      if [[ "${TASK}" == "queens" ]]; then CELL_SIZE="${CELL_SIZE:-20}"; SEQ_LEN="${SEQ_LEN:-49}"; GRID="${GRID:-7}"
      else CELL_SIZE="${CELL_SIZE:-18}"; SEQ_LEN="${SEQ_LEN:-64}"; GRID="${GRID:-8}"; fi
      ARGS+=( experiment=amaze_thinker_v1_controlnet painter.checkpoint="${PAINTER_CKPT}"
              data.cell_size="${CELL_SIZE}" thinker.seq_len="${SEQ_LEN}" translator.grid="${GRID}" )
    fi
    srun python experiments/sample_amaze_metrics.py "${ARGS[@]}"
    ;;

  *) echo "MODEL must be dummy|bagel|janus|dit|trm, got '${MODEL}'" >&2; exit 1 ;;
esac

echo "eval_amaze ${MODEL} ${TASK} done"
