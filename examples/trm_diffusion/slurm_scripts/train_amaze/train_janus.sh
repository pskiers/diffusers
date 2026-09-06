#!/bin/bash -l
#SBATCH --job-name=janus_sft
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Janus-Pro-7B SFT on the AMAZE fine-tuning data, via the authors' own
# third_party/amaze/sft/janus/sft.py.
#
# NB this is the no-CoT arm: sft.py hardcodes an empty assistant turn and takes the
# loss over the 576 output-image VQ tokens only (gen_head). There is no text
# reasoning phase in either training or infer_janus.py.
#
# Usage: sbatch slurm_scripts/train_amaze/train_janus.sh <maze|queens|maze3_800|queens_n4_800>
#
# Prerequisites, all on a LOGIN node (compute nodes may have no internet):
#   python scripts/gen_amaze.py ft <task>     -> data/amaze/ft/<task>/maze_dataset_{train,test}.parquet
#   bash third_party/amaze/setup_ft_code.sh   -> clones sft/janus/Janus
#   HF_HOME="${SCRATCH}/.cache/huggingface" huggingface-cli download deepseek-ai/Janus-Pro-7B
#
# Env overrides: PROJECT_ROOT, VENV, AMAZE_ROOT, MODEL_PATH, WANDB_PROJECT, WANDB_MODE,
#                N_EPOCHS, LR, GRAD_ACCUM, BSZ, VAL_EVERY_STEPS, MAX_CKPTS
# Anything not overridden keeps sft.py's own default.

set -euo pipefail

TASK="${1:?usage: sbatch slurm_scripts/train_amaze/train_janus.sh <maze|queens|maze3_800|queens_n4_800>}"

: "${SCRATCH:?SCRATCH is not set - run this under sbatch on Helios, or export SCRATCH yourself}"

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/Janus-Pro-7B}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/amaze/ft/${TASK}}"

# ── Resolve the AMAZE checkout ─────────────────────────────────────────────────
# Two trees can be present and they are NOT interchangeable:
#   third_party/ear-amaze -> full upstream EaR clone; what this project trains against
#   third_party/amaze     -> the partial tree vendored by setup_ft_code.sh, whose
#                            sft/janus/sft.py is locally modified (validation + wandb)
# ear-amaze wins by default (eval_janus.sh has always used it). Override with AMAZE_ROOT.
if [[ -z "${AMAZE_ROOT:-}" ]]; then
    for _cand in "${PROJECT_ROOT}/third_party/ear-amaze" "${PROJECT_ROOT}/third_party/amaze"; do
        if [[ -f "${_cand}/sft/janus/sft.py" ]]; then AMAZE_ROOT="${_cand}"; break; fi
    done
fi
: "${AMAZE_ROOT:?no AMAZE checkout with sft/janus/sft.py under PROJECT_ROOT/third_party (tried ear-amaze, amaze). Set AMAZE_ROOT=, or run third_party/amaze/setup_ft_code.sh}"

if [[ -f "${PROJECT_ROOT}/third_party/ear-amaze/sft/janus/sft.py" \
   && -f "${PROJECT_ROOT}/third_party/amaze/sft/janus/sft.py" ]]; then
    echo "NOTE: both third_party/ear-amaze and third_party/amaze exist; using ${AMAZE_ROOT##*/}." >&2
    echo "      They ship different sft.py and data/maze_dataset.py. Set AMAZE_ROOT= to force the other." >&2
fi

SFT_DIR="${AMAZE_ROOT}/sft/janus"
SFT_PY="${SFT_DIR}/sft.py"

# ear-amaze ships pristine upstream code; third_party/amaze's copy is locally modified
# (validation + wandb). Passing a flag the target doesn't declare is an instant argparse
# "unrecognized arguments" abort, so probe the source before building the arg list.
sft_has() { grep -q -- "'$1'" "${SFT_PY}" || grep -q -- "\"$1\"" "${SFT_PY}"; }

export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/slurm_outputs"

# sft.py needs `janus` (from the cloned base repo) and `data.maze_dataset` (from the
# AMAZE checkout). PROJECT_ROOT is deliberately NOT on the path: it holds its own
# top-level data/, which would collide with AMAZE's data/ package.
export PYTHONPATH="${SFT_DIR}/Janus:${AMAZE_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_PROJECT WANDB_MODE

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

[[ -d "${VENV}" ]] || { echo "ERROR: venv not found: ${VENV}" >&2; exit 1; }
source "${VENV}/bin/activate"

# ── Preflight: fail in seconds with an actionable message, not 10 min into the job ──
[[ -d "${SFT_DIR}/Janus/janus" ]] || {
    echo "ERROR: Janus base repo missing at ${SFT_DIR}/Janus." >&2
    echo "       Run on a login node: bash ${AMAZE_ROOT}/setup_ft_code.sh" >&2; exit 1; }
[[ -f "${DATA_DIR}/maze_dataset_train.parquet" ]] || {
    echo "ERROR: ${DATA_DIR}/maze_dataset_train.parquet missing." >&2
    echo "       Run: python ${PROJECT_ROOT}/scripts/gen_amaze.py ft ${TASK}" >&2; exit 1; }
# The test parquet is the validation set - only required by the sft.py variants that
# actually have a validation loop (the locally-modified third_party/amaze copy).
if [[ ! -f "${DATA_DIR}/maze_dataset_test.parquet" ]]; then
    if sft_has --val_every_steps; then
        echo "ERROR: ${DATA_DIR}/maze_dataset_test.parquet missing, and this sft.py builds a" >&2
        echo "       validation set from it at startup." >&2
        echo "       Run: python ${PROJECT_ROOT}/scripts/gen_amaze.py ft ${TASK}" >&2; exit 1
    fi
    echo "NOTE: no maze_dataset_test.parquet; this sft.py has no validation loop anyway." >&2
fi
command -v accelerate >/dev/null || { echo "ERROR: 'accelerate' not on PATH in ${VENV}." >&2; exit 1; }
python - <<'PY' || { echo "ERROR: import preflight failed (see above)." >&2; exit 1; }
import sys, traceback
try:
    import torch, accelerate, wandb, transformers      # noqa: F401
    # Importing janus.models is what registers MultiModalityCausalLM with
    # AutoModelForCausalLM; without it from_pretrained can't resolve the config.
    from janus.models import VLChatProcessor           # noqa: F401
    from data.maze_dataset import MazeDataset          # noqa: F401
except Exception:
    traceback.print_exc()
    # janus.models -> attrdict, whose PyPI release does `from collections import Mapping`
    # and therefore cannot import on Python >= 3.10 (this job loads Python 3.11.5).
    if "attrdict" in traceback.format_exc() or "collections" in traceback.format_exc():
        print("\nHINT: 'attrdict' 2.0.1 does `from collections import Mapping` and cannot", file=sys.stderr)
        print("      import on Python 3.11. attrdict3 is the compatible fork, but BOTH", file=sys.stderr)
        print("      install the same 'attrdict' module, so whichever landed last wins.", file=sys.stderr)
        print("      Fix (on an aarch64 compute node - the venv can't run on the login node):", file=sys.stderr)
        print("        pip uninstall -y attrdict attrdict3 && pip install attrdict3", file=sys.stderr)
        print("      Offline fallback: python scripts/fix_attrdict.py \"$VIRTUAL_ENV\"", file=sys.stderr)
    sys.exit(1)
print(f"preflight ok: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"cuda_available={torch.cuda.is_available()}")
PY

# Model must already be on disk if the compute node has no outbound network.
if [[ ! -d "${MODEL_PATH}" && ! -d "${HF_HOME}/hub/models--${MODEL_PATH//\//--}" ]]; then
    echo "WARN: ${MODEL_PATH} is neither a local dir nor cached in ${HF_HOME}/hub." >&2
    echo "      If this node has no internet the run will fail. Pre-fetch on a login node:" >&2
    echo "      HF_HOME=${HF_HOME} huggingface-cli download ${MODEL_PATH}" >&2
fi

cd "${SFT_DIR}"

OUTPUT_DIR="${SFT_DIR}/outputs/${TASK}"
LOG_DIR="${SFT_DIR}/train_logs"
EXPERIMENT_NAME="janus_train_${TASK}"
RUN_NAME="${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Build the sft.py argument list against the flags THIS sft.py declares ──────
LAUNCH_ARGS=()
add_arg() {  # add_arg <flag> <value>  -- skip, with a note, if unsupported
    if sft_has "$1"; then
        LAUNCH_ARGS+=( "$1" "$2" )
    else
        echo "NOTE: $(basename "${AMAZE_ROOT}")'s sft.py has no $1 -> not passing it." >&2
    fi
}

for _req in --model_path --data_path --output_dir; do
    sft_has "${_req}" || { echo "ERROR: ${SFT_PY} does not accept ${_req}; wrong sft.py?" >&2; exit 1; }
done
LAUNCH_ARGS+=( --model_path "${MODEL_PATH}" --data_path "${DATA_DIR}" --output_dir "${OUTPUT_DIR}" )

# These shape the checkpoint path that eval_janus.sh globs for.
add_arg --experiment_name "${EXPERIMENT_NAME}"
add_arg --run_name        "${RUN_NAME}"
add_arg --log_dir         "${LOG_DIR}"
# Present only in the locally-modified copy; harmless to skip on upstream.
add_arg --wandb_project   "${WANDB_PROJECT}"
add_arg --wandb_mode      "${WANDB_MODE}"

# Optional overrides: only when the matching env var is set.
[[ -n "${N_EPOCHS:-}"        ]] && add_arg --n_epochs                    "${N_EPOCHS}"
[[ -n "${LR:-}"              ]] && add_arg --learning_rate               "${LR}"
[[ -n "${GRAD_ACCUM:-}"      ]] && add_arg --gradient_accumulation_steps "${GRAD_ACCUM}"
[[ -n "${BSZ:-}"             ]] && add_arg --train_bsz_per_gpu           "${BSZ}"
[[ -n "${VAL_EVERY_STEPS:-}" ]] && add_arg --val_every_steps             "${VAL_EVERY_STEPS}"
[[ -n "${MAX_CKPTS:-}"       ]] && add_arg --max_ckpts                   "${MAX_CKPTS}"

if ! sft_has --val_every_steps; then
    echo "NOTE: this sft.py has no validation loop -> no val/loss curve, and it hardcodes" >&2
    echo "      wandb project=--experiment_name (${EXPERIMENT_NAME}) mode=online." >&2
    echo "      For validation + a configurable wandb project, re-run with:" >&2
    echo "        AMAZE_ROOT=${PROJECT_ROOT}/third_party/amaze sbatch ... ${TASK}" >&2
fi

echo "============================================="
echo "Janus SFT training (no-CoT arm)"
echo "Task       : ${TASK}"
echo "AMAZE root : ${AMAZE_ROOT}"
echo "Data       : ${DATA_DIR}"
echo "Model      : ${MODEL_PATH}"
echo "Checkpoints: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
if sft_has --wandb_project; then
    echo "wandb      : ${WANDB_PROJECT} (mode=${WANDB_MODE})"
else
    echo "wandb      : ${EXPERIMENT_NAME} (hardcoded by this sft.py, mode=online)"
fi
echo "sft.py args: ${LAUNCH_ARGS[*]}"
echo "============================================="

# `accelerate launch`, exactly as in sft.py's own docstring. The five env vars
# below are the only addition, and they are needed: sft.py calls
# dist.get_world_size() with no is_initialized() guard in three places (train()'s
# num_training_steps, TrainingMetrics.__init__, get_metric()'s all_reduce). The
# authors ran this multi-GPU, where accelerate creates a process group; on ONE GPU
# it sets distributed_type=NO and creates none, so the first of those raises
# "Default process group has not been initialized" a few minutes in, right after
# the model loads. Setting the standard torch.distributed variables makes
# accelerate initialise a 1-rank group, so get_world_size() returns 1 and the
# authors' code runs unmodified. (`--multi_gpu` can't be used instead: accelerate
# rejects it below 2 processes.)
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# Keep --num_processes at 1: sft.py calls model.language_model.model(...) directly,
# bypassing DDP's forward hook, so >1 rank would silently skip gradient sync.
# sft.py builds its own Accelerator(mixed_precision='bf16').
srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    sft.py \
    "${LAUNCH_ARGS[@]}"

echo "Finished -> ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
echo "Score it with: sbatch slurm_scripts/sample_amaze/eval_janus.sh ${TASK} <shape-if-maze>"
