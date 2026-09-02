#!/bin/bash -l
#SBATCH --job-name=amaze_cond_probe
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# Interpret the TRM's ControlNet conditioning (maze) with four tools, on Helios (GH200):
#   1. conditioning_heatmaps.py    — 12x12 energy + 11 raw-logit channel maps (good/bad)
#   2. conditioning_trajectory.py  — painter x0(t) alongside conditioning(t) (TRM; also DiT if given)
#   3. conditioning_swap.py         — inject maze-B conditioning onto maze-A (same n, diff shapes)
#   4. decode_conditioning.py       — linear probes: on_path/dist/wall/marker vs floors (+ failure attribution)
#
# Usage:
#   sbatch slurm_scripts/conditioning_probe.sh <thinker.pt> <painter.pt> [dit.pt]
# Env (defaults in parens):
#   COMBO(square_n7) N_EACH(10) SWAP_N(7) SWAP_A(square) SWAP_B(hexagon)
#   DECODE_N(200) DECODE_FRAC(0.9) DECODE_USE_ZH(false) STEPS(8)
#   WANDB_PROJECT(amaze) WANDB_RUN_ID('' -> uses <ckpt-dir>/wandb_run_id.txt) DATA_ROOT
#
# NOTE: attaches to the run recorded in <thinker.pt dir>/wandb_run_id.txt (DiT to <dit.pt dir>/...),
# or pass WANDB_RUN_ID to force one. Additive; does not touch any other job's files.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"

usage() {
  echo "usage: sbatch slurm_scripts/conditioning_probe.sh <thinker.pt> <painter.pt> [dit.pt]" >&2
  exit 1
}

THINKER="${1:-}"; PAINTER="${2:-}"; DIT="${3:-}"
[[ -z "${THINKER}" || -z "${PAINTER}" ]] && usage

COMBO="${COMBO:-square_n7}"
N_EACH="${N_EACH:-10}"
SWAP_N="${SWAP_N:-7}"
SWAP_A="${SWAP_A:-square}"
SWAP_B="${SWAP_B:-hexagon}"
DECODE_N="${DECODE_N:-200}"
DECODE_FRAC="${DECODE_FRAC:-0.9}"
DECODE_USE_ZH="${DECODE_USE_ZH:-false}"
STEPS="${STEPS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/amaze}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

TRM=( experiment=amaze_thinker_v2_controlnet painter.checkpoint="${PAINTER}"
      +checkpoint="${THINKER}" +task=maze +data_root="${DATA_ROOT}"
      +trajectory_combo="${COMBO}" +n_each="${N_EACH}" run.wandb_project="${WANDB_PROJECT}" )
[[ -n "${WANDB_RUN_ID}" ]] && TRM+=( +wandb_run_id="${WANDB_RUN_ID}" )

run_tool() {  # $1 = label, rest = command
  local label="$1"; shift
  echo "======== ${label} ========"
  if "$@"; then echo "[OK] ${label}"; else echo "[WARN] ${label} failed (continuing)"; fi
}

run_tool heatmaps   srun python experiments/conditioning_heatmaps.py   "${TRM[@]}" '+heatmap_fracs=[0.9,0.5,0.1]'
run_tool trajectory srun python experiments/conditioning_trajectory.py "${TRM[@]}" +trajectory_num_steps="${STEPS}"
run_tool swap       srun python experiments/conditioning_swap.py       "${TRM[@]}" \
                        +swap_n="${SWAP_N}" +swap_shape_a="${SWAP_A}" +swap_shape_b="${SWAP_B}" +n_pairs="${N_EACH}"
run_tool decode     srun python experiments/decode_conditioning.py     "${TRM[@]}" \
                        +decode_n_mazes="${DECODE_N}" +decode_frac="${DECODE_FRAC}" +decode_use_zH="${DECODE_USE_ZH}"

if [[ -n "${DIT}" ]]; then
  DITARGS=( experiment=amaze_dit_maze +checkpoint="${DIT}" +task=maze +data_root="${DATA_ROOT}"
            +trajectory_combo="${COMBO}" +n_each="${N_EACH}" +trajectory_num_steps="${STEPS}"
            run.wandb_project="${WANDB_PROJECT}" )
  [[ -n "${WANDB_RUN_ID}" ]] && DITARGS+=( +wandb_run_id="${WANDB_RUN_ID}" )
  run_tool trajectory_dit srun python experiments/conditioning_trajectory.py "${DITARGS[@]}"
fi

echo "conditioning probe complete — PNGs under $(dirname "${THINKER}")/conditioning/ ; decode table in the .out log."
