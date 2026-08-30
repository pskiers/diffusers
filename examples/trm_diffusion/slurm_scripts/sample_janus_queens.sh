#!/bin/bash -l
#SBATCH --job-name=amaze_janus_qsample
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

# Score EVERY epoch-checkpoint of the already-trained Janus-queens run on the AMAZE metrics
# and report which epoch won. This run predates validation logging (no val loss), so we pick
# the checkpoint on the real task metric instead. Janus INFERENCE (7B, ~14 GB) fits one A100-40GB.
#
# Usage: sbatch slurm_scripts/sample_janus_queens.sh
# Env: RUN (ft_janus_queens), TASK (queens), SAMPLES (5), BY (pass1|pass5),
#      WANDB_PROJECT (amaze_final), VENV (default $SCRATCH/trm_sokoban/venv).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_sokoban/venv}"
RUN="${RUN:-ft_janus_queens}"
TASK="${TASK:-queens}"
SAMPLES="${SAMPLES:-5}"
BY="${BY:-pass1}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"

if [[ -n "${MODULES:-}" ]]; then module load ${MODULES}; fi
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/third_party/amaze/sft/janus/Janus:${PROJECT_ROOT}/third_party/amaze:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"

mapfile -t CKPTS < <(find "${PROJECT_ROOT}/runs/${RUN}" -type d -name tfmr 2>/dev/null | sort -V)
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "ERROR: no checkpoint-*/tfmr under runs/${RUN} (copied from Helios?)." >&2; exit 1; }

AMAZE_OUT_ROOT="${PROJECT_ROOT}/data/amaze" python scripts/gen_amaze.py test "${TASK}"   # build the test set ONCE
echo ">> scoring ${#CKPTS[@]} Janus checkpoint(s) on ${TASK} (one wandb run per epoch)."
for CKPT in "${CKPTS[@]}"; do
  TAG=$(basename "$(dirname "${CKPT}")")           # checkpoint-<epoch>-<step>
  GEN_DIR="runs/${RUN}/generated/${TAG}"
  if [[ -f "${GEN_DIR}/.scored" ]]; then echo ">> [${TAG}] already scored -> skip."; continue; fi
  echo ">> [${TAG}] sampling: ${CKPT}"
  python experiments/generate_amaze_ft.py "${TASK}" --backend janus \
    --checkpoint "${CKPT}" --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
  && python experiments/score_amaze_images.py "${TASK}" \
    --gen-dir "${GEN_DIR}" --samples-per-puzzle "${SAMPLES}" \
    --wandb-project "${WANDB_PROJECT}" --run-name "${RUN}-${TAG}" \
  && touch "${GEN_DIR}/.scored" \
  || echo "WARN: [${TAG}] sampling/scoring failed -- continuing with the next checkpoint."
done

echo ">> ranking epochs by ${BY}:"
python experiments/report_best_epoch.py "runs/${RUN}/generated" "${TASK}" --by "${BY}"
echo "sample_janus_queens done -> wandb ${WANDB_PROJECT}."
