#!/bin/bash -l
#SBATCH --job-name=amaze_janus_qsample
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
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
# the checkpoint on the real task metric instead. Runs on Helios (checkpoints already live there);
# Janus INFERENCE (7B, ~14 GB) fits one GH200.
#
# Usage: sbatch slurm_scripts/sample_janus_queens.sh
# Env: RUN (ft_janus_queens), TASK (queens), SAMPLES (5),
#      WANDB_PROJECT (amaze_final), VENV (default $SCRATCH/trm_helios_venv).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
RUN="${RUN:-ft_janus_queens}"
TASK="${TASK:-queens}"
SAMPLES="${SAMPLES:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-amaze_final}"

if [[ -n "${MODULES:-}" ]]; then module load ${MODULES}; else module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0; fi
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
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

echo ">> best epoch by Pass@1 (each epoch is also a separate run in ${WANDB_PROJECT}):"
python - "runs/${RUN}/generated" "${TASK}" <<'PY'
import glob, json, os, sys
root, task = sys.argv[1], sys.argv[2]
best = None
for jf in glob.glob(os.path.join(root, "*", f"amaze_metrics_{task}.json")):
    try:
        ov = json.load(open(jf)).get("overall", {})
    except Exception:
        continue
    p1, tag = ov.get("pass1"), os.path.basename(os.path.dirname(jf))
    if p1 is not None and (best is None or p1 > best[1]):
        best = (tag, p1, ov.get("pass5") or 0.0)
print(f"  BEST: {best[0]}  P@1={best[1]*100:.2f}%  P@5={best[2]*100:.2f}%" if best else "  (no scores found)")
PY
echo "sample_janus_queens done -> wandb ${WANDB_PROJECT}."
