#!/bin/bash -l
#SBATCH --job-name=cot_smoke_janus
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# CoT smoke test: 2 puzzles, 1 attempt, one shape, one size. Writes to a
# throwaway directory so nothing existing is touched.
#   sbatch slurm_scripts/test_cot_janus.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
EAR="${PROJECT_ROOT}/third_party/ear_amaze"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1
# Mirror eval_janus.sh's environment exactly (lines 57-68): infer_janus.py
# imports `data.*` relative to the ear_amaze root.
export HF_HOME="${SCRATCH}/.cache/huggingface"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${EAR}/sft/janus/Janus:${EAR}:${PYTHONPATH:-}"
cd "${EAR}"

CKPT="${CKPT:-${EAR}/sft/janus/outputs/maze/janus_train_maze/maze_20260914_070835/checkpoint-7-239904/tfmr}"
OUT="${PROJECT_ROOT}/runs/cot_smoke/janus_maze_square"
rm -rf "${OUT}"; mkdir -p "${OUT}"

COMMON=(
    --checkpoint_path "${CKPT}"
    --data_path "${PROJECT_ROOT}/data/amaze/ft/maze"
    --split test
    --batch_size 2
    --temperature 1.0
    --num_attempts 1
    --filter_shape square
    --filter_size_min 5
    --filter_size_max 5
    --samples_per_size 2
)

echo "############ RUN 1/2: THINK=0 (baseline) ############"
THINK=0 srun python "${EAR}/infer/infer_janus.py" "${COMMON[@]}" --output_dir "${OUT}/nocot"

echo
echo "############ RUN 2/2: THINK=1 (CoT) ############"
THINK=1 MAX_THINK_TOKENS=128 srun python "${EAR}/infer/infer_janus.py" "${COMMON[@]}" --output_dir "${OUT}/cot"

echo
echo "############ RESULT ############"
echo "images without CoT: $(find "${OUT}/nocot" -name '*_attempt*' 2>/dev/null | wc -l)"
echo "images with    CoT: $(find "${OUT}/cot"   -name '*_attempt*' 2>/dev/null | wc -l)"
echo "(look for '[CoT] plan[0]:' above -- that line proves the planning pass ran)"
