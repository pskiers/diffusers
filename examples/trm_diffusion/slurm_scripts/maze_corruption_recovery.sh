#!/bin/bash -l
#SBATCH --job-name=maze_corrupt
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# Maze corruption-recovery probe (scenario A): TRM vs DiT in ONE job.
# Compares whether each model fixes deliberately injected path mistakes
# (GAP / ADD / WALL) at context levels 10/45/80%, noised to t_start and denoised.
#
# One-command launch (from the trm_diffusion dir):
#     sbatch slurm_scripts/maze_corruption_recovery.sh
#
# Short validation smoke (n=8, dumps before/after PNGs) instead of the full run:
#     sbatch --export=ALL,SMOKE=1 slurm_scripts/maze_corruption_recovery.sh
#
# Env overrides: NUM_SAMPLES(128) T_START(40) BATCH(32) SMOKE(false)
#                DATA(data/amaze/test_maze/square/all_square_test.parquet) OUT_DIR(runs/maze_corruption)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
T_STARTS="${T_STARTS:-[15,20,25,30]}"
BATCH="${BATCH:-32}"
DATA="${DATA:-data/amaze/test_maze/square/all_square_test.parquet}"
SMOKE="${SMOKE:-false}"
OUT_DIR="${OUT_DIR:-runs/maze_corruption}"

module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${VENV}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs "${OUT_DIR}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"

is_true(){ case "${1,,}" in true|1|yes|on) return 0;; *) return 1;; esac; }
EXTRA=()
is_true "${SMOKE}" && EXTRA+=("+probe.smoke=true")

echo "=== [maze corruption] TRM  (t_starts=${T_STARTS} n=${NUM_SAMPLES} smoke=${SMOKE}) ==="
srun python experiments/maze_corruption_recovery_probe.py \
  experiment=amaze_thinker_v2_controlnet \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
  +probe.model_name=trm +probe.t_starts="${T_STARTS}" +probe.batch_size="${BATCH}" \
  +probe.num_samples="${NUM_SAMPLES}" +probe.data_parquet="${DATA}" \
  +probe.out="${OUT_DIR}/trm.json" "${EXTRA[@]}"

echo "=== [maze corruption] DiT  (t_starts=${T_STARTS} n=${NUM_SAMPLES} smoke=${SMOKE}) ==="
srun python experiments/maze_corruption_recovery_probe.py \
  experiment=amaze_dit_maze \
  +checkpoint=runs/dit_maze_final/checkpoint_final.pt \
  +probe.model_name=dit +probe.t_starts="${T_STARTS}" +probe.batch_size="${BATCH}" \
  +probe.num_samples="${NUM_SAMPLES}" +probe.data_parquet="${DATA}" \
  +probe.out="${OUT_DIR}/dit.json" "${EXTRA[@]}"

echo "DONE -> ${OUT_DIR}/{trm,dit}.json"
