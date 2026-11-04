#!/bin/bash -l
#SBATCH --job-name=claude_eval_loops
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): eval_loops.py - original vs loop-trained plain TRM and PaTh thinker on the loop test sets.
# Usage: sbatch experiments/eval_loops_helios.sh <out_dir> [extra hydra overrides]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; OUT=$1; shift; mkdir -p "$OUT"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
export PLAIN_CKPTS="${PLAIN_CKPTS:-orig:runs/claude_plain_trm/checkpoint_final.pt,loops:runs/claude_plain_trm_loops/checkpoint_last.pt}"
export PATH_CKPTS="${PATH_CKPTS:-orig:runs/pt_maze_final_thinker/checkpoint_final.pt,loops:runs/claude_thinker_loops/checkpoint_final.pt}"
python experiments/eval_loops.py experiment=amaze_thinker_v2_controlnet \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  +interprebability_config.out_dir=$OUT hydra.run.dir=$OUT/hydra "$@"
