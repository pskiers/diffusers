#!/bin/bash -l
#SBATCH --job-name=claude_first_step
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): interp_first_step.py, PaTh thinker at the first denoising step vs plain TRMs, square n8-n16.
# Usage: sbatch experiments/first_step_helios.sh <out_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$1"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
export PLAIN_CKPTS=plain:runs/claude_plain_trm/checkpoint_final.pt,noisy:runs/claude_plain_trm_noisy/checkpoint_final.pt
python experiments/interp_first_step.py experiment=amaze_thinker_v2_controlnet \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
  +interprebability_config.out_dir=$1 hydra.run.dir=$1/hydra
