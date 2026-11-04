#!/bin/bash -l
#SBATCH --job-name=claude_plain_loops
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:45:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): plain TRM trained from scratch on the loop dataset (40% perfect, 15% each loop kind), same recipe as
# claude_plain_trm. Resumable: resubmit to continue from runs/claude_plain_trm_loops/checkpoint_last.pt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p runs/claude_plain_trm_loops
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
srun python experiments/train_plain_trm.py --train runs/claude_loop_data/maze/maze_dataset_train_144.parquet \
  --test runs/claude_loop_data/maze/square/n8_mixed_test.parquet --out_dir runs/claude_plain_trm_loops --max_minutes 395 --workers 12
