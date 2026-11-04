#!/bin/bash -l
#SBATCH --job-name=claude_discrete_trm
#SBATCH --account=plgdiffusion3-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): discrete token-per-cell TRM (train_discrete_trm.py) on athena, data/discrete copied from titan.
# Stops after 35 h with a checkpoint; resumable: resubmit to continue from runs/claude_discrete_trm/checkpoint_last.pt.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load Python/3.10.4 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_athena_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
OUT=runs/claude_discrete_trm; mkdir -p $OUT
srun python experiments/train_discrete_trm.py --train data/discrete/train.npz --test_dir data/discrete --test_sizes 5,8,10,11,13,16 \
  --out_dir $OUT --steps 40000 --keep_steps 4000,10000,20000,30000 --max_minutes 2100 >> $OUT/train_stdout.txt 2>&1
