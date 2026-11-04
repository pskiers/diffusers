#!/bin/bash -l
#SBATCH --job-name=claude_discrete_trm
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): discrete token-per-cell TRM (train_discrete_trm.py) on helios. First generates the discrete data
# (discrete_maze.py, amaze generator via node, only if data/discrete/train.npz is missing), then trains.
# Resumable: resubmit to continue from runs/claude_discrete_trm/checkpoint_last.pt.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# nodejs before the venv: loading a module after activation puts the module python ahead of the venv on PATH
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0 GCCcore/13.2.0 nodejs/20.9.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p data/discrete
if [ ! -f data/discrete/train.npz ]; then
  python experiments/discrete_maze.py convert_test --sizes 5,7,8,9,10,11,13,16 --out_dir data/discrete
  python experiments/discrete_maze.py gen_train --per_size 4284 --out data/discrete/train.npz --nproc 60 --tmp "$PWD/data/discrete"
fi
OUT=runs/claude_discrete_trm; mkdir -p $OUT
python experiments/train_discrete_trm.py --train data/discrete/train.npz --test_dir data/discrete --test_sizes 5,8,10,11,13,16 \
  --out_dir $OUT --steps 40000 --keep_steps 4000,10000,20000,30000 --max_minutes 1320 >> $OUT/train_stdout.txt 2>&1
