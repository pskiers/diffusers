#!/bin/bash -l
#SBATCH --job-name=discrete-trm-sokoban
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=plgdyplomancipw2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=/net/tscratch/people/plgmgrzanka/training-outputs/discrete-trm-sokoban/slurm/stdout_%j.out
#SBATCH --error=/net/tscratch/people/plgmgrzanka/training-outputs/discrete-trm-sokoban/slurm/stderr_%j.err

# ── Environment setup ──
module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgmgrzanka/reasoning_diffusion_env

# ── Navigate to project root ──
cd /net/people/plgrid/plgmgrzanka//diffusers/examples/trm_diffusion

# ── Ensure WandB is configured (set WANDB_API_KEY in your environment or .bashrc) ──
export WANDB_PROJECT="discrete-trm-sokoban"

# ── Create output directories ──
mkdir -p /net/tscratch/people/plgmgrzanka/training-outputs/discrete-trm-sokoban/slurm

# ── Launch training ──
accelerate launch --mixed_precision="bf16" --num_processes=1 \
    sokoban/diffusion/train_discrete.py \
    "$@"
