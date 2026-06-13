#!/bin/bash -l
#SBATCH --job-name=sokoban-std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=/net/tscratch/people/plgmgrzanka/trm_sokoban/stdout/sokoban_std_%j.out
#SBATCH --error=/net/tscratch/people/plgmgrzanka/trm_sokoban/stderr/sokoban_std_%j.err

module load Miniconda3/23.3.1-0
source /net/tscratch/people/plgmgrzanka/trm_sokoban/venv/bin/activate

export PYTHONUNBUFFERED=1

export PYTHONPATH=$PYTHONPATH:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/src:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

srun python sokoban/bit_diffusion/standard/train_std.py
