#!/bin/bash -l
#SBATCH --job-name=sokoban-std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --account=plgdynamic3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/slurm_outputs/stdout/sokoban_std_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/slurm_outputs/stderr/sokoban_std_%j.err

module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/venv/bin/activate

export SSL_CERT_FILE=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/cacert.pem
export PYTHONUNBUFFERED=1

export PYTHONPATH=$PYTHONPATH:/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/src:/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

srun python sokoban/bit_diffusion/standard/train_std.py
