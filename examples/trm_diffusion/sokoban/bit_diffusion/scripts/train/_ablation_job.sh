#!/bin/bash -l
#SBATCH --job-name=sokoban-abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=/net/tscratch/people/plgmgrzanka/trm_sokoban/stdout/abl_%x_%j.out
#SBATCH --error=/net/tscratch/people/plgmgrzanka/trm_sokoban/stderr/abl_%x_%j.err
TRAIN_SCRIPT="$1"
RUN_NAME="$2"
WANDB_GROUP="$3"
shift 3

module load ML-bundle/24.06a
source /net/tscratch/people/plgmgrzanka/trm_sokoban/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/src:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

echo "=================================================="
echo "run_name : $RUN_NAME"
echo "script   : $TRAIN_SCRIPT"
echo "group    : $WANDB_GROUP"
echo "overrides: $*"
echo "=================================================="

srun python "$TRAIN_SCRIPT" \
    run_name="$RUN_NAME" \
    wandb_group="$WANDB_GROUP" \
    "$@"
