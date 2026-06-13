#!/bin/bash -l
#SBATCH --job-name=sokoban-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=plgdyplomancipw3tt-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=/net/tscratch/people/plgmgrzanka/trm_sokoban/stdout/sokoban_std_%j.out
#SBATCH --error=/net/tscratch/people/plgmgrzanka/trm_sokoban/stderr/sokoban_std_%j.err

if [ -z "$1" ]; then
    echo "USage: sbatch eval_shifts.sh [train run name]"
    exit 1
fi

RUN_NAME=$1

module load ML-bundle/24.06a
source /net/tscratch/people/plgmgrzanka/trm_sokoban/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/src:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

SHIFTS=(0.0 0.05 0.1 0.2 0.25)

echo "Evaluation for run: $RUN_NAME"

for SHIFT in "${SHIFTS[@]}"; do
    echo "Generation for time shift time_shift_xi = $SHIFT"
    echo "---------------------------------------------------"

    # Przekazanie zmiennych do skryptu za pomocą składni Hydra
    srun python sokoban/bit_diffusion/standard/sample.py \
        run_name="$RUN_NAME" \
        time_shift_xi=$SHIFT
done
