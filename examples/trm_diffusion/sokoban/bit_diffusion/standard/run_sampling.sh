#!/bin/bash -l
#SBATCH --job-name=sokoban-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=plgdynamic3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/slurm_outputs/stdout/sokoban_eval_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/slurm_outputs/stderr/sokoban_eval_%j.err

if [ -z "$1" ]; then
    echo "USage: sbatch eval_shifts.sh [train run name]"
    exit 1
fi

RUN_NAME=$1

module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/venv/bin/activate

export SSL_CERT_FILE=/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/cacert.pem
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/src:/net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/scratch/hscra/plgrid/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

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
