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
#SBATCH --output=/net/tscratch/people/plgmgrzanka/trm_sokoban/stdout/sokoban_eval_%x_%j.out
#SBATCH --error=/net/tscratch/people/plgmgrzanka/trm_sokoban/stderr/sokoban_eval_%x_%j.err
#
# Sample one trained run across several time_shift_xi values and log the test
# metrics back onto its W&B training charts.
#
# Args:
#   $1  RUN_NAME     train run name (also the checkpoint dir name)
#   $2  CONFIG_NAME  Hydra config-name to load (default: standard_diffusion).
#                    For resumed runs pass the config produced by fetch_run_config.py,
#                    e.g. "_resumed/<RUN_NAME>", so the model + checkpoint paths match.
#
# Usually launched per-run by sample_runs.sh (which generates the config first).

if [ -z "$1" ]; then
    echo "Usage: sbatch run_sampling.sh <run_name> [config_name]"
    exit 1
fi

RUN_NAME="$1"
CONFIG_NAME="${2:-standard_diffusion}"

module load ML-bundle/24.06a
source /net/tscratch/people/plgmgrzanka/trm_sokoban/venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/src:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

SHIFTS=(0.0 0.05 0.1 0.2 0.25)

echo "Evaluation for run: $RUN_NAME (config: $CONFIG_NAME)"

for SHIFT in "${SHIFTS[@]}"; do
    echo "Generation for time shift time_shift_xi = $SHIFT"
    echo "---------------------------------------------------"

    srun python sokoban/bit_diffusion/sample.py \
        --config-name="$CONFIG_NAME" \
        run_name="$RUN_NAME" \
        time_shift_xi=$SHIFT
done
