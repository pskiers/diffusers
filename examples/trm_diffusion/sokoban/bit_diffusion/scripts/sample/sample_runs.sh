#!/bin/bash -l
# Sample one or more trained runs by name.
#
# For each run name it pulls the run's full config from W&B (on the login node, fail-fast on typos), then submits a SLURM job (run_sampling.sh) that sweeps time_shift_xi and logs the test metrics back onto that run's existing W&B charts.
#
# Usage:
#   PROJECT=<wandb_project> [ENTITY=<entity>] ./sample_runs.sh <run_name> [run_name ...]
#
# Example:
#   PROJECT=Sokoban-Ablation-Uncond ENTITY=my-team ./sample_runs.sh trm_base emb_base
#
# Env:
#   PROJECT   W&B project the runs live in   (default: Sokoban-Ablation-Uncond)
#   ENTITY    W&B entity/team

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/run_sampling.sh"
FETCH="$SCRIPT_DIR/fetch_run_config.py"

PROJECT="${PROJECT:-Sokoban-Ablation-Uncond}"
ENTITY="${ENTITY:-}"

if [ "$#" -lt 1 ]; then
    echo "Usage: PROJECT=<wandb_project> [ENTITY=<entity>] $0 <run_name> [run_name ...]"
    exit 1
fi

module load ML-bundle/24.06a
source /net/tscratch/people/plgmgrzanka/trm_sokoban/venv/bin/activate

export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
export PYTHONPATH=$PYTHONPATH:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/src:/net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

cd /net/tscratch/people/plgmgrzanka/trm_sokoban/diffusers/examples/trm_diffusion

FETCH_ARGS=(--project "$PROJECT")
if [ -n "$ENTITY" ]; then
    FETCH_ARGS+=(--entity "$ENTITY")
fi

for RUN_NAME in "$@"; do
    echo "=================================================="
    echo "Fetching W&B config for '$RUN_NAME' (project=$PROJECT)"

    CONFIG_NAME="$(python "$FETCH" --run-name "$RUN_NAME" "${FETCH_ARGS[@]}" | tail -n 1)"
    if [ -z "$CONFIG_NAME" ]; then
        echo "Could not resolve config for '$RUN_NAME', skipping."
        continue
    fi

    echo "config: $CONFIG_NAME"
    sbatch --job-name="eval-$RUN_NAME" "$JOB" "$RUN_NAME" "$CONFIG_NAME"
done

echo "=================================================="
echo "Submitted. Monitor with: squeue --me"
