#!/bin/bash
# Cluster-aware sbatch wrapper.
#
# The launchers carry Helios (#SBATCH --account=...-gh200, --partition=plgrid-gpu-gh200)
# in their headers. Command-line flags override script directives, so this picks the
# right pair from the hostname and forwards everything else untouched.
#
#   bash slurm_scripts/submit.sh slurm_scripts/sample_amaze/eval_bagel.sh maze square
#   THINK=1 bash slurm_scripts/submit.sh --time=06:00:00 slurm_scripts/... queens
set -euo pipefail

case "$(hostname -f 2>/dev/null || hostname)" in
    *athena*)
        ACCOUNT="${SLURM_ACCOUNT_OVERRIDE:-plgdiffusion3-gpu-a100}"
        PARTITION="${SLURM_PARTITION_OVERRIDE:-plgrid-gpu-a100}"
        ;;
    *helios*)
        ACCOUNT="${SLURM_ACCOUNT_OVERRIDE:-plgdiffusion3-gpu-gh200}"
        PARTITION="${SLURM_PARTITION_OVERRIDE:-plgrid-gpu-gh200}"
        ;;
    *)
        echo "submit.sh: unknown cluster $(hostname -f). Set SLURM_ACCOUNT_OVERRIDE and SLURM_PARTITION_OVERRIDE." >&2
        exit 1
        ;;
esac

echo "submit.sh: account=${ACCOUNT} partition=${PARTITION}" >&2
exec sbatch --account="${ACCOUNT}" --partition="${PARTITION}" "$@"
