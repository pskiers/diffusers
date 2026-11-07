#!/bin/bash
# Run a python script on the interactive GPU allocation with the project venv.
# Usage: ./run.sh <script.py> [args...]   (JOBID env overrides the allocation)
JOBID="${JOBID:-24039991}"
cd "$(dirname "$0")"
exec srun --jobid="$JOBID" --overlap bash -lc '
module load Python/3.11.5 CUDA/12.4.0 >/dev/null 2>&1
export LD_LIBRARY_PATH=/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
source /net/scratch/hscra/plgrid/plgmgrzanka/trm_helios_venv/bin/activate
export PYTHONUNBUFFERED=1
python "$@"' _ "$@" 2>&1 | grep -v " loaded\.$"
