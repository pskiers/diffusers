#!/bin/bash -l
#SBATCH --job-name=amaze_gen_all
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --qos=now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# Generate the generalisation training subsets in ONE job, one after another, each
# with all the CPUs (generation is CPU-bound; the GPU is only there because the
# partition requires one). Not in parallel: generation holds a whole subset's images
# in memory at once (the September run peaked at its 64 GB limit), so four at a time
# would not fit in this QOS's ~120 GB. circle_only alone holds 2.7x the images of the
# whole September set, so it does not fit at all and is NOT in the default list: it
# runs as its own job with more memory (see submit_generalisation.sh).
# Each subset runs gen_generalisation_data.sh unchanged —
# same generation and the same row-count / marker-size checks — with its own log
# in slurm_outputs/<job name>_<job id>_<subset>.log. The job fails if any subset fails,
# so training jobs that depend on it (afterok) never start on bad data.
#
# QOS "now": +20000 priority, at most 12 h, 1 GPU, 72 CPUs, ~120 GB, and only ONE
# job per user at a time with this QOS. Generation of one subset took < 6 h in
# September (all maze + queens data together: 5 h 54 min).
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/sample_amaze/gen_generalisation_data_all.sh [subset ...]   # default: square_8 hexagon_only sizes_5_8_13
# Env: GEN_NPROC (workers; default all CPUs of the job), GEN_ROOT, OVERWRITE
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PROJECT_ROOT
source "${PROJECT_ROOT}/slurm_scripts/sample_amaze/generalisation_subsets.sh"
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs

if (( $# > 0 )); then SUBSETS=("$@"); else SUBSETS=(square_8 hexagon_only sizes_5_8_13); fi
for s in "${SUBSETS[@]}"; do subset_shapes "${s}" > /dev/null || exit 1; done

CPUS="${SLURM_CPUS_PER_TASK:-72}"
export GEN_NPROC="${GEN_NPROC:-${CPUS}}"
PREFIX="slurm_outputs/${SLURM_JOB_NAME:-amaze_gen_all}_${SLURM_JOB_ID:-local}"

FAILED=()
for s in "${SUBSETS[@]}"; do
  if [[ -f "$(subset_train_parquet "${s}")" ]]; then
    echo ">> ${s}: already generated, skipped"
    continue
  fi
  echo ">> ${s}: started $(date +%T) (${GEN_NPROC} workers, log ${PREFIX}_${s}.log)"
  if bash slurm_scripts/sample_amaze/gen_generalisation_data.sh "${s}" > "${PREFIX}_${s}.log" 2>&1; then
    echo ">> ${s}: OK $(date +%T)  ($(grep -h "marker area" "${PREFIX}_${s}.log" | tail -1))"
  else
    echo ">> ${s}: FAILED $(date +%T) — see ${PREFIX}_${s}.log" >&2
    FAILED+=("${s}")
  fi
done

if (( ${#FAILED[@]} > 0 )); then
  echo "ERROR: subsets failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "gen_generalisation_data_all done: ${SUBSETS[*]}"
