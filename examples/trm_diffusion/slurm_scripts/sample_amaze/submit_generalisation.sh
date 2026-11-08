#!/bin/bash -l
#
# Submit the whole generalisation experiment:
#   - ONE data job on QOS "now" (fast start) that builds the missing small subsets one
#     after another (gen_generalisation_data_all.sh);
#   - circle_only on its own, on the normal queue with more memory: it holds 2.7x the
#     images of the whole September set (which peaked at 64 GB), more than "now" allows;
#   - 7 thinker runs, each waiting (afterok) for the job that builds ITS subset.
# A subset that already exists is not regenerated and its runs get no dependency.
#
#   bash slurm_scripts/sample_amaze/submit_generalisation.sh            # DRY RUN - prints what it would submit
#   bash slurm_scripts/sample_amaze/submit_generalisation.sh submit     # actually sbatch everything
#
# wandb: the runs log with the environment of THIS shell — export WANDB_ENTITY,
# WANDB_KEY_FILE (and optionally WANDB_PROJECT) before submitting to log somewhere
# other than ~/.netrc's default workspace (see train_generalisation.sh).
#
# Runs (KIND per subset):
#   sizes_5_8_13  trm + linear_head   size generalisation
#   circle_only   trm + linear_head   shape generalisation
#   hexagon_only  trm + linear_head   shape generalisation
#   square_8      trm                 extreme generalisation (one shape, one size)
set -uo pipefail

MODE="${1:-dry}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${PROJECT_ROOT}"
source slurm_scripts/sample_amaze/generalisation_subsets.sh

declare -A RUN_KINDS=(
  [sizes_5_8_13]="trm linear_head"
  [circle_only]="trm linear_head"
  [hexagon_only]="trm linear_head"
  [square_8]="trm"
)

submit() {   # prints the job id (or a placeholder in dry-run mode)
  if [[ "${MODE}" == "submit" ]]; then
    sbatch --parsable "$@"
  else
    echo "DRY_${RANDOM}"
  fi
}

echo "=============================================================="
echo " MODE=${MODE}   root=${PROJECT_ROOT}   data=${GEN_ROOT}"
echo " wandb: project ${WANDB_PROJECT:-amaze_generalisation}, workspace ${WANDB_ENTITY:-<~/.netrc default>}, key ${WANDB_KEY_FILE:-~/.netrc}"
echo "=============================================================="

BIG_SUBSETS=(circle_only)                       # own job, more memory (see above)
BIG_SBATCH=( --cpus-per-task=72 --mem=240G --time=24:00:00 )

# Athena (A100): no data generation there (the generator's node_modules are built for
# the aarch64 Helios nodes) — copy the subsets from Helios; missing ones are skipped.
# A100s are slower than GH200s, so training gets Athena's 48 h maximum.
TRAIN_SBATCH=()
CLUSTER=helios
if [[ "${SCRATCH:-}" == /net/tscratch/* ]]; then
  CLUSTER=athena
  TRAIN_SBATCH=( --account="${ATHENA_ACCOUNT:-plgdiffusion3-gpu-a100}" --partition=plgrid-gpu-a100
                 --time="${TRAIN_TIME:-48:00:00}" )
fi
echo " cluster: ${CLUSTER} ${TRAIN_SBATCH[*]}"
ONLY="${ONLY:-}"   # optional: space-separated run names (gen_<subset>_<kind>) to submit; default all

declare -A DEP_OF
declare -A SKIP_SUBSET
SMALL_MISSING=()
for SUBSET in "${GEN_SUBSETS[@]}"; do
  if [[ -f "$(subset_train_parquet "${SUBSET}")" ]]; then
    echo "[data ] ${SUBSET}: exists, not regenerated"
  elif [[ "${CLUSTER}" == athena ]]; then
    echo "[data ] ${SUBSET}: MISSING on Athena (generate on Helios and copy) — its runs are skipped"
    SKIP_SUBSET[${SUBSET}]=1
  elif [[ " ${BIG_SUBSETS[*]} " == *" ${SUBSET} "* ]]; then
    ID=$(submit "${BIG_SBATCH[@]}" --job-name="amaze_gen_${SUBSET}" \
      slurm_scripts/sample_amaze/gen_generalisation_data.sh "${SUBSET}")
    echo "[data ] ${SUBSET}: job ${ID} (normal queue, ${BIG_SBATCH[*]})"
    DEP_OF[${SUBSET}]="--dependency=afterok:${ID}"
  else
    SMALL_MISSING+=("${SUBSET}")
  fi
done
if (( ${#SMALL_MISSING[@]} > 0 )); then
  ID=$(submit slurm_scripts/sample_amaze/gen_generalisation_data_all.sh "${SMALL_MISSING[@]}")
  echo "[data ] ${SMALL_MISSING[*]}: one job ${ID} (QOS now)"
  for SUBSET in "${SMALL_MISSING[@]}"; do DEP_OF[${SUBSET}]="--dependency=afterok:${ID}"; done
fi

for SUBSET in "${GEN_SUBSETS[@]}"; do
  [[ -n "${SKIP_SUBSET[${SUBSET}]:-}" ]] && continue
  for KIND in ${RUN_KINDS[${SUBSET}]}; do
    if [[ -n "${ONLY}" && " ${ONLY} " != *" gen_${SUBSET}_${KIND} "* ]]; then continue; fi
    DEP=${DEP_OF[${SUBSET}]:-}
    RUN_ID=$(submit ${TRAIN_SBATCH[@]+"${TRAIN_SBATCH[@]}"} ${DEP:+"${DEP}"} --job-name="gen_${SUBSET}_${KIND}" \
      slurm_scripts/sample_amaze/train_generalisation.sh "${SUBSET}" "${KIND}")
    echo "[train] gen_${SUBSET}_${KIND}: job ${RUN_ID} ${DEP}"
  done
done

[[ "${MODE}" == "submit" ]] || echo -e "\nDry run only. Re-run with 'submit' to queue the jobs."
