#!/bin/bash -l
#
# Submit the whole generalisation experiment:
#   - data for missing subsets (Helios: ONE QOS "now" job for the small ones,
#     gen_generalisation_data_all.sh; lem: one CPU-partition job per subset);
#   - circle_only on its own, on the normal queue with more memory: it holds 2.7x the
#     images of the whole September set (which peaked at 64 GB), more than "now" allows;
#   - 7 thinker runs, each waiting (afterok) for the job that builds ITS subset.
# A subset that already exists is not regenerated and its runs get no dependency.
#
#   bash slurm_scripts/sample_amaze/submit_generalisation.sh            # DRY RUN - prints what it would submit
#   bash slurm_scripts/sample_amaze/submit_generalisation.sh submit     # actually sbatch everything
#
# Cluster (Helios / Athena / lem), account, partition, GPU type and the wandb workspace
# + key file come from slurm_scripts/cluster_env.sh; runs log to PaTh-rebuttal.
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

source slurm_scripts/cluster_env.sh
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
export WANDB_PROJECT
echo "=============================================================="
echo " MODE=${MODE}   cluster=${CLUSTER}   root=${PROJECT_ROOT}   data=${GEN_ROOT}"
echo " wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} (group generalisation), key ${WANDB_KEY_FILE}"
echo "=============================================================="
# Slurm needs the --output directory to exist before the job starts (on lem a missing
# one makes the job fail without any log).
mkdir -p slurm_outputs

# Training: the cluster's account/partition/GPU type, and only the CPUs/memory a run
# uses (4 data-loader workers; the parquet is < 1 GB). A100s are slower -> 48 h there.
TRAIN_SBATCH=( ${CLUSTER_GPU_SBATCH[@]+"${CLUSTER_GPU_SBATCH[@]}"} )
case "${CLUSTER}" in
  athena) TRAIN_SBATCH+=( --time="${TRAIN_TIME:-48:00:00}" ) ;;
  lem)    TRAIN_SBATCH+=( --time="${TRAIN_TIME:-30:00:00}" --cpus-per-task=8 --mem=64G ) ;;   # H100: ~14 h per run
esac
echo " train sbatch: ${TRAIN_SBATCH[*]}"
ONLY="${ONLY:-}"   # optional: space-separated run names (gen_<subset>_<kind>) to submit; default all

# Data generation (CPU-bound; holds a whole subset's images in memory: the September
# set peaked at 64 GB, circle_only needs ~180 GB).
#   helios  small subsets in one QOS-now job, circle_only alone with 240 GB
#   lem     one CPU-partition job per subset (circle_only with 240 GB)
#   athena  not possible (no generator set up) -> copy the subset, its runs are skipped
BIG_SUBSETS=(circle_only)
BIG_SBATCH=( --cpus-per-task=72 --mem=240G --time=24:00:00 )
[[ "${CLUSTER}" == lem ]] && BIG_SBATCH=( "${CLUSTER_CPU_SBATCH[@]}" --cpus-per-task=64 --mem=240G --time=24:00:00 )
SMALL_LEM_SBATCH=( ${CLUSTER_CPU_SBATCH[@]+"${CLUSTER_CPU_SBATCH[@]}"} --cpus-per-task=32 --mem=100G --time=12:00:00 )

declare -A DEP_OF
declare -A SKIP_SUBSET
SMALL_MISSING=()
for SUBSET in "${GEN_SUBSETS[@]}"; do
  if [[ -f "$(subset_train_parquet "${SUBSET}")" ]]; then
    echo "[data ] ${SUBSET}: exists, not regenerated"
  elif [[ "${CLUSTER}" == athena ]]; then
    echo "[data ] ${SUBSET}: MISSING on Athena (generate elsewhere and copy) — its runs are skipped"
    SKIP_SUBSET[${SUBSET}]=1
  elif [[ " ${BIG_SUBSETS[*]} " == *" ${SUBSET} "* ]]; then
    ID=$(submit "${BIG_SBATCH[@]}" --job-name="amaze_gen_${SUBSET}" \
      slurm_scripts/sample_amaze/gen_generalisation_data.sh "${SUBSET}")
    echo "[data ] ${SUBSET}: job ${ID} (${BIG_SBATCH[*]})"
    DEP_OF[${SUBSET}]="--dependency=afterok:${ID}"
  elif [[ "${CLUSTER}" == lem ]]; then
    ID=$(submit "${SMALL_LEM_SBATCH[@]}" --job-name="amaze_gen_${SUBSET}" \
      slurm_scripts/sample_amaze/gen_generalisation_data.sh "${SUBSET}")
    echo "[data ] ${SUBSET}: job ${ID} (${SMALL_LEM_SBATCH[*]})"
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
