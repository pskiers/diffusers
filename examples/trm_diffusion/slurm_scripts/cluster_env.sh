# Cluster detection + environment for the AMAZE / PaTh-rebuttal jobs (sourced, not run).
#
#   source slurm_scripts/cluster_env.sh            # sets CLUSTER, VENV, CLUSTER_GPU_SBATCH, CLUSTER_CPU_SBATCH
#   cluster_activate                               # inside a job: modules + venv + PYTHONPATH
#
# Detection (override with CLUSTER=helios|athena|lem):
#   Helios  SCRATCH=/net/scratch/hscra/...   aarch64 GH200 96 GB
#   Athena  SCRATCH=/net/tscratch/...        x86_64  A100 40 GB
#   lem     WCSS (/lustre/pd03 project dir)  x86_64  H100 (Hopper)
# The #SBATCH lines in the job scripts are Helios'; submit with "${CLUSTER_GPU_SBATCH[@]}"
# (or CLUSTER_CPU_SBATCH for CPU-only work) to get the right account/partition/GPU type.

LEM_PROJECT="${LEM_PROJECT:-/lustre/pd03/hpc-kamildeja-1773916679}"

if [[ -z "${CLUSTER:-}" ]]; then
  case "${SCRATCH:-}" in
    /net/tscratch/*)     CLUSTER=athena ;;
    /net/scratch/hscra/*) CLUSTER=helios ;;
    *) if [[ -d "${LEM_PROJECT}" ]]; then CLUSTER=lem; else CLUSTER=unknown; fi ;;
  esac
fi

case "${CLUSTER}" in
  helios)
    VENV="${VENV:-${SCRATCH}/trm_helios_venv}"
    CLUSTER_MODULES="${MODULES:-Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0}"
    CLUSTER_NODE_MODULES="${NODE_MODULES:-GCCcore/14.3.0 nodejs/22.17.1}"
    CLUSTER_GPU_SBATCH=( --account=plgdiffusion3-gpu-gh200 --partition=plgrid-gpu-gh200 )
    CLUSTER_CPU_SBATCH=( "${CLUSTER_GPU_SBATCH[@]}" )      # Helios has GPU partitions only
    ;;
  athena)
    VENV="${VENV:-${SCRATCH}/trm_athena_venv}"
    CLUSTER_MODULES="${MODULES:-Python/3.10.4 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0}"
    CLUSTER_NODE_MODULES="${NODE_MODULES:-}"
    CLUSTER_GPU_SBATCH=( --account=plgdiffusion3-gpu-a100 --partition=plgrid-gpu-a100 )
    CLUSTER_CPU_SBATCH=( "${CLUSTER_GPU_SBATCH[@]}" )
    ;;
  lem)
    # torch comes from the cu124 wheel (its own CUDA runtime), so only Python is
    # needed at run time; the Python module also brings GCCcore 12.3, whose
    # libstdc++ the compiled adam_atan2 backend was built against.
    VENV="${VENV:-${LEM_PROJECT}/mgrzanka/trm_lem_venv}"
    CLUSTER_MODULES="${MODULES:-Python/3.11.3-GCCcore-12.3.0}"
    CLUSTER_NODE_MODULES="${NODE_MODULES:-}"
    CLUSTER_GPU_SBATCH=( --account=hpc-kamildeja-1773916679 --partition="${LEM_GPU_PARTITION:-lem-gpu-short}"
                         --gres=gpu:hopper:1 )
    CLUSTER_CPU_SBATCH=( --account=hpc-kamildeja-1773916679 --partition="${LEM_CPU_PARTITION:-lem-cpu-short}"
                         --gres=none )
    ;;
  *)
    VENV="${VENV:-}"; CLUSTER_MODULES="${MODULES:-}"; CLUSTER_NODE_MODULES="${NODE_MODULES:-}"
    CLUSTER_GPU_SBATCH=(); CLUSTER_CPU_SBATCH=()
    ;;
esac

# wandb: all rebuttal runs go to the user's workspace; the key is read from a file
# inside the job (never on a command line). Override any of these via the environment.
export WANDB_ENTITY="${WANDB_ENTITY:-magmatic624-politechnika-warszawska}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.wandb_key_magmatic624}"

cluster_activate() {   # cluster_activate [--node]  (--node also loads the nodejs modules)
  [[ "${CLUSTER}" == unknown && -z "${VENV}" ]] && { echo "cluster_env: unknown cluster, set VENV/MODULES" >&2; return 1; }
  if [[ "${CLUSTER}" == lem ]] && ! type module >/dev/null 2>&1; then
    source /usr/local/sbin/modules.sh
  fi
  # shellcheck disable=SC2086
  [[ -n "${CLUSTER_MODULES}" ]] && module load ${CLUSTER_MODULES}
  # shellcheck disable=SC2086
  [[ "${1:-}" == --node && -n "${CLUSTER_NODE_MODULES}" ]] && module load ${CLUSTER_NODE_MODULES}
  [[ "${CLUSTER}" == helios ]] && export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
  source "${VENV}/bin/activate"
  # The diffusers of THIS checkout (<repo>/src), not whatever the venv's editable install points at.
  local src
  src="$(cd "${PROJECT_ROOT:-$PWD}/../.." && pwd)/src"
  [[ -d "${src}/diffusers" ]] && export PYTHONPATH="${src}:${PYTHONPATH:-}"
  if [[ -r "${WANDB_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
    export WANDB_API_KEY
  fi
  export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
  echo ">> cluster=${CLUSTER} host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(python -V 2>&1) venv=${VENV}" \
       "wandb=${WANDB_ENTITY} (key: $([[ -n "${WANDB_API_KEY:-}" ]] && echo "${WANDB_KEY_FILE}" || echo '~/.netrc'))"
}
