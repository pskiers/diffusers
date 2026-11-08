#!/bin/bash -l
#SBATCH --job-name=amaze_dit_sized
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# The maze DiT baseline (experiment=amaze_dit_maze, exactly the dit_maze_final recipe: 80k steps, batch 128 x
# accumulation 6, cfg_prob 0) at a larger MODEL SIZE, for the FLOPs-vs-accuracy plot (Sudoku had 3 DM sizes).
# Only depth and width change; tokens (144) and patch size (12) stay fixed. Exact parameter counts:
#   20m     12 layers,  6 heads x 48 = 288     20.4M   (rebuttal: small)
#   base    12 layers,  8 heads x 64 = 512     62.6M   (the existing dit_maze_final; reference, not retrained)
#   medium  16 layers, 12 heads x 64 = 768    184.9M
#   200m    18 layers, 12 heads x 64 = 768    207.7M  (rebuttal: large)
#   large   24 layers, 16 heads x 64 = 1024   488.3M
#
# Usage (from the trm_diffusion dir; account/partition/GPU from slurm_scripts/cluster_env.sh):
#   source slurm_scripts/cluster_env.sh && sbatch "${CLUSTER_GPU_SBATCH[@]}" slurm_scripts/train_amaze/train_dit_sized.sh <20m|200m|medium|large>
# Env: STEPS (80000), RUN_NAME (dit_maze_<size>), WANDB_PROJECT (PaTh-rebuttal, group flops; "null" disables wandb),
#      STEP_TEST=1  -> a short timing run only: no data generation, no eval, no wandb, writes runs/<RUN_NAME>_steptest,
#                      and reports it/s, peak GPU memory and the exact parameter count.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIZE="${1:?usage: train_dit_sized.sh <20m|200m|medium|large>}"
HEAD_DIM=64
case "${SIZE}" in
  20m)    LAYERS=12; HEADS=6; HEAD_DIM=48 ;;
  medium) LAYERS=16; HEADS=12 ;;
  200m)   LAYERS=18; HEADS=12 ;;
  large)  LAYERS=24; HEADS=16 ;;
  *) echo "SIZE must be 20m|200m|medium|large" >&2; exit 1 ;;
esac

STEPS="${STEPS:-80000}"
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
RUN_NAME="${RUN_NAME:-dit_maze_${SIZE}}"
EXTRA=()
if [[ "${STEP_TEST:-0}" == "1" ]]; then
  STEPS="${STEPS_TEST:-300}"; RUN_NAME="${RUN_NAME}_steptest"; WANDB_PROJECT=null
  EXTRA=( eval.log_every=20 eval.eval_every=100000000 eval.save_every=100000000 )
fi

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs runs
echo ">> size=${SIZE} layers=${LAYERS} heads=${HEADS}x${HEAD_DIM} steps=${STEPS} run=${RUN_NAME} host=$(hostname)"

# The maze training set must already exist (same file dit_maze_final trained on).
TRAIN_PARQUET="${PROJECT_ROOT}/data/amaze/maze/maze_dataset_train_144.parquet"
[[ -s "${TRAIN_PARQUET}" ]] || { echo "ERROR: missing ${TRAIN_PARQUET}" >&2; exit 1; }

MEMLOG="runs/${RUN_NAME}.gpumem.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "${MEMLOG}" &
MEMPID=$!
trap 'kill ${MEMPID} 2>/dev/null || true' EXIT

T0=$(date +%s)
srun python train_trm.py experiment=amaze_dit_maze \
  data.amaze_root="${PROJECT_ROOT}/data/amaze/maze" \
  backbone.num_layers=${LAYERS} backbone.num_attention_heads=${HEADS} backbone.attention_head_dim=${HEAD_DIM} \
  train.num_steps=${STEPS} train.max_seconds=0 \
  run.wandb_project="${WANDB_PROJECT}" \
  run.wandb_group=flops run.wandb_job_type=train "run.wandb_tags=[flops,dit,dit_${SIZE},${CLUSTER}]" \
  run.output_dir="runs/${RUN_NAME}" ${EXTRA[@]+"${EXTRA[@]}"}
T1=$(date +%s)

python - "runs/${RUN_NAME}/checkpoint_final.pt" "${STEPS}" "$((T1 - T0))" "${MEMLOG}" <<'PY'
import sys, torch
ck, steps, secs, memlog = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sd = torch.load(ck, map_location="cpu", weights_only=True)["model_state"]
n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
mem = [float(x) for x in open(memlog).read().split() if x.strip()]
print(f"RESULT params={n/1e6:.1f}M wall={secs}s steps={steps} (incl. startup+compile) "
      f"peak_gpu_mem={max(mem)/1024:.1f}GiB" if mem else f"RESULT params={n/1e6:.1f}M wall={secs}s")
PY
echo "${SIZE} DiT -> runs/${RUN_NAME}/checkpoint_final.pt"
