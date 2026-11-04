#!/bin/bash -l
#SBATCH --job-name=claude_plain_variant
#SBATCH --account=plgdiffusion3-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): Athena twin of train_plain_variant_helios.sh. Micro-batch 96 x 8 (= 768, as 192 x 4 on Helios)
# for 40 GB A100s. Usage: sbatch train_plain_variant_athena.sh <out_dir> <train parquet> <test parquet> <flags...>
# Resumable from <out_dir>/checkpoint_last.pt
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT=$1; TRAIN=$2; TEST=$3; shift 3; mkdir -p "$OUT"
module load Python/3.10.4 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_athena_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
srun python experiments/train_plain_trm.py --train "$TRAIN" --test "$TEST" --out_dir "$OUT" \
  --micro_batch 96 --n_micro 8 --max_minutes 585 --workers 8 "$@"
