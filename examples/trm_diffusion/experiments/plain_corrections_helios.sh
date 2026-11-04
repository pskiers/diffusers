#!/bin/bash -l
#SBATCH --job-name=claude_plain_corr
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): plain TRM version of probe_corrections_t70.png. Usage: sbatch plain_corrections_helios.sh <plain checkpoint> <out_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$2"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
python experiments/plain_corrections.py "$1" data/amaze/maze/square/n8_test.parquet "$2"
