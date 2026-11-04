#!/bin/bash -l
#SBATCH --job-name=claude_loops_hm
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): probe_corrections-style heatmaps on the loop mazes, plain TRM and PaTh.
# Usage: sbatch loops_heatmaps_helios.sh <plain checkpoint> <out_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$2"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
COMMON="experiment=amaze_thinker_v2_controlnet painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
  +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet +interprebability_config.out_dir=$2"
python experiments/loops_heatmaps.py $COMMON +interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint="$1" \
  +interprebability_config.checkpoint=none +interprebability_config.batch_size=132
python experiments/loops_heatmaps.py $COMMON +interprebability_config.model_kind=path \
  +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt +interprebability_config.batch_size=44
