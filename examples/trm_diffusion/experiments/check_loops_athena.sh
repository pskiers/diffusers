#!/bin/bash -l
#SBATCH --job-name=claude_loops
#SBATCH --account=plgdiffusion3-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): loop test (check_loops.py) on the plain TRM (40k) and PaTh. Usage: sbatch check_loops_athena.sh <plain checkpoint> <out_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$2"
module load Python/3.10.4 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_athena_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"

COMMON="experiment=amaze_thinker_v2_controlnet painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
  +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet +interprebability_config.out_dir=$2
  +interprebability_config.num_boards=100"
python experiments/check_loops.py $COMMON +interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint="$1" \
  +interprebability_config.checkpoint=none +interprebability_config.batch_size=132
python experiments/check_loops.py $COMMON +interprebability_config.model_kind=path \
  +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt +interprebability_config.batch_size=33
