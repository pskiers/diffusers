#!/bin/bash -l
#SBATCH --job-name=claude_loops2
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
# TEMPORARY (Claude): loop mazes - drawings with images (check_loops.py) and call-by-call traces (check_loops_calls.py),
# plain TRM (40k) and PaTh. Usage: sbatch check_loops2_helios.sh <plain checkpoint> <PaTh prune_mech2.pt> <out_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p "$3"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
COMMON="experiment=amaze_thinker_v2_controlnet painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
  +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet +interprebability_config.out_dir=$3
  +interprebability_config.num_boards=100"
PLAIN="+interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint=$1 +interprebability_config.checkpoint=none"
PATH_="+interprebability_config.model_kind=path +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt"
python experiments/check_loops_calls.py $COMMON $PLAIN +interprebability_config.batch_size=132
python experiments/check_loops_calls.py $COMMON $PATH_ +interprebability_config.ref_pt=$2 +interprebability_config.batch_size=33
python experiments/check_loops.py $COMMON $PLAIN +interprebability_config.batch_size=132
python experiments/check_loops.py $COMMON $PATH_ +interprebability_config.batch_size=33
