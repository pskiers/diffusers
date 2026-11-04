#!/bin/bash -l
#SBATCH --job-name=claude_subtoken
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): interp_subtoken_probe.py runs. Usage: sbatch --time=... experiments/subtoken_helios.sh <out_root> <runs...>
#   runs: square_plain | square_noisy | circle_plain | circle_path
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
OUT=$1; shift
BASE="experiment=amaze_thinker_v2_controlnet painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
  +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt"
PL="+interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint"
CIRC="+interprebability_config.shape=circle +interprebability_config.maze_sizes=[5,7,9,11]"
for r in "$@"; do
  case $r in
    square_plain) A="$PL=runs/claude_plain_trm/checkpoint_final.pt";;
    square_noisy) A="$PL=runs/claude_plain_trm_noisy/checkpoint_final.pt";;
    circle_plain) A="$CIRC $PL=runs/claude_plain_trm/checkpoint_final.pt";;
    circle_path)  A="$CIRC";;
  esac
  mkdir -p $OUT/$r
  echo "##### $r $(date)"
  python experiments/interp_subtoken_probe.py $BASE $A +interprebability_config.out_dir=$OUT/$r hydra.run.dir=$OUT/$r/hydra
done
