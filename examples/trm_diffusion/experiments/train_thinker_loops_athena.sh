#!/bin/bash -l
#SBATCH --job-name=claude_thinker_loops
#SBATCH --account=plgdiffusion3-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): Athena twin of train_thinker_loops_helios.sh: PaTh thinker on the loop dataset.
# Frozen painter runs/pt_maze_final_painter. batch 96 x 8 accumulation (= 768, as 192 x 4 on Helios) for 40 GB A100s.
# Stops after 35 h with a checkpoint; resubmit to continue. wandb offline.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p runs/claude_thinker_loops
module load Python/3.10.4 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_athena_venv/bin/activate"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
RESUME=""; [ -f runs/claude_thinker_loops/checkpoint_final.pt ] && RESUME="++run.resume_from_checkpoint=runs/claude_thinker_loops/checkpoint_final.pt"
srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
  data.amaze_root="$PWD/runs/claude_loop_data/maze" \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  train.batch_size=96 train.gradient_accumulation_steps=8 \
  eval_callbacks=amaze train.num_steps=40000 train.max_seconds=126000 \
  run.output_dir=runs/claude_thinker_loops $RESUME
