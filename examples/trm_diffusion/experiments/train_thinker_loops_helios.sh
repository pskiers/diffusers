#!/bin/bash -l
#SBATCH --job-name=claude_thinker_loops
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=22:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): PaTh thinker trained from scratch on the loop dataset, with the same frozen painter as the analysed
# PaTh (runs/pt_maze_final_painter). Same command as slurm_scripts/train_amaze/train_painter_thinker.sh stage 2.
# wandb offline (sync later with `wandb sync`).
set -euo pipefail
# One job: 40k steps took ~20.6 h for the original thinker; stops at 21.5 h with a checkpoint (resubmit to continue).
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p runs/claude_thinker_loops
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
RESUME=""; [ -f runs/claude_thinker_loops/checkpoint_final.pt ] && RESUME="++run.resume_from_checkpoint=runs/claude_thinker_loops/checkpoint_final.pt"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
  data.amaze_root="$PWD/runs/claude_loop_data/maze" \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  eval_callbacks=amaze train.num_steps=40000 train.max_seconds=77400 \
  run.output_dir=runs/claude_thinker_loops $RESUME
