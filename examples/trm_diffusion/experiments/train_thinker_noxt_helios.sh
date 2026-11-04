#!/bin/bash -l
#SBATCH --job-name=claude_thinker_noxt
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
# TEMPORARY (Claude): PaTh thinker from scratch WITHOUT the noisy image: its encoder reads only the maze
# (SpatialConditionEncoder, 3 channels, no timestep). Painter frozen (runs/pt_maze_final_painter), original data,
# otherwise the stage-2 command of slurm_scripts/train_amaze/train_painter_thinker.sh. wandb offline.
set -euo pipefail
# One job: 40k steps took ~20.6 h for the original thinker; stops at 21.5 h with a checkpoint (resubmit to continue).
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p runs/claude_thinker_noxt
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
RESUME=""; [ -f runs/claude_thinker_noxt/checkpoint_final.pt ] && RESUME="++run.resume_from_checkpoint=runs/claude_thinker_noxt/checkpoint_final.pt"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
srun python train_trm.py experiment=amaze_thinker_v2_controlnet \
  data.amaze_root=/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion/data/amaze/maze \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  condition_encoder._target_=models.condition_encoders.SpatialConditionEncoder condition_encoder.in_channels=3 \
  condition_encoder.with_timestep_emb=false '~condition_encoder.noisy_dropout_p_max' '~condition_encoder.num_train_timesteps' \
  eval_callbacks=amaze train.num_steps=40000 train.max_seconds=77400 \
  run.output_dir=runs/claude_thinker_noxt $RESUME
