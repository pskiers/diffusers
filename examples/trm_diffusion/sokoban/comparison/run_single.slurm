#!/bin/bash
#SBATCH -A plgdyplomancipw2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00

module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgmgrzanka/reasoning_diffusion_env

echo "Starting task: $SLURM_JOB_NAME"
echo "Task ID: $SLURM_JOB_ID"

cd ../..

echo "--- Training ---"
accelerate launch train.py "$@"

echo "--- Sampling ---"
accelerate launch sample.py checkpoint_step=latest num_samples=10000 sample_batch_size=250 "$@"

echo "--- Logging results to W&B ---"
python -c "
import sys
import glob
import json
import wandb
from pathlib import Path

args = sys.argv[1:]
run_name = None
for arg in args:
    if arg.startswith('output_dir='):
        run_name = arg.split('=')[1]
        break

if run_name:
    samples_dir = Path(run_name) / 'samples'
    metrics_files = glob.glob(str(samples_dir / 'sokoban_metrics_rank*.json'))

    if metrics_files:
        rank0_file = samples_dir / 'sokoban_metrics_rank0.json'
        file_to_read = rank0_file if rank0_file.exists() else metrics_files[0]

        with open(file_to_read, 'r') as f:
            metrics = json.load(f)

        wandb.init(project='small-llm-diffusion', name=run_name)

        image_files = sorted(glob.glob(str(samples_dir / '*.png')))
        num_images_to_log = 10

        if image_files:
            wandb_images = [wandb.Image(img_path) for img_path in image_files[:num_images_to_log]]
            metrics['generated_samples'] = wandb_images

        wandb.log(metrics)
        wandb.finish()

        logged_img_count = len(wandb_images) if image_files else 0
        print(f'Metrics and {logged_img_count} images added to w&b')
    else:
        print(f'No metrics in {samples_dir}')
else:
    print('No output_dir in script args')
" "$@"

echo "Task finished."
