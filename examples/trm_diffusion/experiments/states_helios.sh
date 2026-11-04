#!/bin/bash -l
#SBATCH --job-name=claude_states
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
# TEMPORARY (Claude): interp_compare_states.py (what z_L / z_H encode: plain TRM variants vs PaTh thinker) on helios,
# the titan run (states_titan.sh) was lost. Results in runs/claude_states/n8_random and n11_longest.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load Python/3.11.5 CUDA/12.4.0 cuDNN/9.2.1.18-CUDA-12.4.0
source "${SCRATCH}/trm_helios_venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/net/software/aarch64/el9/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}"
export PLAIN_CKPTS=plain:runs/claude_plain_trm/checkpoint_final.pt,plain4k:runs/claude_plain_trm/checkpoint_step4000.pt,noisy:runs/claude_plain_trm_noisy/checkpoint_final.pt
for spec in "8 random 0" "11 longest 10"; do
  set -- $spec; OUT=runs/claude_states/n$1_$2; mkdir -p $OUT
  python experiments/interp_compare_states.py experiment=amaze_thinker_v2_controlnet \
    painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
    +interprebability_config.model_kind=path +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
    +interprebability_config.out_dir=$OUT ++interprebability_config.data_parquet=data/amaze/maze/square/n$1_test.parquet \
    +interprebability_config.maze_size=$1 +interprebability_config.side_pick=$2 +interprebability_config.min_side=$3 \
    hydra.run.dir=$OUT/hydra_path > $OUT/log.txt 2>&1
  echo "n$1_$2 exit $?"
done
echo ALLDONE > runs/claude_states/done
