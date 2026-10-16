#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:4
#SBATCH --extra=FORCE_RM_TMPDIR

# One tuning run, submitted by tuning/driver.py: train_trm.py with the
# overrides listed in $1 (one per line) at an effective batch of 768.
# Environment (from slurm_scripts/wcss_tune.sh): TUNE_BATCH = per-GPU batch;
# TUNE_DATA = optional dataset copied to $TMPDIR, read as ${oc.env:DATA_LOCAL}.

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
source $ROOT/trm_torch26_env/bin/activate
export HF_HOME=$ROOT/examples/trm_diffusion/cache_dir
export TOKENIZERS_PARALLELISM=false
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd $ROOT/examples/trm_diffusion

if [ -n "$TUNE_DATA" ]; then
  cp -r "$TUNE_DATA" "$TMPDIR/" || { echo "dataset copy failed" >&2; exit 1; }
  export DATA_LOCAL=$TMPDIR/$(basename "$TUNE_DATA")
fi

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
NGPUS=$((SLURM_NNODES * GPUS_PER_NODE))
BATCH=${TUNE_BATCH:-48}
if [ $((768 % (BATCH * NGPUS))) -ne 0 ]; then
  echo "768 is not a multiple of TUNE_BATCH=$BATCH x $NGPUS GPUs" >&2; exit 1
fi
ACCUM=$((768 / (BATCH * NGPUS)))
MULTI_GPU=$([ "$NGPUS" -gt 1 ] && echo --multi_gpu)

mapfile -t OVERRIDES < "$1"
CMD=$(printf '%q ' train_trm.py "${OVERRIDES[@]}" train.batch_size=$BATCH train.gradient_accumulation_steps=$ACCUM)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c "accelerate launch $MULTI_GPU \
  --num_machines $SLURM_NNODES --num_processes $NGPUS --machine_rank \$SLURM_NODEID \
  --main_process_ip $MASTER_ADDR --main_process_port 29500 $CMD"
