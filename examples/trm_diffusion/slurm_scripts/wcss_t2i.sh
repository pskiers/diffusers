#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:4
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%j.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%j.err
#SBATCH --extra=FORCE_RM_TMPDIR

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
source /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/trm_torch26_env/bin/activate

# project paths
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
export SCRATCH=$TMPDIR
export HF_HOME=$ROOT/examples/trm_diffusion/cache_dir
export JUDGE_ENV=$ROOT/llm_judge_env
export COREBENCH_JUDGE_PYTHON=$ROOT/examples/trm_diffusion/slurm_scripts/wcss_judge_python.sh
export DATA_T2I=$ROOT/examples/trm_diffusion/data/std120k
DATA_LOCAL=$TMPDIR/std120k
# project flags
export TOKENIZERS_PARALLELISM=false
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# wcss training cfg
GPUS_PER_NODE=4
NGPUS=$((SLURM_NNODES * GPUS_PER_NODE))
BATCH=48
ACCUM=$((768 / (BATCH * NGPUS)))
WORKERS=8

# multi node job launcher
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
launch() {
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c "accelerate launch \
    --multi_gpu --num_machines $SLURM_NNODES --num_processes $NGPUS --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT $*"
}


# --- PREQUISITES ---
cd $ROOT/examples/trm_diffusion
cp -r $DATA_T2I $TMPDIR/ || { echo "dataset copy failed" >&2; exit 1; }


# --- BUILDING ENV AND DATA ---

# # build t2i dataset
# python -m datasets.std120k_dataset --root $DATA_T2I --workers $SLURM_CPUS_PER_TASK

# # build llm judge env (needs Python >= 3.11, the Python/3.10.4 module breaks vLLM)
# (module purge && module load Python/3.12.3-GCCcore-13.3.0 && PYTHON=python3.12 bash scripts/setup_judge_env.sh $JUDGE_ENV)

# # download llm judge model
# $COREBENCH_JUDGE_PYTHON -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-VL-32B-Thinking'))"


# --- BENCHMARKING ---

# # benchmark training/evaluation speed + batch sizes
# launch experiments/benchmark.py experiment=std120k_thinker_minit2i \
#   data.root=$DATA_LOCAL data.num_workers=$WORKERS \
#   train.batch_size=$BATCH train.gradient_accumulation_steps=$ACCUM \
#   "'+bench.train_batch_sizes=[16,24,32,48,64,96]'" "'+bench.sample_batch_sizes=[8,16,32,64,128]'" \
#   run.output_dir=runs/benchmark_${NGPUS}gpu

# # benchmark the judges
# #   2 judges x 2 GPUs per node
# launch experiments/benchmark.py experiment=std120k_thinker_minit2i data.root=$DATA_LOCAL \
#   +bench.skip_training=true +bench.corebench_prompts_per_dimension=8 run.output_dir=runs/benchmark_judge_2x2
# #   1 judge x 4 GPUs per node
# launch experiments/benchmark.py experiment=std120k_thinker_minit2i data.root=$DATA_LOCAL \
#   +bench.skip_training=true +bench.corebench_prompts_per_dimension=8 run.output_dir=runs/benchmark_judge_1x4 \
#   eval_callbacks.1.judges_per_node=1 eval_callbacks.1.judge_tensor_parallel_size=4
# #   2 judges x 2 GPUs per node, 4096 requests per vLLM call instead of the default 1024
# launch experiments/benchmark.py experiment=std120k_thinker_minit2i data.root=$DATA_LOCAL \
#   +bench.skip_training=true +bench.corebench_prompts_per_dimension=8 run.output_dir=runs/benchmark_judge_2x2_b4096 \
#   "'eval_callbacks.1.judge_args=[--batch-size,4096]'"


# --- TUNING ---

TUNE="experiment=std120k_thinker_minit2i data.root=$DATA_LOCAL data.num_workers=$WORKERS \
  train.batch_size=$BATCH train.gradient_accumulation_steps=$ACCUM data.train_dataset.holdout=512 \
  eval_callbacks.1.judge_python=null eval_callbacks.0.num_log_images=4"
# Two optimizers: thinker.optim_cfg (the TRM) and painter.optim_cfg (condition encoder +
# translator; train/lr logs this one). The zero-init translator gates all steering, so both
# get the same lr. Its cosine decay is off (ratio 1): over a short run it would decay fully
# instead of staying ~flat as in the first 10% of the real run.
lr() { echo "thinker.optim_cfg.lr=$1 painter.optim_cfg.lr=$1 painter.optim_cfg.lr_min_ratio=1.0"; }

# # 1) LR range test (~30 min): lr ramps linearly 0 -> 3e-3 over 100 batches (1600 optimizer
# #    steps), then stops. In wandb, plot train loss against lr: note where it starts rising
# #    or spiking. If that is already in the first ~10 batches, rerun with a lower peak lr.
# launch train_trm.py $TUNE $(lr 3e-3) thinker.optim_cfg.warmup_steps=1600 painter.optim_cfg.warmup_steps=1600 \
#   train.num_steps=1600 eval.log_every=16 eval.eval_every=100000 eval.save_every=100000 \
#   run.output_dir=runs/tune_lr_range

# # 2) short runs (~2.6 h each): 500 batches = the first 10% of the real run (constant lr
# #    after warmup). LRs ~1/3, 1/10, 1/30 of where the range test blew up; keep the current
# #    1e-4 as a reference. Pick the best final val/diff_loss that is still falling (ties: lower lr)
# for LR in 1e-3 3e-4 1e-4; do
#   launch train_trm.py $TUNE $(lr $LR) train.num_steps=8000 \
#     eval.eval_every=1000 eval.save_every=100000 run.output_dir=runs/tune_lr$LR
# done


# --- TRAINING ---

# train PaTh (CoReBench during training with 1 image per prompt instead of 4)
launch train_trm.py experiment=std120k_thinker_minit2i \
  data.root=$DATA_LOCAL data.num_workers=$WORKERS \
  train.batch_size=$BATCH train.gradient_accumulation_steps=$ACCUM \
  eval_callbacks.1.samples_per_prompt=1 \
  run.output_dir=runs/std120k_thinker_minit2i

# eval PaTh
launch eval.py experiment=std120k_thinker_minit2i \
  data.root=$DATA_LOCAL +checkpoint=runs/std120k_thinker_minit2i/checkpoint_final.pt \
  +metrics_path=runs/std120k_thinker_minit2i/corebench_final.json eval_callbacks.1.every_n_steps=0


# # train PaTh (CoReBench during training with 1 image per prompt instead of 4)
# launch train_trm.py experiment=std120k_thinker_minit2i painter=minit2i_b16_pretrain_frozen \
#   data.root=$DATA_LOCAL data.num_workers=$WORKERS \
#   train.batch_size=$BATCH train.gradient_accumulation_steps=$ACCUM \
#   eval_callbacks.1.samples_per_prompt=1 \
#   run.output_dir=runs/std120k_thinker_minit2i_pretrain

# # eval PaTh
# launch eval.py experiment=std120k_thinker_minit2i painter=minit2i_b16_pretrain_frozen \
#   data.root=$DATA_LOCAL +checkpoint=runs/std120k_thinker_minit2i_pretrain/checkpoint_final.pt \
#   +metrics_path=runs/std120k_thinker_minit2i_pretrain/corebench_final.json eval_callbacks.1.every_n_steps=0
