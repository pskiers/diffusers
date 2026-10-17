#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%j.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%j.err
#SBATCH --extra=FORCE_RM_TMPDIR

# MiniT2I-style painter + thinker (models/mmdit_painter.py) on Sudoku and CLEVR,
# one run per job, on one H100 with the configs' own batch sizes:
#   sbatch slurm_scripts/wcss_mmdit.sh <run>
# Stage 1 (painters):  mnist_painter  clevr_painter
# Stage 2 (thinkers, once the painter's checkpoint_final.pt exists):
#   mnist_thinker  mnist_thinker_steercond  mnist_thinker_cnn
#   clevr_thinker  clevr_thinker_steercond
#
# Measured on one H100 (one training micro-batch, torch.compile, peak memory):
#   mnist_painter   256     19.0 GB  0.19 s   -> ~5 h  (100k steps)
#   clevr_painter   256     33.8 GB  0.33 s   -> ~9 h  (100k steps)
#   mnist_thinker   384x2   37.0 GB  4.75 s   -> ~33 h (cnn: 29.9 GB, 3.30 s -> ~23 h)
#   clevr_thinker   256x3   32.2 GB  6.02 s   -> ~47 h
# Sampling during evals costs ~4.0 s (Sudoku, batch 384) / ~2.6 s (CLEVR, batch
# 256) per Euler step, so evals use fewer samples, less often than the defaults.

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
source /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/trm_torch26_env/bin/activate

ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
export SCRATCH=$TMPDIR
export HF_HOME=$ROOT/examples/trm_diffusion/cache_dir
export TOKENIZERS_PARALLELISM=false
cd $ROOT/examples/trm_diffusion

RUN=$1
MNIST_PAINTER=runs/mnist_mmdit_painter/checkpoint_final.pt
CLEVR_PAINTER=runs/clevr_mmdit_painter/checkpoint_final.pt
MNIST_EVAL="eval.eval_every=20000 eval_callbacks.0.num_samples=1024"
CLEVR_EVAL="eval.eval_every=15000 eval_callbacks.1.n_samples=512"
train() { python train_trm.py data.num_workers=8 run.output_dir=runs/$RUN "$@"; }

case $RUN in
  mnist_painter)
    RUN=mnist_mmdit_painter
    train experiment=mnist_mmdit_painter eval.eval_every=10000 eval_callbacks.0.num_samples=1024 ;;
  clevr_painter)
    RUN=clevr_mmdit_painter
    train experiment=clevr_mmdit_painter eval.eval_every=10000 eval_callbacks.1.n_samples=512 ;;
  mnist_thinker)  # as T2I: reads the painter's hidden states, steers the image stream
    RUN=mnist_thinker_mmdit
    train experiment=mnist_thinker_mmdit painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_steercond)  # + steers the puzzle (condition) stream
    RUN=mnist_thinker_mmdit_steercond
    train experiment=mnist_thinker_mmdit translator.steer_cond=true painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn)  # ladder step: the working version's CNN encoder + 11 logits
    RUN=mnist_thinker_mmdit_cnn
    train experiment=mnist_thinker_mmdit_cnn painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  clevr_thinker)
    RUN=clevr_thinker_mmdit
    train experiment=clevr_thinker_mmdit painter.checkpoint=$CLEVR_PAINTER $CLEVR_EVAL ;;
  clevr_thinker_steercond)
    RUN=clevr_thinker_mmdit_steercond
    train experiment=clevr_thinker_mmdit translator.steer_cond=true painter.checkpoint=$CLEVR_PAINTER $CLEVR_EVAL ;;
  *)
    echo "unknown run '$RUN' (see the header of $0)" >&2; exit 1 ;;
esac
