#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-cpu-short
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/tune_start_%j.out

# Start a staged nuisance-hyperparameter tuning session (tuning/driver.py):
#   sbatch slurm_scripts/wcss_tune.sh
# Set FAMILY / CHANGE / FROM and the TRAIN command below. The driver submits
# each stage's runs (slurm_scripts/wcss_tune_job.sh) plus an advance job
# (wcss_tune_advance.sh) that scores them and submits the next stage; the
# result goes to tuning/best/$FAMILY.json, which later sessions start from.
#   progress:  cat runs/tune/$FAMILY/status.txt   (rewritten at every stage)
#   resume:    sbatch slurm_scripts/wcss_tune_advance.sh $FAMILY   (e.g. after fixing a failed stage)
#
# CHANGE:
#   cold   nothing tuned yet: range test, lr, r, lr refine, tau
#   small  scientific change, same architecture (e.g. n_sup, painter checkpoint): lr only,
#          around FROM's best
#   large  architecture change: range test, lr re-centered from FROM's lr / divergence lr, r
#
# Cost of one T2I run on 2 nodes (8 H100, ~41 samples/s): range test ~0.5 h, short (10% of
# the 80k-step run) ~2.6 h, medium (30%) ~7.8 h. Runs of a stage go in parallel; wall time:
#   cold  ~16 h (11 runs, ~84 node-h)   small ~2.6 h (3 runs, ~16 node-h)
#   large ~6 h  (6 runs, ~27 node-h)    + queueing, + up to 2 edge extensions per sweep

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
source $ROOT/trm_torch26_env/bin/activate
cd $ROOT/examples/trm_diffusion

export TUNE_DATA=$ROOT/examples/trm_diffusion/data/std120k
export TUNE_BATCH=48

FAMILY=t2i_pretrain
CHANGE=cold
FROM=
NODES=2
TAU_FRAC=0.25  # cold only: starting weight-decay timescale / run length (sudoku winners: 1/6-1/2)
TRAIN=(
  experiment=std120k_thinker_minit2i
  painter=minit2i_b16_pretrain_frozen
  'data.root=${oc.env:DATA_LOCAL}'
  data.num_workers=8
  data.train_dataset.holdout=512
  '~eval_callbacks'
)

python -m tuning.driver start --family "$FAMILY" --change "$CHANGE" ${FROM:+--from "$FROM"} \
  --nodes "$NODES" --tau-frac "$TAU_FRAC" -- "${TRAIN[@]}"
