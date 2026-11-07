#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%A_%a.err
#SBATCH --extra=FORCE_RM_TMPDIR

# ABOUTME: Job array for rebuttal exp 02 (docs/rebuttal/experiments/02_predictor_corrector.md): one DM eval per task
# ABOUTME: with a corrector after/before every predictor step (models/sampling.py), read from a manifest TSV row.
#
#   sbatch --array=1-45 slurm_scripts/wcss_pc_sweep.sh slurm_scripts/pc_sweep/phase1.tsv
#   check: DRY_RUN=1 SLURM_ARRAY_TASK_ID=<i> bash slurm_scripts/wcss_pc_sweep.sh <manifest>
#
# Manifest columns (whitespace separated, '#' lines ignored; row i = array task i):
#   dataset  extreme | hard | maze | queens | clevr  (the paper-table DM of each, sampled as in the paper)
#   variant  none | langevin | restart
#   K        corrector steps per predictor step (0 for none)
#   snr      Langevin signal-to-noise ratio r (- otherwise)
#   subset   full: the main-table eval set (seed 0, as the paper) | tune: a slice for picking r (seed 1)
# Output: results/pc_sweep/<dataset>/<variant>_K<K>[_r<snr>]_<subset>.json (+ .hydra/).

MANIFEST=$1
LINE=$(grep -v '^#' "$MANIFEST" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r DATASET VARIANT K SNR SUBSET <<< "$LINE"
[[ -n $SUBSET ]] || { echo "no manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2; exit 1; }

NAME=${VARIANT}_K$K; [[ $VARIANT == langevin ]] && NAME+=_r$SNR; NAME+=_$SUBSET
OUT=results/pc_sweep/$DATASET/$NAME
case $VARIANT in
  none)     CORR=() ;;
  langevin) CORR=(+sampling.corrector._target_=models.sampling.LangevinCorrector +sampling.corrector.n_steps=$K
                  +sampling.corrector.snr=$SNR) ;;
  restart)  CORR=(+sampling.corrector._target_=models.sampling.RestartCorrector +sampling.corrector.n_steps=$K) ;;
  *) echo "unknown variant $VARIANT" >&2; exit 1 ;;
esac
SEED=0; [[ $SUBSET == tune ]] && SEED=1
ENVV=(); CLEVR=0
SUDOKU=(eval.py experiment=mnist_unet_concat_painter_srm_1000t diffusion.num_train_timesteps=4800
        eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt eval.batch_size=384 train.compile=false)
case $DATASET in
  # Sudoku DMs (o1zzl5bf / vl9lpx06): in-training eval = 7680 puzzles, 20 DDPM steps, CFG 2; tune on 1024
  extreme) N=7680; [[ $SUBSET == tune ]] && N=1024
    ARGS=("${SUDOKU[@]}" data=mnist_sudoku +checkpoint=runs/mnist_unet_concat_painter_extr_4800t/checkpoint_final.pt
          eval_callbacks.0.num_samples=$N +eval_seed=$SEED +metrics_path=$OUT.json) ;;
  hard) N=7680; [[ $SUBSET == tune ]] && N=1024
    ARGS=("${SUDOKU[@]}" +checkpoint=runs/mnist_unet_concat_painter_srm_4800t/checkpoint_final.pt
          eval_callbacks.0.num_samples=$N +eval_seed=$SEED +metrics_path=$OUT.json) ;;
  # AMAZE DMs (cb7sx7no / lhj4v8rg): the paper's eval script, 5 samples per board; Maze tune = size 8 only (400 boards)
  maze)
    [[ $SUBSET == tune ]] && ENVV=(MAZE_SCALES=8 MAZE_OOD_SCALES=)
    ARGS=(experiments/amaze_generate_and_calculate_metrics.py experiment=amaze_dit_maze
          +checkpoint=runs/dit_maze_final/checkpoint_final.pt +task=maze +data_root=data/amaze +samples_per_puzzle=5
          run.wandb_project=null +sample_seed=$SEED +out_json=$OUT.json) ;;
  queens)
    ENVV=(QUEEN_OOD_SCALES=)  # no Queens n12 parquet on WCSS
    ARGS=(experiments/amaze_generate_and_calculate_metrics.py experiment=amaze_dit_queens
          +checkpoint=runs/queens_dit_baseline/checkpoint_final.pt +task=queens +data_root=data/amaze
          +samples_per_puzzle=5 run.wandb_project=null +sample_seed=$SEED +out_json=$OUT.json) ;;
  # CLEVR DM 236M (4ia9nlq6), its best eval (spatial 0.688 @40k): 1500 scenes, 20 DDPM steps, CFG 2; tune on 256
  clevr) N=1500; [[ $SUBSET == tune ]] && N=256
    ARGS=(eval.py experiment=clevr_unet_concat_painter "backbone.block_out_channels=[256,512,512]"
          backbone.layers_per_block=3 backbone.cross_attention_dim=1024
          +checkpoint=runs/clevr_unet_concat_painter_very_big/checkpoint_step-40000.pt data.clevr_root=\$SCRATCH/clevr
          eval_callbacks.1.n_samples=$N +eval_seed=$SEED +metrics_path=$OUT.json)
    CLEVR=1 ;;
  *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac
CMD=(env "${ENVV[@]}" python "${ARGS[@]}" "${CORR[@]}" hydra.run.dir=$OUT.hydra)
if [[ -n $DRY_RUN ]]; then echo "${CMD[@]}"; exit 0; fi

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
source $ROOT/trm_torch26_env/bin/activate
cd $ROOT/examples/trm_diffusion
export SCRATCH=$TMPDIR
mkdir -p results/pc_sweep/$DATASET
echo "$(date -Is) ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID $(hostname) git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l): ${CMD[*]}"
if (( CLEVR )); then
  src=$ROOT/examples/trm_diffusion/data/clevr/CLEVR_v1.0 dst=$SCRATCH/clevr/CLEVR_v1.0 t0=$SECONDS
  mkdir -p $dst/images && cp -r $src/scenes $dst/ && cp -r $src/images/val $dst/images/ \
    || { echo "CLEVR copy to $dst failed" >&2; exit 1; }
  echo "CLEVR val copied to $dst in $(( SECONDS - t0 )) s"
  CMD=("${CMD[@]//\$SCRATCH/$SCRATCH}")
fi
t0=$SECONDS
"${CMD[@]}"
echo "exit $? after $(( SECONDS - t0 )) s"
