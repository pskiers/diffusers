#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%A_%a.err
#SBATCH --extra=FORCE_RM_TMPDIR

# ABOUTME: Job array for rebuttal exp 07 analysis 2: PaTh on Maze / Queens / CLEVR with reasoning stopped after the first
# ABOUTME: k of the denoising steps (models.sampling.SteeringReusePredictor), scored by each task's paper eval.
#
#   sbatch --array=1-21 slurm_scripts/wcss_stop_at_k.sh slurm_scripts/stop_at_k/manifest.tsv
#   check: DRY_RUN=1 SLURM_ARRAY_TASK_ID=<i> bash slurm_scripts/wcss_stop_at_k.sh <manifest>
#
# Manifest columns: task (maze | queens | clevr), reason_frac (fraction of the 20 steps reasoned; 1.0 = the paper
# sampler). Output: results/stop_at_k/<task>/reason<frac>.json (+ .hydra/). AMAZE JSONs carry per-size tables.

MANIFEST=$1
LINE=$(grep -v '^#' "$MANIFEST" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r TASK FRAC <<< "$LINE"
[[ -n $FRAC ]] || { echo "no manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2; exit 1; }

OUT=results/stop_at_k/$TASK/reason$FRAC
REUSE=(sampling.predictor._target_=models.sampling.SteeringReusePredictor +sampling.predictor.reason_frac=$FRAC)
ENVV=(); CLEVR=0
AMAZE_PT=(experiments/amaze_generate_and_calculate_metrics.py experiment=amaze_thinker_v2_controlnet data.cell_size=12
          thinker.seq_len=144 translator.grid=12 +data_root=data/amaze +samples_per_puzzle=5 run.wandb_project=null
          +sample_seed=0 +out_json=$OUT.json)
case $TASK in
  # pt_maze_final_thinker (m10k4xhx) / pt_queens_final_thinker (jvvoqptb): the paper eval (standard scorer)
  maze)
    ARGS=("${AMAZE_PT[@]}" data.amaze_root=data/amaze/maze +task=maze
          painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
          +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt) ;;
  queens)
    ENVV=(QUEEN_OOD_SCALES=)  # no Queens n12 parquet on WCSS
    ARGS=("${AMAZE_PT[@]}" data.amaze_root=data/amaze/queens +task=queens
          painter.checkpoint=runs/queens_painter_v2/checkpoint_final.pt
          +checkpoint=runs/pt_queens_final_thinker/checkpoint_final.pt) ;;
  # clevr_thinker_v1_centroidmaskfused_x0hint_t80_sliced_biggrid_forceuncond (0il6x2cq), last checkpoint (90k;
  # evals 0.884 @90k, 0.897 @93k): 1500 scenes, seed 0
  clevr)
    ARGS=(eval.py experiment=clevr_thinker_v1_controlnet painter.checkpoint=runs/clevr_unet_big/checkpoint_step-80000.pt
          "backbone.block_out_channels=[128,256,512]"
          condition_encoder=object_feature_v1_centroidmask_presence_fused_x0hint condition_encoder.threshold=80
          data.include_centroid_mask=true data.centroid_mask_presence_only=true translator=controlnet_clevr_v1_sliced
          "translator.painter_channels=[128,256,512]" thinker.seq_len=266 +train.force_unconditional_painter=true
          +checkpoint=runs/clevr_thinker_v1_centroidmaskfused_x0hint_t80_sliced_biggrid_forceuncond/checkpoint_step-90048.pt
          data.clevr_root=\$SCRATCH/clevr eval_callbacks.1.n_samples=1500 +eval_seed=0 +metrics_path=$OUT.json)
    CLEVR=1 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
CMD=(env "${ENVV[@]}" python "${ARGS[@]}" "${REUSE[@]}" hydra.run.dir=$OUT.hydra)
if [[ -n $DRY_RUN ]]; then echo "${CMD[@]}"; exit 0; fi

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
source $ROOT/trm_torch26_env/bin/activate
cd $ROOT/examples/trm_diffusion
export SCRATCH=$TMPDIR
mkdir -p results/stop_at_k/$TASK
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
