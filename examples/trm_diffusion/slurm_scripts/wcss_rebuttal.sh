#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:2
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%j.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%j.err
#SBATCH --extra=FORCE_RM_TMPDIR

# ABOUTME: Cluster launcher for the ICLR rebuttal runs (docs/rebuttal/experiments/), one run per job on 1 or 2 H100s.
# ABOUTME: DRY_RUN=1 prints the exact command (run name, overrides) without setting up the environment or training.
#
#   sbatch slurm_scripts/wcss_rebuttal.sh <run> [extra train_trm.py overrides...]
#   1-GPU runs (marked below): sbatch --gres=gpu:hopper:1 slurm_scripts/wcss_rebuttal.sh <run>
#   resume: sbatch slurm_scripts/wcss_rebuttal.sh <run> run.resume_from_checkpoint=runs/<output dir>/checkpoint_step-N.pt
#   check:  DRY_RUN=1 bash slurm_scripts/wcss_rebuttal.sh <run>
#
# Runs:
#   extreme_nsup<N>  (04_trained_recursion_budget) EXTREME thinker trained at its own recursion budget n_sup=<N>
#                    (H 3, L 6), otherwise the recipe of mnist_painter_halt_pairhead_singlestage_nsup16 (y86cr9av:
#                    thinker_act + pair halt head, wd 1.0), 160k steps (the paper's), batch 384 x 2 GPUs = 768.
#                    Output/W&B name: mnist_painter_halt_pairhead_singlestage_nsup<N>.
#   amaze_trm_<task> (03_trm_baselines) TRM as the diffusion backbone, recipe of mnist_trm_diffusion_backbone (vu78kfn4),
#                    on AMAZE <task> = maze | queens; 1 GPU, batch 256 (as vu78kfn4), 100k steps (each batch
#                    makes n_sup = 16 optimizer steps), CFG dropout 0.1. Output/W&B name:
#                    amaze_trm_diffusion_backbone_<task>.
#   hard_trm         (03_trm_baselines) the same TRM denoiser on Sudoku HARD, trained on 0-26 givens like PaTh HARD
#                    (sudoku_hard_trm_0_26_x0_hint_repro, pvpx8kbs), CFG dropout 0.1; 1 GPU, batch 256.
#                    Output/W&B name: mnist_trm_diffusion_backbone_hard.
#                    The TRM denoiser runs on 1 GPU: on 2 GPUs one rank skips the DDP wrap and the job hangs
#                    (LOG.md, 2026-11-07).

RUN=$1
NGPU=2
XARGS=("${@:2}")  # extra train_trm.py overrides

case $RUN in
  extreme_nsup[0-9]*)
    N=${RUN#extreme_nsup}
    OUT=mnist_painter_halt_pairhead_singlestage_nsup$N
    ARGS=(experiment=mnist_thinker_v1_controlnet painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt
          condition_encoder=x0_hint_v1 condition_encoder.threshold=80 +condition_encoder.enabled=false
          condition_encoder.inner.with_timestep_emb=false mode=thinker_act thinker.with_halt_head=true
          thinker.n_sup=$N train.act_halt_threshold=0.0 train.act_continue_bias_init=-1.0
          +train.act_halt_label_margin=0.0001 train.act_halt_exploration_prob=0.1 train.batch_size=384
          train.gradient_accumulation_steps=1 train.num_steps=160000 eval.log_every=200) ;;
  amaze_trm_maze|amaze_trm_queens)
    TASK=${RUN#amaze_trm_}
    OUT=amaze_trm_diffusion_backbone_$TASK
    CB=amaze; [[ $TASK == queens ]] && CB=amaze_queens
    ARGS=(experiment=amaze_trm_diffusion_backbone data.amaze_root=data/amaze/$TASK eval_callbacks=$CB
          train.batch_size=256)
    NGPU=1 ;;
  hard_trm)
    OUT=mnist_trm_diffusion_backbone_hard
    ARGS=(experiment=mnist_trm_diffusion_backbone data=mnist_sudoku_srm "data.train_dataset.given_cells_range=[0,26]"
          train.batch_size=256 train.gradient_accumulation_steps=1 train.grad_clip=256 train.cfg_prob=0.1)
    NGPU=1 ;;
  *)
    echo "unknown run '$RUN' (see the header of $0)" >&2; exit 1 ;;
esac

if (( NGPU > 1 )); then LAUNCH=(accelerate launch --multi_gpu --num_processes $NGPU); else LAUNCH=(python); fi
CMD=("${LAUNCH[@]}" train_trm.py "${ARGS[@]}" run.output_dir=runs/$OUT "${XARGS[@]}")
if [[ -n $DRY_RUN ]]; then echo "${CMD[@]}"; exit 0; fi

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
source /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/trm_torch26_env/bin/activate

ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
export SCRATCH=$TMPDIR
cd $ROOT/examples/trm_diffusion
echo "$(date -Is) $SLURM_JOB_ID $(hostname) git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l): ${CMD[*]}"
"${CMD[@]}"
