#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
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
#   mnist_thinker  mnist_thinker_steer1  mnist_thinker_cnn  mnist_thinker_steercond
#   clevr_thinker  clevr_thinker_steercond
# Oracle-screen follow-ups (Sudoku, steering from block 1):
#   mnist_thinker_uncond_tshift  painter conditioning nulled + training t shifted noisy
#   mnist_thinker_tshift         training t shifted noisy only
#   mnist_thinker_uncond         painter conditioning nulled only
#   mnist_thinker_cnn12_uncond   thinker reads puzzle + x_t through the CNN of the working grid12 run
#                                (W&B s26ed73n: factor 12, 144 tokens) instead of painter states; painter nulled
#   mnist_thinker_cnn12_cond     the same, painter keeps its puzzle conditioning
#   One change each from mnist_thinker_cnn12_uncond:
#   mnist_thinker_cnn12_uncond_temb   translator gets a timestep embedding (as both working translators)
#   mnist_thinker_cnn12_uncond_chain  translator = DiT-direct's IP construction (shared Linear, per-block
#                                     to_v + zero-init to_out), no timestep embedding
#   mnist_thinker_cnn12_uncond_chain_temb  both: DiT-direct's IP translator incl. its timestep embedding
#                                     (only its bridge differs: normalized here, raw logits there)
#   mnist_thinker_cnn12_uncond_x0     stage-2 loss = x0 MSE (velocity MSE x (1-t)^2); its loss values are
#                                     not comparable to the other runs', its accuracies are
# Patch-16 painter variants (patches aligned with the 9x9 cells, as the working DiT painter) —
# stage 1 then stage 2, each thinker run = DiT-direct's recipe on our side (CNN over puzzle + x_t,
# 81 tokens; IP-chain translator with timestep embedding; steering from block 1; painter blinded):
#   mnist_painter_p16{,_uncond,_tcond,_ddpm,_all}   then   mnist_thinker_p16{,_uncond,_tcond,_ddpm,_all}
#   (none)  patch 16 only (control)
#   _uncond painter trained unconditionally (condition always dropped)
#   _tcond  painter timestep-conditioned (adaLN-Zero)
#   _ddpm   painter trained with DDPM x0 prediction (100-step cosine, uniform t), sampled as the working runs
#   _all    _uncond + _tcond + _ddpm: the closest MM-DiT replica of the working DiT-direct painter
#   _small  painter depth 3, width 512, 8 heads (the working DiT painter's size)
#   _nocond no condition stream at all (NullCondTokenizer): an unconditional, single-stream painter
#   thinker-only: _steer0 (steer every block, from block 0), _fp32 (painter in fp32 in stage 2),
#           _ditopt (DiT-direct's encoder/translator optimizer: lr 3e-5, warmup 1000, eps 3.84e-6)
#   suffixes compose, e.g. mnist_painter_p16_all_small_nocond -> mnist_thinker_p16_all_small_nocond_steer0_fp32
# Measured (A100, freshly initialized painter): thinker batch 384 fits, peak 30.6 GB.
# Bisection on the working DiT-direct painter: mnist_thinker_dit_residual_steer{0,1} = our steering
#   pipeline (CNN thinker, IP-chain translator with timestep embedding, our optimizer) on the frozen
#   pixel DiT of mnist_thinker_dit_direct, steering blocks 0-2 or 1-2.
# Control on the working setup: clevr_unet_condpainter = the CLEVR centroid-mask run
# (W&B 0il6x2cq) without force_unconditional_painter.
# The thinker experiments use translator.bridge_mode=normalized and thinker weight
# decay 0.3 (runs/*_norm*): with raw outputs and weight decay 1.0 the thinker's
# output shrank to ~4% of its initial size and the steering stayed ~0
# (runs/mnist_thinker_mmdit, runs/clevr_thinker_mmdit).
#
# Measured on one H100 (one training micro-batch, torch.compile, peak memory):
#   mnist_painter   256     19.0 GB  0.19 s   -> ~5 h  (100k steps)
#   clevr_painter   256     33.8 GB  0.33 s   -> ~9 h  (100k steps)
#   mnist_thinker   384x2   37.0 GB  4.75 s   -> ~33 h (cnn: 29.9 GB, 3.30 s -> ~23 h)
#   clevr_thinker   256x3   32.2 GB  6.02 s   -> ~47 h
# Sampling during evals costs ~4.0 s (Sudoku, batch 384) / ~2.6 s (CLEVR, batch
# 256) per Euler step, so evals use fewer samples than the defaults (thinkers: 1024
# samples every 5k steps, ~5 min each).

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
MNIST_EVAL="eval.eval_every=5000 eval_callbacks.0.num_samples=1024"
CLEVR_EVAL="eval.eval_every=5000 eval_callbacks.1.n_samples=1024"
MNIST_S1="painter.steer_from_block=1 translator.num_blocks=7"
MNIST_CNN12="condition_encoder.factor=12 translator.grid=12 thinker.seq_len=144 train.batch_size=192 train.gradient_accumulation_steps=4"
train() { python train_trm.py data.num_workers=8 run.output_dir=runs/$RUN "$@"; }

case $RUN in
  mnist_painter)
    RUN=mnist_mmdit_painter
    train experiment=mnist_mmdit_painter eval.eval_every=10000 eval_callbacks.0.num_samples=1024 ;;
  clevr_painter)
    RUN=clevr_mmdit_painter
    train experiment=clevr_mmdit_painter eval.eval_every=10000 eval_callbacks.1.n_samples=1024 ;;
  mnist_thinker)  # as T2I: reads the painter's hidden states after block 4, steers the image stream of blocks 4-7
    RUN=mnist_thinker_mmdit_norm
    train experiment=mnist_thinker_mmdit painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_steer1)  # reads after block 1, steers blocks 1-7 (same painter checkpoint)
    RUN=mnist_thinker_mmdit_norm_steer1
    train experiment=mnist_thinker_mmdit painter.steer_from_block=1 translator.num_blocks=7 \
      painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn)  # reads the raw puzzle + x_t through the working runs' CNN encoder instead
    RUN=mnist_thinker_mmdit_cnn
    train experiment=mnist_thinker_mmdit_cnn painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_steercond)  # + steers the puzzle (condition) stream
    RUN=mnist_thinker_mmdit_norm_steercond
    train experiment=mnist_thinker_mmdit translator.steer_cond=true painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_uncond_tshift)
    RUN=mnist_thinker_mmdit_s1_uncond_mu-1.6
    train experiment=mnist_thinker_mmdit $MNIST_S1 +train.force_unconditional_painter=true painter.t_logit_mean=-1.6 \
      painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_tshift)
    RUN=mnist_thinker_mmdit_s1_mu-1.6
    train experiment=mnist_thinker_mmdit $MNIST_S1 painter.t_logit_mean=-1.6 painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_uncond)
    RUN=mnist_thinker_mmdit_s1_uncond
    train experiment=mnist_thinker_mmdit $MNIST_S1 +train.force_unconditional_painter=true \
      painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_uncond)  # batch 192x4: 384 per micro-batch did not fit a 40 GB A100
    RUN=mnist_thinker_mmdit_cnn12_s1_uncond
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 +train.force_unconditional_painter=true \
      painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_cond)
    RUN=mnist_thinker_mmdit_cnn12_s1_cond
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_uncond_temb)
    RUN=mnist_thinker_mmdit_cnn12_s1_uncond_temb
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 +train.force_unconditional_painter=true \
      translator.with_timestep_emb=true painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_uncond_chain)
    RUN=mnist_thinker_mmdit_cnn12_s1_uncond_chain
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 +train.force_unconditional_painter=true \
      translator.hidden_dim=384 painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_uncond_chain_temb)
    RUN=mnist_thinker_mmdit_cnn12_s1_uncond_chain_temb
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 +train.force_unconditional_painter=true \
      translator.hidden_dim=384 translator.with_timestep_emb=true painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_thinker_cnn12_uncond_x0)
    RUN=mnist_thinker_mmdit_cnn12_s1_uncond_x0
    train experiment=mnist_thinker_mmdit_cnn $MNIST_S1 $MNIST_CNN12 +train.force_unconditional_painter=true \
      painter.loss_weighting=x0 painter.checkpoint=$MNIST_PAINTER $MNIST_EVAL ;;
  mnist_painter_p16*|mnist_thinker_p16*)
    # suffixes after _p16 compose, e.g. mnist_thinker_p16_all_small_nocond_steer0_fp32;
    # painter suffixes pick the painter checkpoint, thinker-only ones (_steer0, _fp32) do not
    DDPM="diffusion=ddpm sampling=cfg painter.objective=ddpm painter.time_cond_scale=1"
    PSIDE="painter.net.patch_size=16"; STAGE1=""; STAGE2=""; PV=""; TV=""; WIDTH=384; DEPTH=8; STEER=1
    for tok in $(echo "${RUN##*_p16}" | tr '_' ' '); do
      case $tok in
        uncond) STAGE1="$STAGE1 painter.label_drop_rate=1.0"; PV="${PV}_$tok" ;;
        tcond) PSIDE="$PSIDE painter.net.time_cond=true"; PV="${PV}_$tok" ;;
        ddpm) PSIDE="$PSIDE $DDPM"; STAGE2="$STAGE2 translator.timestep_scale=1"; PV="${PV}_$tok" ;;
        all) STAGE1="$STAGE1 painter.label_drop_rate=1.0"; PSIDE="$PSIDE painter.net.time_cond=true $DDPM"
             STAGE2="$STAGE2 translator.timestep_scale=1"; PV="${PV}_$tok" ;;
        small) PSIDE="$PSIDE painter.net.depth=3 painter.net.hidden_size=512 painter.net.num_heads=8 painter.net.head_dim=64"
               STAGE1="$STAGE1 painter.steer_from_block=1"  # config default 4 is invalid for 3 blocks
               WIDTH=512; DEPTH=3; PV="${PV}_$tok" ;;
        nocond) PSIDE="$PSIDE painter.cond_tokenizer._target_=models.mmdit_painter.NullCondTokenizer painter.net.preamble_depth=0"
                PV="${PV}_$tok" ;;
        steer0) STEER=0; TV="${TV}_$tok" ;;
        fp32) STAGE2="$STAGE2 painter.painter_dtype=null"; TV="${TV}_$tok" ;;
        ditopt) STAGE2="$STAGE2 painter.optim_cfg.lr=3e-5 painter.optim_cfg.warmup_steps=1000 +painter.optim_cfg.eps=3.84e-6"
                TV="${TV}_$tok" ;;  # DiT-direct's encoder/translator optimizer
        *) echo "unknown variant '$tok' in $RUN" >&2; exit 1 ;;
      esac
    done
    PAINTER_CKPT=runs/mnist_mmdit_painter_p16$PV/checkpoint_final.pt
    if [[ $RUN == mnist_painter_p16* ]]; then
      [[ -n $TV ]] && { echo "$TV are thinker-stage variants, not painter ones" >&2; exit 1; }
      RUN=mnist_mmdit_painter_p16$PV
      train experiment=mnist_mmdit_painter $PSIDE $STAGE1 eval.eval_every=10000 eval_callbacks.0.num_samples=1024
    else
      # the thinker always sees the puzzle (no condition dropout); the painter is blinded instead
      RUN=mnist_thinker_mmdit_p16$PV${TV}_chain_temb
      train experiment=mnist_thinker_mmdit_cnn painter.checkpoint=$PAINTER_CKPT $PSIDE $STAGE2 \
        painter.steer_from_block=$STEER translator.num_blocks=$((DEPTH - STEER)) translator.painter_grid=9 \
        translator.hidden_dim=$WIDTH translator.with_timestep_emb=true \
        +train.force_unconditional_painter=true painter.label_drop_rate=0.0 $MNIST_EVAL
    fi ;;
  mnist_thinker_dit_residual_steer0|mnist_thinker_dit_residual_steer1)
    # bisection: our steering pipeline on the working DiT-direct painter (ResidualSteeredDiTPainter)
    STEER=${RUN##*_steer}
    RUN=mnist_thinker_dit_residual_steer$STEER
    train experiment=mnist_thinker_dit_residual painter.checkpoint=runs/mnist_dit_painter_pixel/checkpoint_final.pt \
      painter.steer_from_block=$STEER translator.num_blocks=$((3 - STEER)) $MNIST_EVAL ;;
  clevr_unet_condpainter)  # exactly W&B 0il6x2cq's config, but the painter keeps its object conditioning
    RUN=clevr_thinker_v1_centroidmaskfused_x0hint_t80_sliced_biggrid_condpainter
    train experiment=clevr_thinker_v1_controlnet painter.checkpoint=runs/clevr_unet_big/checkpoint_step-80000.pt \
      "backbone.block_out_channels=[128,256,512]" condition_encoder=object_feature_v1_centroidmask_presence_fused_x0hint \
      condition_encoder.threshold=80 translator=controlnet_clevr_v1_sliced "translator.painter_channels=[128,256,512]" \
      thinker.seq_len=266 data.include_centroid_mask=true data.centroid_mask_presence_only=true ;;
  clevr_thinker)
    RUN=clevr_thinker_mmdit_norm
    train experiment=clevr_thinker_mmdit painter.checkpoint=$CLEVR_PAINTER $CLEVR_EVAL ;;
  clevr_thinker_steercond)
    RUN=clevr_thinker_mmdit_norm_steercond
    train experiment=clevr_thinker_mmdit translator.steer_cond=true painter.checkpoint=$CLEVR_PAINTER $CLEVR_EVAL ;;
  *)
    echo "unknown run '$RUN' (see the header of $0)" >&2; exit 1 ;;
esac
