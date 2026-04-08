#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%j.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%j.err


source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0
source /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/env/bin/activate

cd /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion


# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   dataset.dataset_mode="reduced" \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_vit_std \
#   output_dir="clevr_reduced_vit_std" \
#   checkpoint_step=125000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   dataset.dataset_mode="reduced" \
#   experiment=clevr_relative_trm \
#   model=clevr_condition_unet_trm_v2 \
#   output_dir="clevr_reduced_unet_trm_v2" \
#   checkpoint_step=15000 \
#   num_samples=10000 \
#   sample_batch_size=2000

accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
  dataset.dataset_mode="reduced" \
  mixed_precision=bf16 \
  experiment=clevr_relative_trm \
  model=clevr_vit_trm_v2 \
  output_dir="clevr_reduced_vit_trm_v2" \
  checkpoint_step=30000 \
  num_samples=10000 \
  sample_batch_size=2000


# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   experiment=clevr_relative_trm \
#   model=clevr_condition_unet_trm_v2 \
#   output_dir="clevr_unet_trm_v2" \
#   checkpoint_step=40000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_trm \
#   model=clevr_vit_trm_v2 \
#   output_dir="clevr_vit_trm_v2" \
#   checkpoint_step=70000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_vit_std \
#   output_dir="clevr_vit_std" \
#   checkpoint_step=70000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   experiment=clevr_relative_std \
#   model=clevr_ratatouille_control \
#   output_dir="clevr_ratatouille_control" \
#   checkpoint_step=35000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   experiment=clevr_relative_std \
#   model=clevr_ratatouille_concat \
#   output_dir="clevr_ratatouille_concat" \
#   checkpoint_step=45000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_ratatouille_dit_residual \
#   output_dir="clevr_ratatouille_dit_residual" \
#   checkpoint_step=50000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_ratatouille_dit_concat \
#   output_dir="clevr_ratatouille_dit_concat" \
#   checkpoint_step=55000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="bf16" --num_processes=1 sample.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_vit_std \
#   output_dir="clevr_vit_std" \
#   checkpoint_step=70000 \
#   num_samples=10000 \
#   sample_batch_size=2000

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   output_dir="cifar_unet_trm_v2_nsup4_grad_not_every_n_exp_loss" \
#   train_batch_size=64 \
#   grad_every_n_sup=false \
#   trm_loss_nsup_decay=0.7 \
#   phases=[10000000000] \
#   n_sup_phases=[4] \
#   checkpoint_step=210000 \
#   num_samples=10000 \
#   sample_batch_size=2000 \
#   samples_dir="samples_210k"

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   output_dir="cifar_unet_trm_v2_nsup4_grad_not_every_n" \
#   train_batch_size=64 \
#   grad_every_n_sup=false \
#   phases=[10000000000] \
#   n_sup_phases=[4] \
#   checkpoint_step=210000 \
#   num_samples=10000 \
#   sample_batch_size=2000
#   samples_dir="samples_210k"

# python analyze_early_stopping.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   mixed_precision=fp16 \
#   model.n_sup=16 \
#   output_dir=cifar_unet_trm_v2_fast \
#   checkpoint_step=50000 \
#   +analysis.num_images=64 \
#   +analysis.num_seeds=12 \
#   +analysis.output_dir=early_stopping_analysis_cifar_unet \
#   +analysis.thresholds=[100.0,5.0,1.0,0.5,0.1]


# accelerate launch --mixed_precision="bf16" --num_processes=1 train.py \
#   mixed_precision=bf16 \
#   experiment=cond_cifar100_trm \
#   model=cifar100_vit_trm_v2 \
#   ~model.core_model.num_class_embeds \
#   output_dir="cifar_vit_trm_v2" \
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="bf16" --num_processes=1 train.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_std \
#   model=clevr_vit_std \
#   output_dir="clevr_reduced_vit_std" \
#   dataset.dataset_mode="reduced"\
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   experiment=clevr_relative_trm \
#   model=clevr_ratatouille_control \
#   output_dir="clevr_ratatouille_control" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   experiment=clevr_relative_trm \
#   model=clevr_ratatouille_concat \
#   output_dir="clevr_ratatouille_concat" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="bf16" --num_processes=1 train.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_trm \
#   model=clevr_ratatouille_dit_residual \
#   output_dir="clevr_ratatouille_dit_residual" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="bf16" --num_processes=1 train.py \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_trm \
#   model=clevr_ratatouille_dit_concat \
#   output_dir="clevr_ratatouille_dit_concat" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"

# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   output_dir="cifar_unet_trm_v2_fast" \
  # gradient_accumulation_steps=2 \
  # train_batch_size=32 \
#   grad_every_n_sup=false \
#   trm_loss_nsup_decay=0.7 \
#   phases=[30000,50000,10000000000] \
#   n_sup_phases=[4,8,16] \
#   resume_from_checkpoint="latest"

# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   output_dir="cifar_unet_trm_v2_nsup16_grad_not_every_n" \
#   gradient_accumulation_steps=2 \
#   train_batch_size=32 \
#   grad_every_n_sup=false \
#   phases=[10000000000] \
#   n_sup_phases=[16] \
#   resume_from_checkpoint="latest"

# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   experiment=cond_cifar100_trm \
#   model=unet2d_trm_v2 \
#   output_dir="cifar_unet_trm_v2_nsup16_grad_not_every_n_exp_loss" \
#   gradient_accumulation_steps=2 \
#   train_batch_size=32 \
#   grad_every_n_sup=false \
#   trm_loss_nsup_decay=0.7 \
#   phases=[10000000000] \
#   n_sup_phases=[16] \
#   resume_from_checkpoint="latest"


# accelerate launch --mixed_precision="fp16" --num_processes=1 train.py \
#   dataset.dataset_mode="reduced" \
#   experiment=clevr_relative_trm \
#   model=clevr_condition_unet_trm_v2 \
#   output_dir="clevr_reduced_unet_trm_v2" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"

# accelerate launch --mixed_precision="bf16" --num_processes=1 train.py \
#   dataset.dataset_mode="reduced" \
#   mixed_precision=bf16 \
#   experiment=clevr_relative_trm \
#   model=clevr_vit_trm_v2 \
#   output_dir="clevr_reduced_vit_trm_v2" \
#   train_batch_size=256 \
#   resume_from_checkpoint="latest"