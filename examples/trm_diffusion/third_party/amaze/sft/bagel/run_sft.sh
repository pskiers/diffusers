#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,6,7 # Adjust based on available GPUs
NUM_GPUS=4  # Number of GPUs to use
MASTER_PORT=29500  # Port for distributed training

MAZE_DATASET_PATH="/path/to/maze_dataset"
DATASET_CONFIG_FILE="/path/to/dataset_config.yaml"  # Fallback config (may not be used)
# Model Paths
MODEL_PATH="/path/to/model"  # Path to BAGEL model

# Training Configuration
TOTAL_STEPS=5000  # Total training steps
SAVE_EVERY=100   # Save checkpoint every N steps
LOG_EVERY=1     # Log every N steps
WARMUP_STEPS=10 # Learning rate warmup steps

# Learning Rate Settings
LR=1e-5          # Peak learning rate
LR_SCHEDULER="cosine"  # "cosine" or "constant"
MIN_LR=1e-7       # Minimum LR for cosine scheduler

# Batch and Memory Settings
EXPECTED_NUM_TOKENS=5000     # Target tokens per batch
MAX_NUM_TOKENS=5000         # Hard limit on tokens per batch
MAX_NUM_TOKENS_PER_SAMPLE=5000  # Max tokens per individual sample
GRADIENT_ACCUMULATION_STEPS=8    # Gradient accumulation steps

# Output Settings
RESULTS_DIR="results/sft_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="results/sft_$(date +%Y%m%d_%H%M%S)/checkpoints"
WANDB_PROJECT="maze_sft"
WANDB_NAME="sft_$(date +%Y%m%d_%H%M%S)"
WANDB_OFFLINE=true  # Set to true for offline mode


# Model Configuration
VISUAL_GEN=true   # Enable image generation
VISUAL_UND=true   # Enable image understanding
FREEZE_LLM=false  # Whether to freeze language model
FREEZE_VIT=true  # Whether to freeze ViT
FREEZE_VAE=true   # Whether to freeze VAE

# =============================================================================
# Environment Setup
# =============================================================================

# Set environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# NCCL settings for better distributed training
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

FINETUNE_FROM_EMA=true
RESUME_MODEL_ONLY=true

echo "============================================="
echo "Maze SFT Training Configuration"
echo "============================================="
echo "GPUs: $NUM_GPUS"
echo "Maze Dataset: $MAZE_DATASET_PATH"
echo "Model Path: $MODEL_PATH"
echo "Results Dir: $RESULTS_DIR"
echo "Total Steps: $TOTAL_STEPS"
echo "Learning Rate: $LR"
echo "Batch Tokens: $EXPECTED_NUM_TOKENS"
echo "============================================="

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"

# =============================================================================
# Training Command
# =============================================================================
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    sft.py \
    --maze_dataset_path "$MAZE_DATASET_PATH" \
    --dataset_config_file "$DATASET_CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --visual_gen $VISUAL_GEN \
    --visual_und $VISUAL_UND \
    --resume_from "$MODEL_PATH" \
    --finetune_from_ema $FINETUNE_FROM_EMA \
    --resume_model_only $RESUME_MODEL_ONLY \
    --results_dir "$RESULTS_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --wandb_offline $WANDB_OFFLINE \
    --total_steps $TOTAL_STEPS \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --eval_every 50 \
    --eval_samples 8 \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --lr_scheduler "$LR_SCHEDULER" \
    --min_lr $MIN_LR \
    --expected_num_tokens $EXPECTED_NUM_TOKENS \
    --max_num_tokens $MAX_NUM_TOKENS \
    --max_num_tokens_per_sample $MAX_NUM_TOKENS_PER_SAMPLE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --freeze_llm $FREEZE_LLM \
    --freeze_vit $FREEZE_VIT \
    --freeze_vae $FREEZE_VAE \
    --auto_resume true \
    --finetune_from_hf true \
    --num_workers 1 \
    --prefetch_factor 1 \
    --max_buffer_size 1 \
    --prefer_buffer_before 10000 \
    --max_grad_norm 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-15 \
    --ce_weight 0.000001 \
    --mse_weight 1.0 \
    --timestep_shift 1.0 \
    --max_latent_size 64 \
    --latent_patch_size 2 \
    --vit_patch_size 14 \
    --vit_max_num_patch_per_side 70 \
    --text_cond_dropout_prob 0 \
    --vae_cond_dropout_prob 0 \
    --vit_cond_dropout_prob 0 \
    --connector_act "gelu_pytorch_tanh" \
    --interpolate_pos false \
    --vit_rope false \
    --llm_qk_norm true \
    --tie_word_embeddings false \
    --layer_module "Qwen2MoTDecoderLayer" \
    --copy_init_moe true \
    --use_flex false \
    --global_seed 4396 \
    --sharding_strategy "HYBRID_SHARD" \
    --backward_prefetch "BACKWARD_PRE" \
    --num_replicate 1 \
    --num_shard 4 \
    --cpu_offload true \
    --use_lora false \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_lora_checkpoint false

echo "Training completed!"
