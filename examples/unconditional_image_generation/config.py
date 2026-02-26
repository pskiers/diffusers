import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Script for Diffusion Models")

    # ---------------------------------------------------------
    # Core & Paths
    # ---------------------------------------------------------
    core_group = parser.add_argument_group("Core & Paths")
    core_group.add_argument("--output_dir", type=str, default="ddpm-model-64", help="Output directory for model predictions and checkpoints.")
    core_group.add_argument("--overwrite_output_dir", action="store_true")
    core_group.add_argument("--cache_dir", type=str, default=None, help="Directory for downloaded models and datasets.")
    core_group.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    core_group.add_argument("--logging_dir", type=str, default="logs")
    core_group.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # ---------------------------------------------------------
    # Dataset & Dataloading
    # ---------------------------------------------------------
    data_group = parser.add_argument_group("Dataset & Dataloading")
    data_group.add_argument("--dataset_type", type=str, default="hf", choices=["hf", "clevr"], help="Type of dataset to load.")
    data_group.add_argument("--dataset_name", type=str, default=None, help="Name of the Dataset from the HuggingFace hub.")
    data_group.add_argument("--dataset_config_name", type=str, default=None, help="Config of the Dataset.")
    data_group.add_argument("--train_data_dir", type=str, default=None, help="Folder containing the training data.")
    data_group.add_argument("--dataset_mode", type=str, default="relative", choices=["absolute", "relative"], help="CLEVR dataset mode.")
    data_group.add_argument("--test_split_name", type=str, default="test", help="Name of split to use for testing")
    data_group.add_argument("--image_key", type=str, default="img", help="Image key in the dataset")
    data_group.add_argument("--class_key", type=str, default="fine_label", help="Class key in the dataset")
    data_group.add_argument("--resolution", type=int, default=64, help="Resolution for input images.")
    data_group.add_argument("--center_crop", action="store_true", default=False)
    data_group.add_argument("--random_flip", action="store_true", default=False)
    data_group.add_argument("--dataloader_num_workers", type=int, default=0)
    data_group.add_argument("--epoch_max_batches_train", type=int, default=1000, help="Max number of batches per epoch for train")
    data_group.add_argument("--epoch_max_batches_eval", type=int, default=250, help="Max number of batches per epoch for eval")

    # ---------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------
    arch_group = parser.add_argument_group("Architecture")
    arch_group.add_argument("--model_type", type=str, default="unet2d", choices=["unet2d", "unified_class", "unified_sequence"], help="Explicitly choose the model architecture.")
    arch_group.add_argument("--model_config_name_or_path", type=str, default=None)
    arch_group.add_argument("--vae_name", type=str, default=None, help="Path to VAE if doing Latent Diffusion.")
    arch_group.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
    arch_group.add_argument("--num_classes", type=int, default=100, help="Number of classes for conditional generation")
    arch_group.add_argument("--channels", nargs="+", type=int, default=[128, 128], help="Channels in each UNet block.")
    arch_group.add_argument("--down_block_types", nargs="+", type=str, default=["AttnDownBlock2D", "DownBlock2D"])
    arch_group.add_argument("--up_block_types", nargs="+", type=str, default=["UpBlock2D", "AttnUpBlock2D"])
    arch_group.add_argument("--layers_per_block", type=int, default=2, help="Layers per UNet block")

    # ---------------------------------------------------------
    # "Small Loop" & Recursive Thinking Arguments
    # ---------------------------------------------------------
    loop_group = parser.add_argument_group("Recursive Loop")
    loop_group.add_argument("--T", type=int, default=3, help="Number of macroscopic steps (T)")
    loop_group.add_argument("--n", type=int, default=6, help="Number of microscopic iterations (n)")
    loop_group.add_argument("--N_supervision", type=int, default=4, help="Supervision factor")
    loop_group.add_argument("--use_small_loop", action="store_true", help="Enable the recursive latent loop")

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--train_batch_size", type=int, default=16)
    train_group.add_argument("--eval_batch_size", type=int, default=16)
    train_group.add_argument("--num_epochs", type=int, default=100)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_group.add_argument("--learning_rate", type=float, default=1e-4)
    train_group.add_argument("--lr_scheduler", type=str, default="cosine")
    train_group.add_argument("--lr_warmup_steps", type=int, default=500)
    train_group.add_argument("--adam_beta1", type=float, default=0.95)
    train_group.add_argument("--adam_beta2", type=float, default=0.999)
    train_group.add_argument("--adam_weight_decay", type=float, default=1e-6)
    train_group.add_argument("--adam_epsilon", type=float, default=1e-08)
    train_group.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # ---------------------------------------------------------
    # Diffusion Settings
    # ---------------------------------------------------------
    diff_group = parser.add_argument_group("Diffusion & CFG Settings")
    diff_group.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])
    diff_group.add_argument("--ddpm_num_steps", type=int, default=1000)
    diff_group.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    diff_group.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    diff_group.add_argument("--cfg_drop_rate", type=float, default=0.1, help="Probability of dropping class labels for CFG.")
    diff_group.add_argument("--guidance_scale", type=float, default=4.0, help="CFG guidance scale during evaluation.")

    # ---------------------------------------------------------
    # EMA Settings
    # ---------------------------------------------------------
    ema_group = parser.add_argument_group("EMA")
    ema_group.add_argument("--use_ema", action="store_true")
    ema_group.add_argument("--ema_inv_gamma", type=float, default=1.0)
    ema_group.add_argument("--ema_power", type=float, default=3 / 4)
    ema_group.add_argument("--ema_max_decay", type=float, default=0.9999)

    # ---------------------------------------------------------
    # Saving & Checkpointing
    # ---------------------------------------------------------
    save_group = parser.add_argument_group("Saving & Checkpointing")
    save_group.add_argument("--save_images_epochs", type=int, default=10)
    save_group.add_argument("--save_model_epochs", type=int, default=10)
    save_group.add_argument("--checkpointing_steps", type=int, default=500)
    save_group.add_argument("--checkpoints_total_limit", type=int, default=None)
    save_group.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args
