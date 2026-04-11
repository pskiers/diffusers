import os
import logging
import torch
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import math
import json

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from hydra.utils import instantiate
from safetensors.torch import load_file

from eval_utils import generate_image_batch
from model_utils import load_with_backward_compatibility
from sokoban.sokoban_utils import SokobanSampler


logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)

    # 1. Load Model
    if args.get("checkpoint_step") is not None:
        if str(args.checkpoint_step).lower() == "latest":
            if not os.path.exists(args.output_dir):
                raise FileNotFoundError(f"Output directory {args.output_dir} does not exist.")

            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))

            if not dirs:
                raise ValueError(f"No checkpoints found in {args.output_dir}")
            resolved_ckpt_path = os.path.join(args.output_dir, dirs[-1])
        else:
            resolved_ckpt_path = os.path.join(args.output_dir, f"checkpoint-{args.checkpoint_step}")
    elif args.get("checkpoint_path") is not None:
        resolved_ckpt_path = args.checkpoint_path
    else:
        raise ValueError("You must provide either 'checkpoint_step' or 'checkpoint_path'!")

    if not os.path.exists(resolved_ckpt_path):
        raise FileNotFoundError(f"Resolved checkpoint path does not exist: {resolved_ckpt_path}")

    logger.info(f"Loading weights from {resolved_ckpt_path}")

    # ---------------------------------------------------------
    # 2. Load Model Architecture and Weights
    # ---------------------------------------------------------
    logger.info(f"Instantiating model from config...")
    unet = instantiate(args.model, _convert_="all")

    if hasattr(unet, "load"):
        logger.info("Loading extra TRM modules and strategy states...")
        unet.load(resolved_ckpt_path)

    if args.get("use_ema", False):
        logger.info("Loading EMA weights...")
        unet_dir = os.path.join(resolved_ckpt_path, "unet_ema")
    else:
        unet_dir = os.path.join(resolved_ckpt_path, "unet")
    sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

    if os.path.exists(sf_path):
        raw_state_dict = load_file(sf_path)
    elif os.path.exists(bin_path):
        raw_state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

    # FIX: Safely point to the core model if the wrapper exists
    unet_to_load = unet.core_model if hasattr(unet, "core_model") else unet

    # Pass the core model through your translator
    load_with_backward_compatibility(unet_to_load, raw_state_dict, logger)

    unet.eval()

    # Determine the correct dtype based on accelerator mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # 1. Safely move the core model to the GPU and CAST the dtype
    if hasattr(unet, "core_model"):
        unet.core_model.to(accelerator.device, dtype=weight_dtype)
    else:
        unet.to(accelerator.device, dtype=weight_dtype)

    # 2. Safely move any extra Mixin layers (norm_y, fusion, etc.) to the GPU
    if hasattr(unet, "get_trainable_modules"):
        for m in unet.get_trainable_modules().values():
            if isinstance(m, torch.nn.Module):
                m.to(accelerator.device, dtype=weight_dtype)

    # 2. Load VAE and Scheduler
    vae, vae_scaling_factor = None, 1.0
    if args.dataset.vae_name is not None:
        vae = AutoencoderKL.from_pretrained(args.dataset.vae_name).to(accelerator.device, dtype=torch.float32)
        vae.requires_grad_(False)
        vae.eval()
        vae_scaling_factor = vae.config.scaling_factor

    SchedulerClass = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler_kwargs = {"num_train_timesteps": args.ddpm_num_steps, "beta_schedule": args.ddpm_beta_schedule}
    if "prediction_type" in SchedulerClass.__init__.__code__.co_varnames:
        scheduler_kwargs["prediction_type"] = args.prediction_type

    scheduler = SchedulerClass(**scheduler_kwargs)
    scheduler.set_timesteps(args.ddpm_num_inference_steps)

    # 3. Distributed Math & Setup
    world_size = accelerator.num_processes
    process_index = accelerator.process_index

    output_dir = Path(args.output_dir) / args.get("samples_dir", "samples")
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(process_index)

    sokoban_sampler = SokobanSampler(args) if getattr(args.dataset, 'dataset_type', None) == 'sokoban' else None

    if sokoban_sampler and sokoban_sampler.prompt_iterator is not None: # for conditioning images, batch size must be adjusted
        total_samples = sokoban_sampler.n_images_to_eval * sokoban_sampler.n_images_per_cond
        num_per_gpu = total_samples // world_size
        batch_size = min(args.sample_batch_size, num_per_gpu)
        batch_size = max(sokoban_sampler.n_images_per_cond, (batch_size // sokoban_sampler.n_images_per_cond) * sokoban_sampler.n_images_per_cond)
    else:
        num_per_gpu = args.num_samples // world_size
        batch_size = min(args.sample_batch_size, num_per_gpu)

    num_batches = math.ceil(num_per_gpu / batch_size)

    # 4. The Generation Loop
    logger.info(f"Generating {num_per_gpu} samples on process {process_index}...")

    # Create/overwrite the JSONL file for this specific GPU process
    metadata_path = output_dir / f"metadata_rank{process_index}.jsonl"
    with open(metadata_path, "w") as f:
        pass  # Just to clear the file if it already exists from an old run

    for b_idx in tqdm(range(num_batches), disable=not accelerator.is_local_main_process):
        current_bsz = min(batch_size, num_per_gpu - (b_idx * batch_size))

        cond_images_prompt, class_labels_prompt = None, None
        if sokoban_sampler:
            cond_images_prompt, class_labels_prompt, current_bsz = sokoban_sampler.prepare_for_conditioning(accelerator, weight_dtype, current_bsz)

        # --- Call the Shared Engine ---
        images, metadata = generate_image_batch(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            vae_scaling_factor=vae_scaling_factor,
            args=args,
            bsz=current_bsz,
            generator=generator,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            single_scene=False,
            show_progress=True,
            early_stopping_threshold=args.get("early_stopping_threshold", None),
            cond_images=cond_images_prompt,
            class_labels=class_labels_prompt
        )

        # Format and save to disk
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        images_np = (images_np * 255).round().astype(np.uint8)

        if sokoban_sampler:
            sokoban_sampler.register_generated(images_np)
            images_np = np.array([sokoban_sampler._render(board) for board in sokoban_sampler.all_gen_boards_list[-1]], dtype=np.uint8)

        with open(metadata_path, "a") as f:
            for i, (img, meta) in enumerate(zip(images_np, metadata)):
                global_idx = (process_index * num_per_gpu) + (b_idx * batch_size) + i
                filename = f"sample_{global_idx:06d}.png"
                Image.fromarray(img).save(output_dir / filename)
                meta["file_name"] = filename
                f.write(json.dumps(meta) + "\n")

    if sokoban_sampler:
        sokoban_sampler.sampling_evaluation(logger, process_index, output_dir)

if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
