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
from backward_compatibility import load_with_backward_compatibility

logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.checkpoint_path is None:
        raise ValueError("You must provide a checkpoint_path to sample from!")

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

    unet_dir = os.path.join(resolved_ckpt_path, "unet")
    sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

    # Load raw state dict from disk
    if os.path.exists(sf_path):
        raw_state_dict = load_file(sf_path)
    elif os.path.exists(bin_path):
        raw_state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

    # Pass it through our translator
    load_with_backward_compatibility(unet, raw_state_dict, logger)

    if args.use_small_loop:
        y_path = os.path.join(args.output_dir, "y_init.pt")
        z_path = os.path.join(args.output_dir, "z_init.pt")

        if not os.path.exists(y_path) or not os.path.exists(z_path):
            raise FileNotFoundError(f"TRM anchors (y_init.pt / z_init.pt) not found in {args.output_dir}. Did the training script save them?")

        unet.y_init = torch.load(y_path, map_location="cpu")
        unet.z_init = torch.load(z_path, map_location="cpu")

    unet.eval()
    unet = accelerator.prepare(unet)
    unet = accelerator.unwrap_model(unet)  # Safely unwrap it once before the loop starts!

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
    num_per_gpu = args.num_samples // world_size
    batch_size = min(args.sample_batch_size, num_per_gpu)
    num_batches = math.ceil(num_per_gpu / batch_size)

    output_dir = Path(args.output_dir) / "samples"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(process_index)

    # 4. The Generation Loop
    logger.info(f"Generating {num_per_gpu} samples on process {process_index}...")

    # Create/overwrite the JSONL file for this specific GPU process
    metadata_path = output_dir / f"metadata_rank{process_index}.jsonl"
    with open(metadata_path, "w") as f:
        pass # Just to clear the file if it already exists from an old run

    for b_idx in tqdm(range(num_batches), disable=not accelerator.is_local_main_process):
        current_bsz = min(batch_size, num_per_gpu - (b_idx * batch_size))

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
            weight_dtype=torch.float32,
            show_progress=False,  # Disable inner progress bar to prevent terminal spam
        )

        # Format and save to disk
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)

        with open(metadata_path, "a") as f:
            for i, (img, meta) in enumerate(zip(images, metadata)):
                global_idx = (process_index * num_per_gpu) + (b_idx * batch_size) + i
                filename = f"sample_{global_idx:06d}.png"

                img_path = output_dir / filename
                Image.fromarray(img).save(img_path)

                meta["file_name"] = filename
                f.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
