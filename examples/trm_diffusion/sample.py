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

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from hydra.utils import instantiate
from safetensors.torch import load_file

from eval_utils import generate_image_batch

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
    logger.info(f"Instantiating model from config...")
    unet = instantiate(args.model, _convert_="all")

    unet_dir = os.path.join(args.checkpoint_path, "unet")
    sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

    if os.path.exists(sf_path):
        unet.load_state_dict(load_file(sf_path))
    elif os.path.exists(bin_path):
        unet.load_state_dict(torch.load(bin_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

    if args.use_small_loop:
        unet.y_init = torch.load(os.path.join(args.checkpoint_path, "y_init.pt"), map_location="cpu")
        unet.z_init = torch.load(os.path.join(args.checkpoint_path, "z_init.pt"), map_location="cpu")

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

    for b_idx in tqdm(range(num_batches), disable=not accelerator.is_local_main_process):
        current_bsz = min(batch_size, num_per_gpu - (b_idx * batch_size))

        # --- Call the Shared Engine ---
        images = generate_image_batch(
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

        for i, img in enumerate(images):
            global_idx = (process_index * num_per_gpu) + (b_idx * batch_size) + i
            img_path = output_dir / f"sample_{global_idx:06d}.png"
            Image.fromarray(img).save(img_path)


if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
