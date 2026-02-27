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

# Import your custom logic
from trm_utils import get_model_output

logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.checkpoint_path is None:
        raise ValueError("You must provide a checkpoint_path to sample from!")

    # ---------------------------------------------------------
    # 1. Load the Model Architecture and Weights
    # ---------------------------------------------------------
    logger.info(f"Instantiating model from config...")
    unet = instantiate(args.model, _convert_="all")

    # Diffusers saves UNets in a "unet/" subfolder
    unet_dir = os.path.join(args.checkpoint_path, "unet")
    sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

    if os.path.exists(sf_path):
        unet.load_state_dict(load_file(sf_path))
    elif os.path.exists(bin_path):
        unet.load_state_dict(torch.load(bin_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

    # Load TRM Anchors if needed
    if args.use_small_loop:
        y_path = os.path.join(args.checkpoint_path, "y_init.pt")
        z_path = os.path.join(args.checkpoint_path, "z_init.pt")
        unet.y_init = torch.load(y_path, map_location="cpu")
        unet.z_init = torch.load(z_path, map_location="cpu")

    unet.eval()
    unet = accelerator.prepare(unet)

    # ---------------------------------------------------------
    # 2. Load VAE and Scheduler
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. Distributed Math & Setup
    # ---------------------------------------------------------
    world_size = accelerator.num_processes
    process_index = accelerator.process_index
    num_per_gpu = args.num_samples // world_size
    batch_size = min(args.sample_batch_size, num_per_gpu)
    num_batches = math.ceil(num_per_gpu / batch_size)

    sample_size = args.dataset.resolution if vae is None else args.dataset.resolution // 8
    sample_channels = args.dataset.input_channels

    output_dir = Path(args.output_dir) / "samples"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(process_index)

    # Model configuration flags
    is_unified_class = getattr(args.model, "condition_mode", None) == "class"
    is_unified_sequence = getattr(args.model, "condition_mode", None) == "sequence"
    is_standard_conditional = "UNet2DModel" in args.model._target_ and args.dataset.num_classes
    do_cfg = args.guidance_scale > 1.0 and (is_unified_class or is_standard_conditional)

    # ---------------------------------------------------------
    # 4. The Unified Generation Loop
    # ---------------------------------------------------------
    logger.info(f"Generating {num_per_gpu} samples on process {process_index}...")

    for b_idx in tqdm(range(num_batches), disable=not accelerator.is_local_main_process):
        current_bsz = min(batch_size, num_per_gpu - (b_idx * batch_size))

        # --- A. Build Conditions ---
        conds, masks = None, None
        unconds = None

        if is_unified_class or is_standard_conditional:
            conds = torch.randint(
                0, args.dataset.num_classes, [current_bsz], generator=generator, device=accelerator.device
            )
            if do_cfg:
                unconds = torch.full_like(conds, args.dataset.num_classes)

        elif is_unified_sequence:
            from clevr_dataset import sample_random_scene, make_tensor_from_scene

            c_list, m_list = [], []
            for _ in range(current_bsz):
                scene = sample_random_scene(num_objects=4, mode=args.dataset.dataset_mode)
                c, m = make_tensor_from_scene(scene)
                c_list.append(c)
                m_list.append(m)
            conds = torch.cat(c_list, dim=0).to(accelerator.device)
            masks = torch.cat(m_list, dim=0).to(accelerator.device)

        # --- B. Denoising Loop ---
        latents = torch.randn(
            (current_bsz, sample_channels, sample_size, sample_size), generator=generator, device=accelerator.device
        )

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            class_input = torch.cat([conds, unconds]) if do_cfg else conds
            mask_input = torch.cat([masks, masks]) if (do_cfg and masks is not None) else masks

            with torch.no_grad():
                if args.use_small_loop:
                    from trm_utils import deep_recursion

                    bsz = latent_model_input.shape[0]
                    # Safely unwrap the model to access custom attributes hidden by DDP
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    y = unwrapped_unet.y_init.expand(bsz, -1, -1, -1).to(accelerator.device)
                    z = unwrapped_unet.z_init.expand(bsz, -1, -1, -1).to(accelerator.device)

                    for _ in range(args.N_supervision):
                        # Ensure these arguments match the signature in your trm_utils.py
                        noise_pred, y, z = deep_recursion(
                            unet, latent_model_input, y, z, t, class_input, mask_input, args.n, args.T
                        )
                else:
                    noise_pred = get_model_output(unet, latent_model_input, t, class_input, mask_input)

            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- C. VAE Decoding & Formatting ---
        if vae is not None:
            latents = 1 / vae_scaling_factor * latents
            image = vae.decode(latents.to(vae.dtype)).sample
        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype(np.uint8)

        # --- D. Save to Disk ---
        for i, img in enumerate(image):
            global_idx = (process_index * num_per_gpu) + (b_idx * batch_size) + i
            img_path = output_dir / f"sample_{global_idx:06d}.png"
            Image.fromarray(img).save(img_path)


if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
