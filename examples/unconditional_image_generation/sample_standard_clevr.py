import argparse
import inspect
import logging
import os
from datetime import timedelta
from pathlib import Path
from PIL import Image

import datasets
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from train_clevr import CLEVRDiffusionModel, sample_random_scene, make_tensor_from_scene


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--num_samples", type=int, default=20000, help="How many images to sample.")
    parser.add_argument("--max_bs", type=int, default=10000, help="Max batch size that fits on a single gpu.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--vae_name",
        type=str,
        required=False,
        default=None,
        help="If doing ldm pass path to VAE here, otherwise don't set it to anything."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model

    model = CLEVRDiffusionModel.from_pretrained(args.model_config_name_or_path)

    vae = AutoencoderKL.from_pretrained(args.vae_name, cache_dir=args.cache_dir)
    vae.requires_grad_(False)
    vae_scaling_factor = vae.config.scaling_factor
    vae.to(accelerator.device, dtype=torch.float32)
    vae.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(model)

    unet = accelerator.unwrap_model(model)

    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    )

    # Get the world size (total number of processes)
    world_size = accelerator.state.num_processes
    num_per_gpu = args.num_samples // world_size
    batch_size = min(args.max_bs, num_per_gpu)
    num_batches = num_per_gpu // batch_size

    # run pipeline in inference (sample random noise and denoise)
    with accelerator.accumulate(model):
        process_number = accelerator.state.process_index
        generator = torch.Generator(device=pipeline.device).manual_seed(process_number)
        for i in range(num_batches):
            conds = []
            masks = []

            # 3 objects, they should form something of a triangle:
            #   gray cube - left front,
            #   red sphere - right middle,
            #   blue cylinder - middle back
            scene_dict = {
                "objects": [
                    {"color": "gray", "shape": "cube", "size": "small", "material": "rubber", "rotation": 0, "3d_coords": [0, 0, 0], "pixel_coords": [100, 200, 10]},
                    {"color": "red", "shape": "sphere", "size": "small", "material": "rubber", "rotation": 0, "3d_coords": [0, 0, 0], "pixel_coords": [200, 100, 10]},
                    {"color": "blue", "shape": "cylinder", "size": "large", "material": "metal", "rotation": 0, "3d_coords": [0, 0, 0], "pixel_coords": [20, 200, 10]},
                    {"color": "green", "shape": "cylinder", "size": "large", "material": "metal", "rotation": 0, "3d_coords": [0, 0, 0], "pixel_coords": [100, 20, 10]},
                ],
                "relationships": {
                    "left": [[1,2,3], [3], [1,3], []],
                    "right": [[], [0, 2], [0], [0,1,2]],
                    "front": [[1,2,3], [2,3], [3], []],
                    "behind": [[], [0], [0,1], [0,1,2]],
                    # "left": [[1,2], [], [1]],
                    # "right": [[], [0, 2], [0]],
                    # "front": [[1,2], [2], []],
                    # "behind": [[], [0], [0,1]],
                    # "left": [[1], []],
                    # "right": [[], [0]],
                    # "front": [[1], []],
                    # "behind": [[], [0]],
                },
                "mode": "absolute"
            }
            for _ in range(batch_size):
                # scene_dict = sample_random_scene(num_objects=4, mode="relative")
                cond, mask = make_tensor_from_scene(scene_dict)
                conds.append(cond)
                masks.append(mask)
            cond_tensor = torch.cat(conds, dim=0).to(pipeline.device)
            mask = torch.cat(masks, dim=0).to(pipeline.device)

            images = pipeline(
                generator=generator,
                batch_size=batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="pt",
                raw_objects=cond_tensor,
                obj_mask=mask,
            ).images
            latents = images / vae_scaling_factor
            latents = latents.to(torch.float32)

            with torch.no_grad():
                images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1).cpu().float()
            # denormalize the images and save to tensorboard
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images_processed = pipeline.numpy_to_pil(images)

            # Save the processed images to the output directory in PNG format
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for j, image in enumerate(images_processed):
                image_path = output_dir / f"sample_{process_number * num_batches * batch_size + i * batch_size + j}.png"
                image.save(image_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
