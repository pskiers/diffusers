import argparse
import inspect
import logging
import math
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

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMScheduler, DDIMPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available


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
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
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
    parser.add_argument("--T", type=int, default=3, help="T")
    parser.add_argument("--n", type=int, default=6, help="n")
    parser.add_argument("--N_supervision", type=int, default=4, help="N_supervision")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM instead of DDPM.", default=False)

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

    T=args.T
    n=args.n
    N_supervision=args.N_supervision

    sample_size = args.resolution

    model = UNet2DModel.from_pretrained(args.model_config_name_or_path)
    model.y_init = torch.nn.Buffer(trunc_normal_init_(torch.empty((1, 3, sample_size, sample_size), dtype=model.dtype), std=1), persistent=True)
    model.z_init = torch.nn.Buffer(trunc_normal_init_(torch.empty((1, 3, sample_size, sample_size), dtype=model.dtype), std=1), persistent=True)

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
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDIMScheduler.__init__).parameters.keys())
    SchedulerClass = DDIMScheduler if args.use_ddim else DDPMScheduler
    if accepts_prediction_type:
        noise_scheduler = SchedulerClass(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = SchedulerClass(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)


    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(model)

    model.old_forward = model.forward
    def latent_recursion(model, x, y, z, timesteps, n=6):
        for _ in range(n):
            _, z = model.old_forward(torch.cat([x, y, z], dim=1), timesteps).sample.chunk(2, dim=1)
        y, _ = model.old_forward(torch.cat([x, y, z], dim=1), timesteps).sample.chunk(2, dim=1)
        return y, z

    def deep_recursion(model, x, y, z, timesteps, n=6, T=3):
        x = x[:, :3]
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = latent_recursion(model, x, y, z, timesteps, n)
        y, z = latent_recursion(model, x, y, z, timesteps, n)
        return y, y.detach(), z.detach()

    unet = accelerator.unwrap_model(model)

    class OutputUnet:
        pass
    def new_forward(sample, timesteps, *args, **kwargs):
        y = torch.cat([model.module.y_init for _ in range(sample.shape[0])], dim=0).to(unet.device)
        z = torch.cat([model.module.z_init for _ in range(sample.shape[0])], dim=0).to(unet.device)
        for _ in range(N_supervision):
            model_output, y, z = deep_recursion(unet, sample, y, z, timesteps, n, T)
        output = OutputUnet()
        output.sample = model_output
        return output
    unet.old_forward = unet.forward
    unet.forward = new_forward
    unet.config.in_channels = 3

    PipelineCls = DDIMPipeline if args.use_ddim else DDPMPipeline
    pipeline = PipelineCls(
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
            images = pipeline(
                generator=generator,
                batch_size=batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
            ).images

            # denormalize the images and save to tensorboard
            images_processed = (images * 255).round().astype("uint8")

            # Save the processed images to the output directory in PNG format
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for j, image in enumerate(images_processed):
                image_path = output_dir / f"sample_{process_number * num_batches * batch_size + i * batch_size + j}.png"
                Image.fromarray(image).save(image_path)

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


if __name__ == "__main__":
    args = parse_args()
    main(args)
