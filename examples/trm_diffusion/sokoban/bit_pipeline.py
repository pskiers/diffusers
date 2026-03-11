from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from sokoban.ddpm_scheduler import DDPMScheduler


class BitPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoqise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    # @torch.enable_grad()
    def __call__(
        self,
        prompt: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        num_images_per_prompt: Optional[int] = 1,
        cond_fn: Optional[callable] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`torch.Tensor`, *optional*):
                A tensor of shape `(batch_size, in_channels, height, width)` representing the prompt image. If not
                provided, random noise is sampled.
            class_labels (`torch.Tensor`, *optional*):
                A tensor of shape `(batch_size, )` representing the class labels for class conditional models.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        in_channels = self.unet.config.in_channels
        if prompt is not None:
            batch_size = prompt.shape[0] * num_images_per_prompt
            in_channels = in_channels // 2

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                in_channels,
                *self.unet.config.sample_size,
            )

        if self.unet.config.conditional and prompt is None:
            # prompt = torch.zeros(image_shape, device=self.device)
            raise ValueError("Prompt must be provided for conditional models.")

        if self.unet.config.class_conditional and class_labels is None:
            # class_labels = torch.randint(
            #     0, len(self.unet.config.distances), (batch_size,), device=self.device
            # )
            raise ValueError("Class labels must be provided for class conditional models.")

        # Prepare prompt for conditional models
        if prompt is not None:
            prompt = int2bits(prompt, in_channels, self.device, self.unet.dtype)
            prompt = (prompt * 2 - 1.0) * self.scheduler.config.clip_sample_range
            prompt = prompt.permute(0, 3, 1, 2)
            prompt = prompt.to(self.unet.dtype)


        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Sample gaussian noise to begin loop
        image = randn_tensor(
            image_shape,
            generator=generator,
            device=self._execution_device,
            dtype=self.unet.dtype,
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        with torch.no_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # TODO: Adapt to conditional
                if prompt is not None:
                    model_input = torch.cat([prompt, image], dim=1)
                else:
                    model_input = image

                model_output = self.unet(model_input, t, class_labels=class_labels).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1

                if isinstance(self.scheduler, DDIMScheduler):
                    # TODO: Add guidance
                    image = self.scheduler.step(
                        model_output,
                        t,
                        image,
                        eta=eta,
                        use_clipped_model_output=use_clipped_model_output,
                        generator=generator,
                        return_dict=return_dict,
                    ).prev_sample
                elif isinstance(self.scheduler, DDPMScheduler):
                    scheduler_out = self.scheduler.step(model_output, t, image, generator=generator)
                    model_mean = scheduler_out.prev_sample
                    variance_with_noise = scheduler_out.variance_with_noise
                    variance = scheduler_out.variance

                    # apply guidance if cond_fn is provided
                    if cond_fn is not None:
                        gradient = cond_fn(image, t)
                        model_mean = model_mean + (variance * gradient)

                    image = model_mean + variance_with_noise
                else:
                    raise ValueError(f"Scheduler {self.scheduler.__class__.__name__} is not supported.")

        image = image.cpu().permute(0, 2, 3, 1)
        image = bits2int(image > 0, device="cpu", out_dtype=torch.uint8)
        image = image.numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


# From https://github.com/google-research/pix2seq/blob/main/utils.py
def int2bits(x, n, device, out_dtype=None):
    """Convert an integer x in (...) into bits in (..., n)."""
    x = torch.bitwise_right_shift(torch.unsqueeze(x, -1), torch.arange(n).to(device))
    x = torch.remainder(x, 2)
    if out_dtype and out_dtype != x.dtype:
        x = x.to(out_dtype)
    return x


def bits2int(x, device, out_dtype):
    """Converts bits x in (..., n) into an integer in (...)."""
    x = x.to(out_dtype)
    x = torch.sum(x * (2 ** torch.arange(x.shape[-1]).to(device)), -1)
    return x


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
