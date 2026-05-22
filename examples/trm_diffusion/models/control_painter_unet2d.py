import torch
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput


class ControlPainterUNet(UNet2DModel):
    """
    UNet2DModel extended with ControlNet-style residual injection.

    Identical to UNet2DModel in every way except forward() accepts down_block_additional_residuals
    and mid_block_additional_residual, which are added to the skip connections before the up-blocks
    (standard ControlNet math).
    """

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        class_labels=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
    ):
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        t_emb = self.time_proj(timestep).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels required for class conditioning")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            emb = emb + self.class_embedding(class_labels).to(dtype=self.dtype)

        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down = ()
            for orig, add in zip(down_block_res_samples, down_block_additional_residuals):
                new_down += (orig + add,)
            down_block_res_samples = new_down

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timestep = timestep.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timestep

        if not return_dict:
            return (sample,)
        return UNet2DOutput(sample=sample)
