import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config


class UnifiedConditionUNet(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,
        condition_mode="class",  # "class" (ImageNet) or "sequence" (CLEVR)
        num_classes=1000,
        raw_dim=21,
        # --- Explicitly unroll standard UNet args so ConfigMixin captures them properly ---
        sample_size=64,
        in_channels=3,
        out_channels=3,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(256,),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1024,
        attention_head_dim=8,
    ):
        # Call parent explicitly, skipping num_class_embeds so we don't trigger native class logic
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

        # Note: We use self.config here because @register_to_config puts arguments there
        if self.config.condition_mode == "class":
            self.condition_projector = nn.Embedding(self.config.num_classes + 1, self.config.cross_attention_dim)
        elif self.config.condition_mode == "sequence":
            self.condition_projector = nn.Sequential(
                nn.Linear(self.config.raw_dim, self.config.cross_attention_dim),
                nn.SiLU(),
                nn.Linear(self.config.cross_attention_dim, self.config.cross_attention_dim),
            )
        else:
            raise ValueError(f"Unknown condition_mode: {self.config.condition_mode}")

    def forward(self, sample, timestep, condition_tensors, attention_mask=None, **kwargs):
        """
        Generic forward pass handling both discrete classes and continuous sequences.
        """
        if self.config.condition_mode == "class":
            encoder_hidden_states = self.condition_projector(condition_tensors).unsqueeze(1)
        else:
            encoder_hidden_states = self.condition_projector(condition_tensors)

        return super().forward(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            **kwargs,
        )
