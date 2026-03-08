import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, Transformer2DModel
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


class UnifiedConditionDiT(Transformer2DModel):
    @register_to_config
    def __init__(
        self,
        condition_mode="class",  # "class", "class_adaln", or "sequence"
        num_classes=1000,
        raw_dim=21,
        # --- Standard Transformer2DModel args ---
        sample_size=32,
        in_channels=4,
        out_channels=4,
        num_layers=12,
        patch_size=2,
        attention_head_dim=64,
        num_attention_heads=16,
        cross_attention_dim=1024,
        activation_fn="gelu-approximate",
    ):
        # Configure norm and cross-attention based on mode
        is_adaln = condition_mode == "class_adaln"

        # FIX: DiTs ALWAYS require ada_norm_zero to process the diffusion timestep.
        norm_type = "ada_norm_zero"

        # If class_adaln, we need embeddings for all classes + 1 dropout token.
        # Otherwise, we just need 1 dummy embedding to carry the timestep.
        num_embeds_ada_norm = num_classes + 1 if is_adaln else 1

        # In adaLN mode, strip out cross-attention. Otherwise, use it.
        actual_cross_attn_dim = None if is_adaln else cross_attention_dim

        # Call parent explicitly
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            patch_size=patch_size,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=actual_cross_attn_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            norm_type=norm_type,
        )

        if self.config.condition_mode == "class":
            self.condition_projector = nn.Embedding(self.config.num_classes + 1, self.config.cross_attention_dim)
        elif self.config.condition_mode == "sequence":
            self.condition_projector = nn.Sequential(
                nn.Linear(self.config.raw_dim, self.config.cross_attention_dim),
                nn.SiLU(),
                nn.Linear(self.config.cross_attention_dim, self.config.cross_attention_dim),
            )
        elif self.config.condition_mode == "class_adaln":
            # Diffusers handles the adaLN embedding internally, so we don't need a projector
            self.condition_projector = None
        else:
            raise ValueError(f"Unknown condition_mode: {self.config.condition_mode}")

    def forward(self, sample, timestep, condition_tensors=None, attention_mask=None, **kwargs):
        """
        Generic forward pass matching your custom UNet signature.
        """
        encoder_hidden_states = None

        # Default dummy class_labels to carry the timestep through the adaLN block
        class_labels = torch.zeros((sample.shape[0],), dtype=torch.long, device=sample.device)

        if condition_tensors is not None:
            if self.config.condition_mode == "class":
                encoder_hidden_states = self.condition_projector(condition_tensors).unsqueeze(1)
            elif self.config.condition_mode == "sequence":
                encoder_hidden_states = self.condition_projector(condition_tensors)
            elif self.config.condition_mode == "class_adaln":
                # Override the dummy labels with the actual condition labels
                class_labels = condition_tensors
        elif self.config.condition_mode == "class_adaln":
            # Unconditional inference fallback for class_adaln
            class_labels = torch.full(
                (sample.shape[0],), self.config.num_classes, dtype=torch.long, device=sample.device
            )

        return super().forward(
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            encoder_attention_mask=attention_mask,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Bypass the buggy diffusers class remapping logic for custom classes.
        This forces diffusers to instantiate THIS class during EMA saving/loading.
        """
        init_dict, unused_kwargs, hidden_config_dict = cls.extract_init_dict(config, **kwargs)
        return cls(**init_dict)
