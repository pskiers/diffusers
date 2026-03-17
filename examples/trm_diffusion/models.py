import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class SpatialEncoder(nn.Module):
    """
    Compresses the high-resolution noisy image into a low-dimensional spatial grid.
    """

    def __init__(self, in_channels, out_channels, factor=4):
        super().__init__()
        self.factor = factor

        if factor == 1:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=factor, stride=factor, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels),
                nn.SiLU(),
            )

    def forward(self, x):
        return self.net(x)


class AttentiveBridge(nn.Module):
    """
    Perceiver IO style readout utilizing Flash Attention via PyTorch SDPA.
    High-resolution queries pull data from the low-resolution blueprint.
    """

    def __init__(self, in_channels, out_channels, out_resolution, factor=4, num_heads=4):
        super().__init__()
        self.factor = factor
        self.out_resolution = out_resolution

        if factor == 1:
            self.net = nn.Identity()
        else:
            self.query_dim = out_channels
            self.num_heads = num_heads
            self.head_dim = out_channels // num_heads
            assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

            # Positional queries for the high-res target grid.
            self.queries = nn.Parameter(
                torch.randn(1, out_resolution * out_resolution, self.query_dim) / math.sqrt(self.query_dim)
            )

            # Linear projections
            self.q_proj = nn.Linear(self.query_dim, self.query_dim)
            self.k_proj = nn.Linear(in_channels, self.query_dim)
            self.v_proj = nn.Linear(in_channels, self.query_dim)
            self.out_proj = nn.Linear(self.query_dim, self.query_dim)

    def forward(self, x_low):
        if self.factor == 1:
            return x_low

        B, C, H, W = x_low.shape
        N_high = self.out_resolution * self.out_resolution

        # 1. Flatten the Thinker's 2D grid: (B, C, H, W) -> (B, N_low, C)
        x_low_flat = x_low.view(B, C, -1).transpose(1, 2)

        # 2. Project Q, K, V
        Q_unproj = self.queries.expand(B, -1, -1)
        Q = self.q_proj(Q_unproj)
        K = self.k_proj(x_low_flat)
        V = self.v_proj(x_low_flat)

        # 3. Reshape for SDPA: (B, SeqLen, Heads, HeadDim) -> (B, Heads, SeqLen, HeadDim)
        Q = Q.view(B, N_high, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Flash Attention. PyTorch automatically uses the optimal backend (FlashAttention-2 if fp16/bf16 on CUDA)
        attn_out = F.scaled_dot_product_attention(Q, K, V)

        # 5. Reshape back and apply final projection
        attn_out = attn_out.transpose(1, 2).reshape(B, N_high, self.query_dim)
        out_flat = self.out_proj(attn_out)

        # 6. Reconstruct the 2D spatial grid for the Painter
        out_grid = out_flat.transpose(1, 2).view(B, self.query_dim, self.out_resolution, self.out_resolution)

        return out_grid


class ConditioningPyramid(nn.Module):
    """
    Extracts multi-scale features from a spatial blueprint for ControlNet-style
    injection into a diffusers UNet.
    """
    def __init__(self, in_channels, block_out_channels=(128, 256, 512, 512)):
        super().__init__()

        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i, out_channels in enumerate(block_out_channels):
            # First block doesn't downsample, just projects.
            # Subsequent blocks downsample by 2x.
            stride = 1 if i == 0 else 2

            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )
            )
            current_channels = out_channels

    def forward(self, blueprint):
        residuals = []
        x = blueprint
        for block in self.blocks:
            x = block(x)
            residuals.append(x)

        # Returns a tuple of feature maps from high-res to low-res
        return tuple(residuals)
