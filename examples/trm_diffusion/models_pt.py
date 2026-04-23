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
        # Configure norm and cross-attention based on mode.
        # "spatial_concat" concatenates the condition map to the noisy input before
        # patch-embedding, so it also needs no cross-attention.
        is_adaln = condition_mode in ("class_adaln", "spatial_concat")

        # FIX: DiTs ALWAYS require ada_norm_zero to process the diffusion timestep.
        norm_type = "ada_norm_zero"

        # class_adaln uses all classes + dropout token; everything else just needs
        # a single dummy embedding to carry the timestep through adaLN.
        num_embeds_ada_norm = num_classes + 1 if condition_mode == "class_adaln" else 1

        # In adaLN / spatial_concat mode, strip out cross-attention. Otherwise, use it.
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
        elif self.config.condition_mode == "spatial_concat":
            # Condition is a (C, H, W) spatial map concatenated to the noisy input
            # before patch-embedding; no projector needed.
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
            elif self.config.condition_mode == "spatial_concat":
                # Concatenate the spatial map to the noisy input before patch-embedding.
                # condition_tensors: (B, C_mask, H, W); sample: (B, C_noise, H, W)
                sample = torch.cat([sample, condition_tensors.to(sample.dtype)], dim=1)
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


class TimestepMLP(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP.

    Produces a continuous embedding of the diffusion timestep.  Initialise
    the downstream projection weights to zero so the model starts as the
    identity and gradually learns to use the signal.

    Args:
        sin_dim:  dimension of the sinusoidal embedding (input to MLP).
        out_dim:  output dimension (used for FiLM projections / T2 additions).
    """

    def __init__(self, sin_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.sin_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


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
    injection. Dynamically unrolls to match the target UNet's exact layer counts.
    """

    def __init__(self, in_channels, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        self.layers_per_block = layers_per_block

        # 1. Match the UNet's initial conv_in
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        current_channels = block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            block_modules = nn.ModuleList()

            # A. Channel projection if transitioning to wider blocks
            if current_channels != out_channels:
                block_modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
                current_channels = out_channels
            else:
                block_modules.append(nn.Identity())

            # B. ResNet equivalents (one for every layer_per_block)
            for _ in range(layers_per_block):
                block_modules.append(
                    nn.Sequential(
                        nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(min(32, current_channels), current_channels),
                        nn.SiLU(),
                    )
                )

            # C. Downsampler (if not the last block)
            if i < len(block_out_channels) - 1:
                block_modules.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))

            self.blocks.append(block_modules)

        # 2. Mid block features
        self.mid_block = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, current_channels), current_channels),
            nn.SiLU(),
        )

    def forward(self, blueprint):
        residuals = []

        # Initial layer
        x = self.conv_in(blueprint)
        residuals.append(x)

        # Down blocks
        for i, block in enumerate(self.blocks):
            x = block[0](x)  # Project channels

            for j in range(self.layers_per_block):
                x = x + block[1 + j](x)  # Lightweight residual step
                residuals.append(x)

            if i < len(self.blocks) - 1:
                downsampler = block[1 + self.layers_per_block]
                x = downsampler(x)
                residuals.append(x)

        # Mid block
        mid_res = x + self.mid_block(x)

        return residuals, mid_res
