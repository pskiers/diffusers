import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix inserted by torch.compile on submodules."""
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


class TimestepMLP(nn.Module):
    """
    Sinusoidal timestep embedding followed by a two-layer MLP.

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
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class SpatialEncoder(nn.Module):
    """
    Compresses the high-resolution noisy image into a low-dimensional spatial grid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 4,
        hidden_channels: tuple = (16, 32),
    ):
        super().__init__()
        self.factor = factor

        if factor == 1:
            self.net = nn.Identity()
        else:
            layers = []
            ch = in_channels
            for h in hidden_channels:
                layers += [
                    nn.Conv2d(ch, h, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.MaxPool2d(2),
                ]
                ch = h
            layers += [
                nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, out_channels), out_channels),
                nn.SiLU(),
            ]
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        grid_size = x.shape[-1] // self.factor
        return nn.functional.adaptive_avg_pool2d(self.net(x), grid_size)


class AttentiveBridge(nn.Module):
    """
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


class SpatialBridge(nn.Module):
    """
    Bilinear upsample + 2 conv layers.
    (B, in_c, H_t, W_t) → (B, bridge_c, painter_size, painter_size)
    """

    def __init__(self, in_channels: int, out_channels: int, painter_size: int):
        super().__init__()
        self.painter_size = painter_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.painter_size, mode="bilinear", align_corners=False)
        return self.conv(x)
