import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers import UNet2DConditionModel, Transformer2DModel, UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.configuration_utils import register_to_config


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix inserted by torch.compile on submodules."""
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


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

    Architecture mirrors the conv backbone of MNISTCellClassifier:
      Conv(in → hidden[0]) → SiLU → MaxPool2
      Conv(hidden[0] → hidden[1]) → SiLU → MaxPool2
      ...
      AdaptiveAvgPool2d(grid_size)   ← grid_size = input_size // factor

    hidden_channels controls intermediate widths; defaults to [16, 32] which
    matches the classifier backbone.  out_channels is the final channel count
    (the last Conv in the stack projects to out_channels).

    factor=1 is a special case: the net is an identity (no downsampling).
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


class SpatialBridge(nn.Module):
    """
    Bilinear upsample + 2 conv layers.
    (B, in_c, H_t, W_t) → (B, bridge_c, painter_size, painter_size)
    Used for V0–V3.  V4 uses AttentiveBridge from models.py instead.
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


class ControlPainterUNet(UNet2DModel):
    """UNet2DModel extended with ControlNet-style residual injection.

    Identical to UNet2DModel in every way except forward() accepts
    down_block_additional_residuals and mid_block_additional_residual, which are
    added to the skip connections before the up-blocks (standard ControlNet math).
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
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
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


class SPADEGroupNorm(nn.Module):
    """GroupNorm + spatially adaptive scale/bias from a semantic map.

    h_out = gamma(s) * GroupNorm(h) + beta(s)
    where gamma, beta are predicted by a small CNN applied to s resized to h's size.
    """

    def __init__(self, num_groups: int, num_channels: int, sem_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        mid = max(num_channels, sem_channels)
        self.shared     = nn.Sequential(nn.Conv2d(sem_channels, mid, 3, padding=1), nn.SiLU())
        self.gamma_proj = nn.Conv2d(mid, num_channels, 3, padding=1)
        self.beta_proj  = nn.Conv2d(mid, num_channels, 3, padding=1)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        s_r    = F.interpolate(s, size=h.shape[-2:], mode="bilinear", align_corners=False)
        feat   = self.shared(s_r)
        return self.gamma_proj(feat) * h_norm + self.beta_proj(feat)


class SPADEResBlock(nn.Module):
    """ResNet block with both GroupNorms replaced by SPADE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sem_channels: int,
        temb_channels: int,
        norm_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = SPADEGroupNorm(norm_groups, in_channels, sem_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = SPADEGroupNorm(norm_groups, out_channels, sem_channels)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.act            = nn.SiLU()
        self.time_emb_proj  = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels))
        self.conv_shortcut  = (nn.Conv2d(in_channels, out_channels, 1)
                               if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, s))
        h = self.conv1(h)
        h = h + self.time_emb_proj(temb)[:, :, None, None]
        h = self.act(self.norm2(h, s))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _SPADEDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, sem_ch, num_layers, add_downsample, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList([
            SPADEResBlock(in_ch if i == 0 else out_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout)
            for i in range(num_layers)
        ])
        self.downsamplers = (
            nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)])
            if add_downsample else None
        )

    def forward(self, hidden, temb, s):
        outputs = ()
        for r in self.resnets:
            hidden = r(hidden, temb, s);  outputs += (hidden,)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                hidden = d(hidden)
            outputs += (hidden,)
        return hidden, outputs


class _SPADEMidBlock(nn.Module):
    def __init__(self, channels, temb_ch, sem_ch, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList([
            SPADEResBlock(channels, channels, sem_ch, temb_ch, norm_groups, dropout)
            for _ in range(2)   # UNetMidBlock2D default: num_layers=1 → 2 resnets
        ])

    def forward(self, hidden, temb, s):
        for r in self.resnets:
            hidden = r(hidden, temb, s)
        return hidden


class _SPADEUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, prev_out_ch, temb_ch, sem_ch, num_layers, add_upsample, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            # Mirrors diffusers UpBlock2D channel formula exactly
            res_skip_ch = in_ch if (i == num_layers - 1) else out_ch
            res_in_ch   = prev_out_ch if i == 0 else out_ch
            self.resnets.append(
                SPADEResBlock(res_in_ch + res_skip_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout)
            )
        self.upsamplers = (
            nn.ModuleList([nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )])
            if add_upsample else None
        )

    def forward(self, hidden, res_tuple, temb, s):
        for r in self.resnets:
            skip       = res_tuple[-1]
            res_tuple  = res_tuple[:-1]
            hidden     = r(torch.cat([hidden, skip], dim=1), temb, s)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                hidden = up(hidden)
        return hidden


class SPADEUNet2D(nn.Module):
    """UNet2DModel with SPADE normalization throughout.

    Takes an extra semantic map `s` (B, sem_channels, H, W) in forward().
    Each SPADEGroupNorm bilinearly resizes `s` to the current feature map
    resolution — no pyramid required.

    Architecturally matches _make_painter_control (in_channels=1, no concat).
    """

    def __init__(
        self,
        painter_size: int,
        sem_channels: int,
        block_out_channels: tuple[int, ...] = (32, 64, 128, 256),
        layers_per_block: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        norm_groups = 32
        while norm_groups > 1 and any(c % norm_groups != 0 for c in block_out_channels):
            norm_groups //= 2

        ch0    = block_out_channels[0]
        temb_ch = ch0 * 4

        self.time_proj      = Timesteps(ch0, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(ch0, temb_ch)
        self.conv_in        = nn.Conv2d(1, ch0, 3, padding=1)

        # Down blocks
        n = len(block_out_channels)
        self.down_blocks = nn.ModuleList()
        cur_ch = ch0
        for i, out_ch in enumerate(block_out_channels):
            self.down_blocks.append(_SPADEDownBlock(
                in_ch=cur_ch, out_ch=out_ch, temb_ch=temb_ch, sem_ch=sem_channels,
                num_layers=layers_per_block, add_downsample=(i < n - 1),
                norm_groups=norm_groups, dropout=dropout,
            ))
            cur_ch = out_ch

        # Mid block
        self.mid_block = _SPADEMidBlock(
            channels=cur_ch, temb_ch=temb_ch, sem_ch=sem_channels,
            norm_groups=norm_groups, dropout=dropout,
        )

        # Up blocks (mirrors UNet2DModel channel formula)
        rev = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        prev_out_ch = rev[0]
        for i, out_ch in enumerate(rev):
            in_skip_ch = rev[min(i + 1, n - 1)]
            self.up_blocks.append(_SPADEUpBlock(
                in_ch=in_skip_ch, out_ch=out_ch, prev_out_ch=prev_out_ch,
                temb_ch=temb_ch, sem_ch=sem_channels,
                num_layers=layers_per_block + 1, add_upsample=(i < n - 1),
                norm_groups=norm_groups, dropout=dropout,
            ))
            prev_out_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(norm_groups, ch0)
        self.conv_act      = nn.SiLU()
        self.conv_out      = nn.Conv2d(ch0, 1, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep, s: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        emb = self.time_embedding(self.time_proj(timestep).to(sample.dtype))

        x = self.conv_in(sample)
        skips = (x,)
        for block in self.down_blocks:
            x, res = block(x, emb, s);  skips += res

        x = self.mid_block(x, emb, s)

        for block in self.up_blocks:
            n_skip   = len(block.resnets)
            res_tup  = skips[-n_skip:]
            skips    = skips[:-n_skip]
            x = block(x, res_tup, emb, s)

        return self.conv_out(self.conv_act(self.conv_norm_out(x)))
