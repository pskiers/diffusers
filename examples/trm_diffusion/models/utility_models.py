import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix inserted by torch.compile on submodules."""
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Load weights from a checkpoint written by train_trm.py.

    Format: {"step": int, "model_state": ..., "ema_state": {"shadow": ...}, ...}

    Falls back to loading the file directly as a state_dict if no known keys
    are found (plain torch.save(model.state_dict(), path)).

    Returns the training step recorded in the checkpoint, or None.

    Shared by eval.py and trajectory_viz.py — not named eval.<this> because
    examples/trm_diffusion also has an eval/ package (eval/steiner_eval.py
    etc.), and `from eval import ...` ambiguously resolves to that package
    rather than the eval.py script.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        step = ckpt.get("step", None)

        # Always load model_state first — covers frozen params and buffers
        # (e.g. H_init, L_init) that EMA doesn't track.
        sd = strip_compiled_prefix(ckpt["model_state"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded model_state (step={step}): missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            logger.info(f"  Missing (first 10): {missing[:10]}")
        if unexpected:
            logger.info(f"  Unexpected (first 10): {unexpected[:10]}")

        if use_ema and ckpt.get("ema_state") is not None:
            # EMAHelper.state_dict() returns self.shadow directly:
            # {param_name: tensor} — no extra nesting.
            ema_state = ckpt["ema_state"]
            if isinstance(ema_state, dict) and ema_state:
                ema_sd = strip_compiled_prefix(ema_state)
                missing, unexpected = model.load_state_dict(ema_sd, strict=False)
                logger.info(
                    f"Loaded EMA weights on top of model_state "
                    f"({len(ema_sd)} EMA params, missing={len(missing)})"
                )
                if missing:
                    logger.info(f"  Missing (first 5): {missing[:5]}")
                return step
            logger.warning("EMA state is empty — using raw model_state")
        return step

    # Fallback: raw state_dict
    sd = strip_compiled_prefix(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded raw state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
    return None


@torch.no_grad()
def recalibrate_batchnorm(model, dataloader, device, n_batches: int) -> int:
    """Reset every BatchNorm's running_mean/running_var/num_batches_tracked
    (nn.BatchNorm2d.reset_running_stats()) and re-accumulate them from
    n_batches of real train()-mode forward passes — no gradients, no weight
    updates, only the running-stat buffers change. Fixes (and was used to
    empirically confirm) a checkpoint whose running stats drifted to
    extreme values under training instability — see models/paper_unet.py.
    Shared by eval.py and trajectory_viz.py.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bn_modules:
        m.reset_running_stats()

    was_training = model.training
    model.train()
    dl_iter = iter(dataloader)
    for _ in range(n_batches):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        sample = model._prepare_training_sample(batch, device)
        model(sample)
    model.train(was_training)
    return len(bn_modules)


def load_frozen_custom_kl_vae(
    checkpoint_path: str,
    in_channels: int = 1,
    out_channels: int = 1,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(32, 64, 128),
    layers_per_block: int = 2,
    latent_channels: int = 4,
    norm_num_groups: int = 32,
    act_fn: str = "silu",
):
    """Construct an AutoencoderKL from scratch and load weights from a local checkpoint.

    Intended for custom-trained VAEs (e.g. MNIST pixel-space) saved by train_vae.py.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=list(down_block_types),
        up_block_types=list(up_block_types),
        block_out_channels=list(block_out_channels),
        layers_per_block=layers_per_block,
        latent_channels=latent_channels,
        norm_num_groups=norm_num_groups,
        act_fn=act_fn,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vae.load_state_dict(strip_compiled_prefix(ckpt["model_state"]))
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_frozen_vae(pretrained_model_name_or_path: str):
    """Load an AutoencoderKL from a pretrained checkpoint and freeze it.

    Thin wrapper so Hydra instantiate can build the VAE via _target_.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


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

    Zero convolutions (1×1 convs, weight and bias initialised to 0) are applied
    at every residual output point and at the mid-block output.  This is the key
    ControlNet trick: at init the pyramid injects exactly zero into the frozen
    backbone, so the backbone behaves identically to how it was trained.
    Gradients gradually "unlock" the zero convs during fine-tuning.
    """

    def __init__(self, in_channels, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        self.layers_per_block = layers_per_block

        # 1. Match the UNet's initial conv_in
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        current_channels = block_out_channels[0]

        # Track output channel at each residual point to build zero-conv list.
        residual_channels: list[int] = [block_out_channels[0]]  # conv_in output

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
                residual_channels.append(current_channels)

            # C. Downsampler (if not the last block)
            if i < len(block_out_channels) - 1:
                block_modules.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))
                residual_channels.append(current_channels)

            self.blocks.append(block_modules)

        # 2. Mid block features
        self.mid_block = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, current_channels), current_channels),
            nn.SiLU(),
        )

        # 3. Zero convolutions — one per residual output + one for mid.
        #    All weights and biases initialised to exactly zero so the pyramid
        #    injects nothing at the start of training.
        self.zero_convs = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1) for c in residual_channels])
        self.mid_zero_conv = nn.Conv2d(current_channels, current_channels, kernel_size=1)
        for m in list(self.zero_convs) + [self.mid_zero_conv]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, blueprint):
        residuals = []
        zero_idx = 0

        # Initial layer
        x = self.conv_in(blueprint)
        residuals.append(self.zero_convs[zero_idx](x))
        zero_idx += 1

        # Down blocks
        for i, block in enumerate(self.blocks):
            x = block[0](x)  # Project channels

            for j in range(self.layers_per_block):
                x = x + block[1 + j](x)  # Lightweight residual step
                residuals.append(self.zero_convs[zero_idx](x))
                zero_idx += 1

            if i < len(self.blocks) - 1:
                downsampler = block[1 + self.layers_per_block]
                x = downsampler(x)
                residuals.append(self.zero_convs[zero_idx](x))
                zero_idx += 1

        # Mid block
        mid_res = x + self.mid_block(x)

        return residuals, self.mid_zero_conv(mid_res)


class ConditioningPyramid1D(nn.Module):
    """1D ControlNet-style conditioning pyramid for ConditionalUnet1D
    (models/action_backbones.py), used via ControlPainterUNet1D.

    Unlike ConditioningPyramid (which mirrors UNet2DModel's one-residual-
    per-resnet-layer structure), ConditionalUnet1D only produces ONE skip
    connection per U-Net *level* (after both of that level's resnets, before
    downsampling) — so this produces exactly one residual per entry in
    block_out_channels plus one mid-block residual, using the identical
    stride-2 Conv1d downsampling (kernel=3, stride=2, padding=1, matching
    Downsample1d exactly) so every residual's temporal length lines up
    exactly with the frozen backbone's skip connections, regardless of the
    input sequence length.

    Zero convolutions (1×1, weight+bias zero-init) on every output — at
    init this injects exactly zero, same ControlNet trick as ConditioningPyramid.
    """

    def __init__(self, in_channels, block_out_channels=(256, 512, 1024), n_groups=8):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            self.down_blocks.append(nn.ModuleDict({
                "proj": (
                    nn.Conv1d(current_channels, out_channels, kernel_size=1)
                    if current_channels != out_channels else nn.Identity()
                ),
                "res": nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(min(n_groups, out_channels), out_channels),
                    nn.SiLU(),
                ),
                # Matches ConditionalUnet1D.Downsample1d exactly.
                "downsample": (
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                    if not is_last else nn.Identity()
                ),
            }))
            current_channels = out_channels

        self.mid_block = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(n_groups, current_channels), current_channels),
            nn.SiLU(),
        )

        self.zero_convs = nn.ModuleList([nn.Conv1d(c, c, kernel_size=1) for c in block_out_channels])
        self.mid_zero_conv = nn.Conv1d(current_channels, current_channels, kernel_size=1)
        for m in list(self.zero_convs) + [self.mid_zero_conv]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, blueprint):
        x = self.conv_in(blueprint)
        residuals = []
        for i, layer in enumerate(self.down_blocks):
            x = layer["proj"](x)
            x = x + layer["res"](x)
            residuals.append(self.zero_convs[i](x))
            x = layer["downsample"](x)

        mid_res = x + self.mid_block(x)
        return residuals, self.mid_zero_conv(mid_res)


class ConditioningPyramidPaperUNet(nn.Module):
    """2D ControlNet-style conditioning pyramid for PaperUNet (models/paper_unet.py),
    used via ControlPainterPaperUNet.

    Like ConditioningPyramid1D (not ConditioningPyramid): PaperUNet captures
    exactly ONE skip per level, before downsampling — so this produces one
    residual per entry in block_out_channels plus one mid-block residual.
    Unlike ConditioningPyramid1D, every level downsamples (PaperUNet has no
    "skip the last downsample" step — the bottleneck always operates on the
    fully-downsampled result), so there is no is_last exception here.

    Zero convolutions (1x1, weight+bias zero-init) on every output — at init
    this injects exactly zero, same ControlNet trick as ConditioningPyramid.
    """

    def __init__(self, in_channels, block_out_channels=(64, 128, 256, 512), n_groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_channels = block_out_channels[0]
        for out_channels in block_out_channels:
            self.down_blocks.append(nn.ModuleDict({
                "proj": (
                    nn.Conv2d(current_channels, out_channels, kernel_size=1)
                    if current_channels != out_channels else nn.Identity()
                ),
                "res": nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(min(n_groups, out_channels), out_channels),
                    nn.SiLU(),
                ),
                # Matches PaperUNet's own downsamples: stride-2 conv at every level.
                "downsample": nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            }))
            current_channels = out_channels

        self.mid_block = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(n_groups, current_channels), current_channels),
            nn.SiLU(),
        )

        self.zero_convs = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1) for c in block_out_channels])
        self.mid_zero_conv = nn.Conv2d(current_channels, current_channels, kernel_size=1)
        for m in list(self.zero_convs) + [self.mid_zero_conv]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, blueprint):
        x = self.conv_in(blueprint)
        residuals = []
        for i, layer in enumerate(self.down_blocks):
            x = layer["proj"](x)
            x = x + layer["res"](x)
            residuals.append(self.zero_convs[i](x))
            x = layer["downsample"](x)

        mid_res = x + self.mid_block(x)
        return residuals, self.mid_zero_conv(mid_res)


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
