import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import Timesteps, TimestepEmbedding


class SPADEGroupNorm(nn.Module):
    """GroupNorm + spatially adaptive scale/bias from a semantic map.

    h_out = gamma(s) * GroupNorm(h) + beta(s)
    where gamma, beta are predicted by a small CNN applied to s resized to h's size.
    """

    def __init__(self, num_groups: int, num_channels: int, sem_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        mid = max(num_channels, sem_channels)
        self.shared = nn.Sequential(nn.Conv2d(sem_channels, mid, 3, padding=1), nn.SiLU())
        self.gamma_proj = nn.Conv2d(mid, num_channels, 3, padding=1)
        self.beta_proj = nn.Conv2d(mid, num_channels, 3, padding=1)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        s_r = F.interpolate(s, size=h.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.shared(s_r)
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
        self.act = nn.SiLU()
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels))
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

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
        self.resnets = nn.ModuleList(
            [
                SPADEResBlock(in_ch if i == 0 else out_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout)
                for i in range(num_layers)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)]) if add_downsample else None
        )

    def forward(self, hidden, temb, s):
        outputs = ()
        for r in self.resnets:
            hidden = r(hidden, temb, s)
            outputs += (hidden,)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                hidden = d(hidden)
            outputs += (hidden,)
        return hidden, outputs


class _SPADEMidBlock(nn.Module):
    def __init__(self, channels, temb_ch, sem_ch, norm_groups, dropout):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                SPADEResBlock(channels, channels, sem_ch, temb_ch, norm_groups, dropout)
                for _ in range(2)  # UNetMidBlock2D default: num_layers=1 → 2 resnets
            ]
        )

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
            res_in_ch = prev_out_ch if i == 0 else out_ch
            self.resnets.append(SPADEResBlock(res_in_ch + res_skip_ch, out_ch, sem_ch, temb_ch, norm_groups, dropout))
        self.upsamplers = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    )
                ]
            )
            if add_upsample
            else None
        )

    def forward(self, hidden, res_tuple, temb, s):
        for r in self.resnets:
            skip = res_tuple[-1]
            res_tuple = res_tuple[:-1]
            hidden = r(torch.cat([hidden, skip], dim=1), temb, s)
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

        ch0 = block_out_channels[0]
        temb_ch = ch0 * 4

        self.time_proj = Timesteps(ch0, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(ch0, temb_ch)
        self.conv_in = nn.Conv2d(1, ch0, 3, padding=1)

        # Down blocks
        n = len(block_out_channels)
        self.down_blocks = nn.ModuleList()
        cur_ch = ch0
        for i, out_ch in enumerate(block_out_channels):
            self.down_blocks.append(
                _SPADEDownBlock(
                    in_ch=cur_ch,
                    out_ch=out_ch,
                    temb_ch=temb_ch,
                    sem_ch=sem_channels,
                    num_layers=layers_per_block,
                    add_downsample=(i < n - 1),
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            cur_ch = out_ch

        # Mid block
        self.mid_block = _SPADEMidBlock(
            channels=cur_ch,
            temb_ch=temb_ch,
            sem_ch=sem_channels,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        # Up blocks (mirrors UNet2DModel channel formula)
        rev = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        prev_out_ch = rev[0]
        for i, out_ch in enumerate(rev):
            in_skip_ch = rev[min(i + 1, n - 1)]
            self.up_blocks.append(
                _SPADEUpBlock(
                    in_ch=in_skip_ch,
                    out_ch=out_ch,
                    prev_out_ch=prev_out_ch,
                    temb_ch=temb_ch,
                    sem_ch=sem_channels,
                    num_layers=layers_per_block + 1,
                    add_upsample=(i < n - 1),
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            prev_out_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(norm_groups, ch0)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch0, 1, 3, padding=1)

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
            x, res = block(x, emb, s)
            skips += res

        x = self.mid_block(x, emb, s)

        for block in self.up_blocks:
            n_skip = len(block.resnets)
            res_tup = skips[-n_skip:]
            skips = skips[:-n_skip]
            x = block(x, res_tup, emb, s)

        return self.conv_out(self.conv_act(self.conv_norm_out(x)))
