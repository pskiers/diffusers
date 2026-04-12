"""
Ratatouille model variants for MNIST Sudoku.

Architecture overview:
  Painter  – UNet2DModel in pixel space (1-channel grayscale).
  Thinker  – a CNN at (possibly compressed) resolution; produces per-cell features.
  Bridge   – bilinear upsample + conv → bridge_channels feature maps concatenated
             to the noisy image before the painter.

Five variants (Model 0–4):

  V0: Thinker at full painter resolution; per-cell avg-pool → digit logits.
      Losses: diffusion + weighted sudoku CE.

  V1: Encoder compresses condition to 9×9; thinker outputs digit logits per cell.
      Losses: diffusion + weighted sudoku CE.

  V2: Same as V1 but no sudoku loss.

  V3: Same as V1 but thinker_out_channels >> 9 (no sudoku loss).

  V4: Same as V3 but compression_factor ≠ cell_size → thinker grid size ≠ 9.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


# ── Building blocks ─────────────────────────────────────────────────────────────

class MultiScaleEncoder(nn.Module):
    """
    Strided-conv chain: halves H,W `n_halvings` times.
    (B, in_c, H, W) → (B, out_c, H/2^n, W/2^n)
    """

    def __init__(self, in_channels: int, out_channels: int, n_halvings: int,
                 base_channels: int = 16):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for i in range(n_halvings):
            out = base_channels * (2 ** i) if i < n_halvings - 1 else out_channels
            layers += [
                nn.Conv2d(ch, out, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, out), out),
                nn.SiLU(),
            ]
            ch = out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SudokuThinkerCNN(nn.Module):
    """
    Plain CNN at thinker resolution.
    (B, in_c, H_t, W_t) → (B, out_c, H_t, W_t)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.SiLU(),
            ]
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SudokuBridge(nn.Module):
    """
    Upsample thinker feature map to painter resolution, then conv to bridge_channels.
    (B, thinker_out_c, H_t, W_t) → (B, bridge_c, painter_size, painter_size)
    """

    def __init__(self, thinker_out_channels: int, bridge_channels: int, painter_size: int):
        super().__init__()
        self.painter_size = painter_size
        self.conv = nn.Sequential(
            nn.Conv2d(thinker_out_channels, bridge_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(bridge_channels, bridge_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.painter_size, mode="bilinear", align_corners=False)
        return self.conv(x)


def _make_painter(
    painter_size: int,
    bridge_channels: int,
    painter_channels: tuple[int, ...],
) -> UNet2DModel:
    n = len(painter_channels)
    # Find the largest norm group count ≤ 32 that divides every channel value.
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return UNet2DModel(
        sample_size=painter_size,
        in_channels=1 + bridge_channels,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=tuple("DownBlock2D" if i == 0 else "AttnDownBlock2D" for i in range(n)),
        up_block_types=tuple("UpBlock2D" if i == n - 1 else "AttnUpBlock2D" for i in range(n)),
        norm_num_groups=norm_num_groups,
    )


# ── Base class ─────────────────────────────────────────────────────────────────

class _RatatouilleBase(nn.Module):
    """
    Subclasses implement `thinker_forward` which returns
    (bridge_features (B,bc,H_p,W_p), sudoku_logits or None).
    """

    def __init__(self, painter: UNet2DModel, bridge: SudokuBridge):
        super().__init__()
        self.painter = painter
        self.bridge  = bridge

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def forward(
        self,
        noisy_images: torch.Tensor,   # (B, 1, H, W)
        timesteps: torch.Tensor,      # (B,)
        condition: torch.Tensor,      # (B, 1, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            noise_pred    – (B, 1, H, W)
            sudoku_logits – (B, 81, num_classes) or None
        """
        bridge_feat, sudoku_logits = self.thinker_forward(condition)
        painter_input = torch.cat([noisy_images, bridge_feat], dim=1)
        noise_pred = self.painter(painter_input, timesteps).sample
        return noise_pred, sudoku_logits


# ── Model 0 ────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV0(_RatatouilleBase):
    """
    Thinker sees condition at FULL painter resolution (no encoder compression).
    Per-cell avg-pool (cell_size × cell_size → 1 vector) → digit logits (B,81,9).
    Losses: diffusion + weighted sudoku CE.
    """

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        thinker_hidden: int = 64,
        thinker_layers: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        bridge  = SudokuBridge(thinker_hidden, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)
        super().__init__(painter=painter, bridge=bridge)

        self.thinker    = SudokuThinkerCNN(1, thinker_hidden, thinker_hidden, thinker_layers)
        self.logit_head = nn.Linear(thinker_hidden, num_classes)
        self._cell_size  = cell_size
        self._num_classes = num_classes

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = condition.shape[0]
        feat = self.thinker(condition)       # (B, thinker_hidden, H, W)

        # Per-cell avg-pool: (B, thinker_hidden, 9, 9)
        cell = F.avg_pool2d(feat, kernel_size=self._cell_size, stride=self._cell_size)
        logits = self.logit_head(cell.permute(0, 2, 3, 1))   # (B,9,9,C)
        logits = logits.reshape(B, 81, self._num_classes)

        # Bridge: upsample thinker features (already full-res, no-op upsample) → bridge
        bridge_out = self.bridge(feat)       # (B, bridge_channels, H, W)
        return bridge_out, logits


# ── Shared base for Models 1-4 ─────────────────────────────────────────────────

class _CompressedBase(_RatatouilleBase):
    """
    Models that first compress the condition with a MultiScaleEncoder.
    """

    def __init__(
        self,
        encoder: MultiScaleEncoder,
        thinker: SudokuThinkerCNN,
        bridge: SudokuBridge,
        painter: UNet2DModel,
    ):
        super().__init__(painter=painter, bridge=bridge)
        self.encoder = encoder
        self.thinker = thinker

    def _encode(self, condition: torch.Tensor) -> torch.Tensor:
        return self.thinker(self.encoder(condition))


# ── Model 1 ────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV1(_CompressedBase):
    """
    Encoder compresses condition to 9×9 (compression_factor = cell_size).
    Thinker outputs num_classes channels → per-cell digit logits (B,81,9).
    Losses: diffusion + weighted sudoku CE.
    """

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_out_channels: int = 16,
        bridge_channels: int = 8,
        thinker_hidden: int = 64,
        thinker_layers: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        n_halvings = int(round(math.log2(cell_size)))
        encoder  = MultiScaleEncoder(1, enc_out_channels, n_halvings)
        thinker  = SudokuThinkerCNN(enc_out_channels, num_classes, thinker_hidden, thinker_layers)
        # Adapter to map from num_classes channels to bridge_channels
        bridge   = SudokuBridge(num_classes, bridge_channels, painter_size)
        painter  = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)
        self._num_classes = num_classes

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B   = condition.shape[0]
        feat = self._encode(condition)           # (B, num_classes, 9, 9)
        logits = feat.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)
        bridge_out = self.bridge(feat)           # (B, bridge_channels, H_p, W_p)
        return bridge_out, logits


# ── Model 2 ────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV2(_CompressedBase):
    """
    Same as V1 but no sudoku loss. Thinker out = bridge_channels.
    """

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_out_channels: int = 16,
        bridge_channels: int = 8,
        thinker_hidden: int = 64,
        thinker_layers: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        n_halvings = int(round(math.log2(cell_size)))
        encoder = MultiScaleEncoder(1, enc_out_channels, n_halvings)
        thinker = SudokuThinkerCNN(enc_out_channels, bridge_channels, thinker_hidden, thinker_layers)
        bridge  = SudokuBridge(bridge_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        feat = self._encode(condition)       # (B, bridge_channels, 9, 9)
        return self.bridge(feat), None


# ── Model 3 ────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV3(_CompressedBase):
    """
    Same as V2 but thinker_out_channels >> 9 (richer thinker representation).
    Still compresses to 9×9. No sudoku loss.
    """

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_out_channels: int = 16,
        thinker_out_channels: int = 64,
        bridge_channels: int = 8,
        thinker_hidden: int = 64,
        thinker_layers: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        n_halvings = int(round(math.log2(cell_size)))
        encoder = MultiScaleEncoder(1, enc_out_channels, n_halvings)
        thinker = SudokuThinkerCNN(enc_out_channels, thinker_out_channels, thinker_hidden, thinker_layers)
        bridge  = SudokuBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        feat = self._encode(condition)
        return self.bridge(feat), None


# ── Model 4 ────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV4(_CompressedBase):
    """
    Same as V3 but compression_factor ≠ cell_size → thinker grid ≠ 9×9.
    Default: compression_factor=16 → 288/16=18×18 thinker grid. No sudoku loss.
    """

    def __init__(
        self,
        painter_size: int = 288,
        compression_factor: int = 16,
        enc_out_channels: int = 16,
        thinker_out_channels: int = 64,
        bridge_channels: int = 8,
        thinker_hidden: int = 64,
        thinker_layers: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        n_halvings = int(round(math.log2(compression_factor)))
        encoder = MultiScaleEncoder(1, enc_out_channels, n_halvings)
        thinker = SudokuThinkerCNN(enc_out_channels, thinker_out_channels, thinker_hidden, thinker_layers)
        bridge  = SudokuBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)

    def thinker_forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        feat = self._encode(condition)
        return self.bridge(feat), None
