from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel

from models.control_painter_unet2d import ControlPainterUNet
from models.spade_painter_unet2d import SPADEUNet2D
from models.utility_models import SpatialBridge, ConditioningPyramid


def make_painter(
    painter_size: int,
    bridge_channels: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
) -> UNet2DModel:
    """
    Build the denoising UNet.  Uses plain conv blocks throughout (no attention).
    """
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return UNet2DModel(
        sample_size=painter_size,
        in_channels=1 + bridge_channels,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


def make_painter_control(
    painter_size: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
) -> ControlPainterUNet:
    """
    Build a ControlNet-capable painter UNet (in_channels=1, no bridge concat).
    """
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return ControlPainterUNet(
        sample_size=painter_size,
        in_channels=1,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


class StandalonePainter(nn.Module):
    """
    Pure diffusion painter with NO thinker.

    Condition: full sudoku solution tokens (B, 81) long, 2-10 range (0=PAD,1=blank,
    2-10=digits).  Converted to one-hot (B, vocab_size, 9, 9) → bridge → painter.

    Used as a sanity check: can the painter reliably render correct MNIST grids
    when given the perfect solution?  Also sets the upper-bound accuracy ceiling
    for painter-thinker models.

    Classifier-free guidance (CFG):
      Training: condition is randomly zeroed out with probability cfg_prob per sample.
      Inference: set cfg_scale > 1.0 on the model instance before sampling; the
                 model will run both conditioned and null passes and combine them.
    """

    token_input: bool = False  # uses solution tokens, handled via _get_condition
    has_realsolution_eval: bool = True  # realsolution IS the only conditioning

    def __init__(
        self,
        painter_size: int = 144,
        cell_size: int = 16,
        vocab_size: int = 11,
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 64),
        painter_layers_per_block: int = 2,
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__()
        self._grid = painter_size // cell_size
        self.vocab_size = vocab_size
        self.cfg_prob = cfg_prob
        self.cfg_scale = cfg_scale  # used at inference time; overridable externally
        self._painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype] if painter_dtype is not None else None
        )
        self.bridge = SpatialBridge(
            in_channels=vocab_size,
            out_channels=bridge_channels,
            painter_size=painter_size,
        )
        self.painter = make_painter(
            painter_size=painter_size,
            bridge_channels=bridge_channels,
            painter_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
        )

    @property
    def n_sup(self) -> int:
        return 1

    def get_painter_params(self) -> list:
        return list(self.parameters())

    def get_thinker_params(self) -> list:
        return []

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long in 2-10 → (B, vocab_size, grid, grid) one-hot float."""
        B = solution_tokens.shape[0]
        idx = solution_tokens.clamp(min=0, max=self.vocab_size - 1)
        onehot = F.one_hot(idx, num_classes=self.vocab_size).float()  # (B, 81, V)
        return onehot.transpose(1, 2).reshape(B, self.vocab_size, self._grid, self._grid)

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,  # (B, 81) long solution tokens 2-10
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        Training: randomly drops conditioning per sample at rate cfg_prob.
        Inference (self.training=False, cfg_scale>1): runs conditioned + null
        passes and combines them for classifier-free guidance.
        """
        spatial = self._solution_to_spatial(condition)

        if self.training and self.cfg_prob > 0:
            drop = torch.rand(spatial.shape[0], 1, 1, 1, device=spatial.device) < self.cfg_prob
            spatial = spatial * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.cfg_scale > 1.0:
            null = torch.zeros_like(spatial)
            s_both = torch.cat([spatial, null], dim=0)
            n_both = noisy.repeat(2, 1, 1, 1)
            t_both = timesteps.repeat(2)
            with ctx:
                bf = self.bridge(s_both)
                pred = self.painter(torch.cat([n_both, bf], dim=1), t_both).sample
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            bridge_feat = self.bridge(spatial)
            noise_pred = self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample
        return noise_pred, None


class StandalonePainterSPADE(StandalonePainter):
    """
    Standalone painter using SPADE conditioning instead of a bridge+concat.

    Solution tokens → one-hot (B, vocab_size, 9, 9) → bilinearly upsampled to
    painter_size → fed as semantic map `s` to SPADEUNet2D (in_channels=1, no concat).
    """

    def __init__(
        self,
        painter_size: int = 144,
        cell_size: int = 16,
        vocab_size: int = 11,
        painter_channels: tuple = (32, 64, 64),
        painter_layers_per_block: int = 2,
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        # Use the smallest valid bridge_channels so parent __init__ doesn't crash,
        # then replace bridge+painter immediately after.
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            vocab_size=vocab_size,
            bridge_channels=1,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )
        self.painter_size = painter_size
        # Discard the bridge (unused in SPADE) and replace UNet.
        del self.bridge
        self.bridge = None
        self.painter = SPADEUNet2D(
            painter_size=painter_size,
            sem_channels=vocab_size,
            block_out_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
        )

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long 2-10 → (B, vocab_size, painter_size, painter_size) float upsampled."""
        spatial = super()._solution_to_spatial(solution_tokens)  # (B, V, grid, grid)
        return F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        s = self._solution_to_spatial(condition)

        if self.training and self.cfg_prob > 0:
            drop = torch.rand(s.shape[0], 1, 1, 1, device=s.device) < self.cfg_prob
            s = s * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.cfg_scale > 1.0:
            null = torch.zeros_like(s)
            s_both = torch.cat([s, null], dim=0)
            n_both = noisy.repeat(2, 1, 1, 1)
            t_both = timesteps.repeat(2)
            with ctx:
                pred = self.painter(n_both, t_both, s_both)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            noise_pred = self.painter(noisy, timesteps, s)
        return noise_pred, None


class StandalonePainterControl(StandalonePainter):
    """
    Standalone painter using ControlNet-style residual injection instead of bridge+concat.

    Solution tokens → one-hot (B, vocab_size, 9, 9) → bilinearly upsampled to
    painter_size → ConditioningPyramid → per-layer residuals injected into
    _ControlPainterUNet (in_channels=1, no bridge concatenation).
    """

    def __init__(
        self,
        painter_size: int = 144,
        cell_size: int = 16,
        vocab_size: int = 11,
        painter_channels: tuple = (32, 64, 64),
        painter_layers_per_block: int = 2,
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            vocab_size=vocab_size,
            bridge_channels=1,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )
        self.painter_size = painter_size
        del self.bridge
        self.bridge = None
        self.painter = make_painter_control(
            painter_size=painter_size,
            painter_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
        )
        self.control_pyramid = ConditioningPyramid(
            in_channels=vocab_size,
            block_out_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
        )

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long 2-10 → (B, vocab_size, painter_size, painter_size) float upsampled."""
        spatial = super()._solution_to_spatial(solution_tokens)
        return F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)

    def _run_painter_ctrl(self, noisy, s, timesteps):
        down_res, mid_res = self.control_pyramid(s)
        return self.painter(
            noisy,
            timesteps,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        s = self._solution_to_spatial(condition)

        if self.training and self.cfg_prob > 0:
            drop = torch.rand(s.shape[0], 1, 1, 1, device=s.device) < self.cfg_prob
            s = s * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.cfg_scale > 1.0:
            B = noisy.shape[0]
            null = torch.zeros_like(s)
            with ctx:
                pred_cond = self._run_painter_ctrl(noisy, s, timesteps)
                pred_uncond = self._run_painter_ctrl(noisy, null, timesteps)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            noise_pred = self._run_painter_ctrl(noisy, s, timesteps)
        return noise_pred, None
