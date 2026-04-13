"""
Ratatouille TRM-DiT model variants for MNIST Sudoku.

Architecture overview:
  Painter  – UNet2DModel in pixel space (1-channel grayscale).
             Input: cat([noisy_image, bridge_features], dim=1).
             Conditioning reaches the painter ONLY via spatial bridge features.

  Thinker  – SpatialTRM: small transformer TRM that reasons about the puzzle.
             Operates on a compressed spatial grid (seq_len = grid_h × grid_w tokens).
             States: z_H (slow), z_L (fast) as (B, seq_len, d_model).
             Output: spatial feature map → bridge → painter conditioning.

  Bridge   – upsamples thinker output to painter resolution.
             V0–V3: bilinear upsample + 2-conv (SpatialBridge).
             V4:    cross-attention (AttentiveBridge from models.py).

  Encoder  – SpatialEncoder from models.py compresses input image(s) to thinker grid.

Training interface (n_sup backward passes per batch, identical to SudokuTRM):
    z_H, z_L = model.get_initial_states(B)
    for _ in range(model.n_sup):
        noise_pred, logits, z_H, z_L = model.reasoning_step(cond, noisy, z_H, z_L, ts)
        loss = mse(noise_pred, noise) + sudoku_w * ce(logits, solution)
        backward(loss); optimizer.step(); zero_grad(); global_step += 1

Inference interface:
    noise_pred, logits = model(noisy_images, timesteps, condition)
    Runs full n_sup × H_cycles × L_cycles loops (no training-specific grad split).

Five variants — each removes one training wheel from the previous:

  V0: Encoder sees CONDITION ONLY; 9×9 TRM; TRM output = digit logits.
      Losses: diffusion + weighted sudoku CE.
      Most constrained: clean input, explicit digit supervision, 9-channel output,
      9×9 grid. Closest to vanilla SudokuTRM + DiT painter.

  V1: Encoder sees cat(condition, noisy); 9×9 TRM; digit logits.
      Training wheel removed: TRM must reason from noisy input too.
      Losses: diffusion + weighted sudoku CE.

  V2: Same as V1 but no sudoku CE loss.
      Training wheel removed: no explicit digit-level supervision.

  V3: Same as V2 but thinker_out_channels >> 9.
      Training wheel removed: output not forced into digit-logit space.

  V4: Same as V3 but compression_factor ≠ cell_size; AttentiveBridge decoder.
      Training wheel removed: grid uncoupled from puzzle cell structure.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

from models import SpatialEncoder, AttentiveBridge   # reuse from trm pipeline
from sudoku_models import TransformerBlock, RMSNorm


# ── Spatial TRM ──────────────────────────────────────────────────────────────────

class SpatialTRM(nn.Module):
    """
    Tiny Recursive Model operating on a compressed 2-D spatial grid.

    Pipeline:
      embed:  (B, in_c, grid_h, grid_w) → tokens (B, H*W, d_model)
      TRM:    shared TransformerBlocks iterate z_L (fast) then z_H (slow)
      decode: tokens → (B, out_c, grid_h, grid_w)

    Recursion is identical to SudokuTRM (L_cycles inner, H_cycles outer,
    n_sup supervision steps with one backward each).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_h: int,
        grid_w: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.L_cycles    = L_cycles
        self.H_cycles    = H_cycles
        self.n_sup       = n_sup
        self.grid_h      = grid_h
        self.grid_w      = grid_w
        seq_len          = grid_h * grid_w
        self.seq_len     = seq_len

        embed_init_std    = 1.0 / math.sqrt(d_model)
        self.embed_scale  = math.sqrt(d_model)

        # Project encoded spatial features to d_model token space.
        # input_proj contains learnable weights — embed() must be re-called on
        # every n_sup step during training (graph freed after each backward()).
        self.input_proj    = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * embed_init_std
        )

        # Shared blocks — same weights for z_L and z_H updates (mirrors SudokuTRM).
        # TransformerBlock is now post-norm (RMSNorm after residual) so no
        # separate norm_z_H / norm_z_L are needed.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_channels)

        # Learnable initial states (trunc_normal std=1, same as SudokuTRM).
        self.z_H_init = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.z_L_init = nn.Parameter(torch.empty(1, seq_len, d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=1.0 / self.embed_scale)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        nn.init.trunc_normal_(self.z_H_init, std=1.0)
        nn.init.trunc_normal_(self.z_L_init, std=1.0)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, grid_h, grid_w) → (B, seq_len, d_model)

        Scaling convention from SudokuTRM: embed_scale * 0.707 so that
        per-element std ≈ 1.0 after summing spatial + positional embeddings.
        """
        tokens = self.input_proj(x).flatten(2).transpose(1, 2)   # (B, seq, d_model)
        return self.embed_scale * 0.707106781 * (tokens + self.pos_embedding)

    def get_initial_states(self, bsz: int):
        z_H = self.z_H_init.expand(bsz, -1, -1).clone()
        z_L = self.z_L_init.expand(bsz, -1, -1).clone()
        return z_H, z_L

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def reasoning_step(
        self,
        input_emb: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
    ):
        """
        One n_sup supervision step.

        (H_cycles-1) macro cycles without gradients, then one final cycle
        with gradients — identical structure to SudokuTRM.reasoning_step.

        Returns:
            z_H_grad     – z_H with gradient (for decode → bridge → painter)
            z_H_detached – state for next n_sup step
            z_L_detached – state for next n_sup step
        """
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self._run_blocks(input_emb + z_H + z_L)
                z_H = self._run_blocks(z_H + z_L)

        # Final macro cycle — gradients flow through this into bridge and painter
        for _ in range(self.L_cycles):
            z_L = self._run_blocks(input_emb + z_H + z_L)
        z_H = self._run_blocks(z_H + z_L)

        return z_H, z_H.detach(), z_L.detach()

    def decode(self, z_H: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, d_model) → (B, out_channels, grid_h, grid_w)"""
        B   = z_H.shape[0]
        out = self.out_proj(self.out_norm(z_H))   # (B, seq, out_channels)
        return out.transpose(1, 2).reshape(B, -1, self.grid_h, self.grid_w)


# ── Bilinear bridge (V0–V3) ──────────────────────────────────────────────────────

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


# ── Painter factory ──────────────────────────────────────────────────────────────

def _make_painter(
    painter_size: int,
    bridge_channels: int,
    painter_channels: tuple[int, ...],
) -> UNet2DModel:
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return UNet2DModel(
        sample_size=painter_size,
        in_channels=1 + bridge_channels,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=tuple(
            "DownBlock2D" if i == 0 else "AttnDownBlock2D" for i in range(n)
        ),
        up_block_types=tuple(
            "UpBlock2D" if i == n - 1 else "AttnUpBlock2D" for i in range(n)
        ),
        norm_num_groups=norm_num_groups,
    )


# ── TRM Ratatouille base ─────────────────────────────────────────────────────────

class _TRMRatatouilleBase(nn.Module):
    """
    Base class for TRM-based Ratatouille models.

    Sub-classes set:
      _encoder_sees_noisy (bool) – encoder gets condition only (V0) or
                                   cat(condition, noisy) (V1–V4).

    Sub-classes override:
      _compute_sudoku_logits(spatial_cond) → tensor | None

    Training: call reasoning_step in a loop (n_sup times per batch).
    Inference: call forward() — runs the full n_sup × H_cycles × L_cycles loop.
    """

    _encoder_sees_noisy: bool = True

    def __init__(
        self,
        encoder: SpatialEncoder,
        thinker: SpatialTRM,
        bridge: nn.Module,
        painter: UNet2DModel,
    ):
        super().__init__()
        self.encoder = encoder
        self.thinker = thinker
        self.bridge  = bridge
        self.painter = painter

    @property
    def n_sup(self) -> int:
        return self.thinker.n_sup

    def get_initial_states(self, bsz: int):
        return self.thinker.get_initial_states(bsz)

    def _get_encoder_input(
        self, condition: torch.Tensor, noisy: torch.Tensor
    ) -> torch.Tensor:
        if self._encoder_sees_noisy:
            return torch.cat([condition, noisy], dim=1)
        return condition

    def _compute_sudoku_logits(
        self, spatial_cond: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def reasoning_step(
        self,
        condition: torch.Tensor,   # (B, 1, H, W)
        noisy: torch.Tensor,       # (B, 1, H, W)
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """
        One n_sup supervision step.

        Encoder and thinker.embed are re-run on every call so each backward()
        gets a fresh computation graph (their parameters are nn.Parameters —
        saved tensors are freed after backward()).

        Returns: (noise_pred, sudoku_logits, z_H_next, z_L_next)
        """
        enc       = self.encoder(self._get_encoder_input(condition, noisy))
        input_emb = self.thinker.embed(enc)
        z_H_grad, z_H_next, z_L_next = self.thinker.reasoning_step(input_emb, z_H, z_L)

        spatial_cond  = self.thinker.decode(z_H_grad)
        bridge_feat   = self.bridge(spatial_cond)
        noise_pred    = self.painter(
            torch.cat([noisy, bridge_feat], dim=1), timesteps
        ).sample
        sudoku_logits = self._compute_sudoku_logits(spatial_cond)
        return noise_pred, sudoku_logits, z_H_next, z_L_next

    def forward(
        self,
        noisy_images: torch.Tensor,   # (B, 1, H, W)
        timesteps: torch.Tensor,      # (B,)
        condition: torch.Tensor,      # (B, 1, H, W)
    ):
        """
        Full inference: n_sup × H_cycles × L_cycles, single painter pass at end.
        No training-specific grad split — all cycles run uniformly (like predict()
        in SudokuTRM).  Encoder run once (condition+noisy fixed during inference).

        Returns: (noise_pred, sudoku_logits)
        """
        B    = noisy_images.shape[0]
        z_H, z_L = self.get_initial_states(B)
        z_H  = z_H.to(noisy_images.device)
        z_L  = z_L.to(noisy_images.device)

        enc  = self.encoder(self._get_encoder_input(condition, noisy_images))

        for _ in range(self.thinker.n_sup):
            input_emb = self.thinker.embed(enc)
            for _ in range(self.thinker.H_cycles):
                for _ in range(self.thinker.L_cycles):
                    z_L = self.thinker._run_blocks(input_emb + z_H + z_L)
                z_H = self.thinker._run_blocks(z_H + z_L)

        spatial_cond  = self.thinker.decode(z_H)
        bridge_feat   = self.bridge(spatial_cond)
        noise_pred    = self.painter(
            torch.cat([noisy_images, bridge_feat], dim=1), timesteps
        ).sample
        sudoku_logits = self._compute_sudoku_logits(spatial_cond)
        return noise_pred, sudoku_logits


# ── V0 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV0(_TRMRatatouilleBase):
    """
    Closest to vanilla SudokuTRM: encoder sees CONDITION ONLY, 9×9 grid,
    TRM output channels = num_classes → direct digit logits per cell.

    The TRM that explicitly solves Sudoku also provides spatial conditioning
    for the DiT painter that renders the MNIST digits — joint training.

    Losses: diffusion MSE + weighted sudoku CE.
    Training wheels present (removed in V1→V4):
      clean input | explicit CE supervision | 9-channel output | 9×9 grid
    """

    _encoder_sees_noisy = False

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        grid_size = painter_size // cell_size   # e.g. 288//32 = 9

        encoder = SpatialEncoder(1, enc_channels, factor=cell_size)   # 1-ch: condition only
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
        )
        bridge  = SpatialBridge(num_classes, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V1 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV1(_TRMRatatouilleBase):
    """
    Same as V0 but encoder sees cat(condition, noisy_image).
    Training wheel removed: TRM must reason from a noisy/corrupted signal.
    Grid: 9×9.  Losses: diffusion + weighted sudoku CE.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        num_classes: int = 9,
        enc_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)   # 2-ch: cond + noisy
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
        )
        bridge  = SpatialBridge(num_classes, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)
        self._num_classes = num_classes

    def _compute_sudoku_logits(self, spatial_cond: torch.Tensor) -> torch.Tensor:
        B = spatial_cond.shape[0]
        return spatial_cond.permute(0, 2, 3, 1).reshape(B, 81, self._num_classes)


# ── V2 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV2(_TRMRatatouilleBase):
    """
    Same as V1 but no sudoku CE loss.
    Training wheel removed: TRM gets no explicit digit-level supervision.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
        )
        bridge  = SpatialBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)


# ── V3 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV3(_TRMRatatouilleBase):
    """
    Same as V2 but thinker_out_channels >> 9 (not constrained to digit-logit space).
    Training wheel removed: output dimensionality is unconstrained.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        cell_size: int = 32,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
        )
        bridge  = SpatialBridge(thinker_out_channels, bridge_channels, painter_size)
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)


# ── V4 ───────────────────────────────────────────────────────────────────────────

class MNISTRatatouilleV4(_TRMRatatouilleBase):
    """
    Same as V3 but compression_factor ≠ cell_size → thinker grid ≠ 9×9,
    and uses AttentiveBridge (Perceiver-IO cross-attention) for upsampling.
    Training wheel removed: grid topology uncoupled from puzzle cell structure.
    Default: compression_factor=16 → 288//16=18×18 thinker grid.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 288,
        compression_factor: int = 16,
        enc_channels: int = 32,
        thinker_out_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 4,
        bridge_channels: int = 8,
        bridge_num_heads: int = 4,
        painter_channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        grid_size = painter_size // compression_factor

        encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
        )
        # AttentiveBridge: Perceiver-IO cross-attention upsamples low-res thinker
        # output to painter_size × painter_size via learned positional queries.
        bridge  = AttentiveBridge(
            in_channels=thinker_out_channels,
            out_channels=bridge_channels,
            out_resolution=painter_size,
            factor=compression_factor,
            num_heads=bridge_num_heads,
        )
        painter = _make_painter(painter_size, bridge_channels, painter_channels)

        super().__init__(encoder=encoder, thinker=thinker, bridge=bridge, painter=painter)
