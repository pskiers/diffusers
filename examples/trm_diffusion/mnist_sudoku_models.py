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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

from models import SpatialEncoder, AttentiveBridge   # reuse from trm pipeline
from sudoku_models import SudokuTRM


# ── Spatial TRM ──────────────────────────────────────────────────────────────────

class SpatialTRM(SudokuTRM):
    """
    Spatial variant of SudokuTRM — identical internals, spatial I/O.

    Inherits unchanged from SudokuTRM:
      blocks (SwiGLU + post-norm RMSNorm), rotary_emb (RoPE), out_norm,
      lm_head (reused as out_proj), z_H_init / z_L_init (trunc_normal std=1),
      reasoning_step, get_initial_states, count_parameters.

    vocab_size=out_channels is passed to the parent so that lm_head has the
    right output dimension.  The token embedding (nn.Embedding) is replaced
    in-place by a Conv2d spatial projection — same attribute name, different
    type.

    embed() mirrors SudokuTRM's: scale + optional puzzle prefix token (RoPE
    handles positions, no separate 2D learned table needed).
    decode() reshapes the lm_head output back to (B, out_channels, H, W).
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
        num_puzzle_ids: int | None = None,
    ):
        seq_len = grid_h * grid_w
        # Parent builds the full TRM stack. vocab_size=out_channels so that
        # lm_head is Linear(d_model, out_channels) — the desired output shape.
        super().__init__(
            vocab_size=out_channels,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
        )
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Replace the token embedding with a Conv2d spatial projection.
        # Everything else (rotary_emb, blocks, lm_head, z_H/z_L init) is unchanged.
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=1)
        nn.init.kaiming_normal_(self.embedding.weight, nonlinearity="relu")
        nn.init.zeros_(self.embedding.bias)

    def embed(self, x: torch.Tensor, puzzle_ids=None) -> torch.Tensor:
        """
        (B, in_channels, grid_h, grid_w) → (B, [1+]seq_len, d_model)

        Mirrors SudokuTRM.embed(): embed_scale * projection + optional puzzle
        prefix token prepended to the sequence (same as SudokuTRM item 6).
        RoPE in _run_blocks handles positions — no separate 2D table needed.
        """
        tokens = self.embedding(x).flatten(2).transpose(1, 2)   # (B, seq, d_model)
        tokens = self.embed_scale * tokens

        if self.puzzle_id_embedding is not None:
            if puzzle_ids is None:
                puzzle_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            puzzle_token = self.puzzle_id_embedding(puzzle_ids).unsqueeze(1)
            tokens = torch.cat([puzzle_token, tokens], dim=1)

        return tokens

    # reasoning_step() is inherited from SudokuTRM unchanged.
    # It returns (lm_head_output, z_H_det, z_L_det) where lm_head_output is
    # (B, seq_len, out_channels) — exactly what decode() below expects.

    def decode(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        (B, seq_len, out_channels) → (B, out_channels, grid_h, grid_w)

        spatial_tokens is reasoning_step()[0]: lm_head has already been applied,
        so this is a pure reshape back to a spatial feature map.
        """
        B = spatial_tokens.shape[0]
        return spatial_tokens.transpose(1, 2).reshape(B, -1, self.grid_h, self.grid_w)


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
    """
    Build the denoising UNet.  Uses plain conv blocks throughout (no attention)
    to keep compute O(pixels) rather than O(pixels²).  For a 9×9 grid of
    independent digits, self-attention across the full image adds little value
    and dominates wall-clock time.
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
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        One n_sup supervision step.

        Encoder and thinker.embed are re-run on every call so each backward()
        gets a fresh computation graph (their parameters are nn.Parameters —
        saved tensors are freed after backward()).

        Returns: (noise_pred, sudoku_logits, z_H_next, z_L_next)
        """
        enc       = self.encoder(self._get_encoder_input(condition, noisy))
        input_emb = self.thinker.embed(enc, puzzle_ids=puzzle_ids)
        spatial_tokens, z_H_next, z_L_next = self.thinker.reasoning_step(input_emb, z_H, z_L)

        spatial_cond  = self.thinker.decode(spatial_tokens)
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
        puzzle_ids: Optional[torch.Tensor] = None,
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
            input_emb = self.thinker.embed(enc, puzzle_ids=puzzle_ids)
            T         = input_emb.shape[1]
            freqs_cis = self.thinker.rotary_emb(T)
            for _ in range(self.thinker.H_cycles):
                for _ in range(self.thinker.L_cycles):
                    z_L = self.thinker._run_blocks(input_emb + z_H + z_L, freqs_cis)
                z_H = self.thinker._run_blocks(z_H + z_L, freqs_cis)

        # Produce spatial tokens the same way reasoning_step() does.
        spatial_tokens = self.thinker.lm_head(
            self.thinker.out_norm(z_H[:, self.thinker.puzzle_emb_len:])
        )
        spatial_cond  = self.thinker.decode(spatial_tokens)
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
        num_puzzle_ids: int | None = None,
    ):
        grid_size = painter_size // cell_size   # e.g. 144//16 = 9

        encoder = SpatialEncoder(1, enc_channels, factor=cell_size)   # 1-ch: condition only
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
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
        num_puzzle_ids: int | None = None,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)   # 2-ch: cond + noisy
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=num_classes,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
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
        num_puzzle_ids: int | None = None,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
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
        num_puzzle_ids: int | None = None,
    ):
        grid_size = painter_size // cell_size

        encoder = SpatialEncoder(2, enc_channels, factor=cell_size)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
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
    Default: compression_factor=16 → 144//16=9×9 thinker grid.
    """

    _encoder_sees_noisy = True

    def __init__(
        self,
        painter_size: int = 144,
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
        num_puzzle_ids: int | None = None,
    ):
        grid_size = painter_size // compression_factor

        encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)
        thinker = SpatialTRM(
            in_channels=enc_channels, out_channels=thinker_out_channels,
            grid_h=grid_size, grid_w=grid_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, dropout=dropout,
            num_puzzle_ids=num_puzzle_ids,
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
