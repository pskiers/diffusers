"""
trm_wrappers.py — Thin wrappers around TinyRecursiveReasoningModel_ACTV1_Inner.

Two public classes:

  OriginalTRMSudoku
      Drop-in thinker for standalone Sudoku training.
      Interface mirrors SudokuTRM:
          get_initial_states(bsz) → (z_H, z_L)
          reasoning_step(inputs, z_H, z_L, puzzle_ids=None) → (logits, z_H, z_L)
          predict(inputs, ...) → logits
          n_sup, vocab_size attributes

  OriginalTRMRatatouilleV0Tok
      Painter-thinker model for MNIST Sudoku (token-input V0 variant).
      The thinker is OriginalTRMSudoku (vocab_size=11, token prediction).
      Spatial conditioning: thinker logits (B,81,11) reshaped to (B,11,9,9),
      upsampled via SpatialBridge, concatenated with noisy image for the painter.
      Interface:
          get_initial_states(bsz) → (z_H, z_L)
          reasoning_step(puzzle_tokens, noisy, z_H, z_L, timesteps, ...) → (noise_pred, logits, z_H, z_L)
          forward(noisy, timesteps, puzzle_tokens, ...) → (noise_pred, logits)
          token_input = True   (tells training loop to use puzzle_tokens)
          n_sup attribute

Design notes:
  * No ACT halting by default (halt_exploration_prob=0, fixed n_sup steps).
    Set halt_exploration_prob > 0 to enable the original ACT exploration.
  * puzzle_emb_ndim > 0 enables the original sparse puzzle embeddings.
    IMPORTANT: when enabled, puzzle embedding weights (inner.puzzle_emb) must be
    trained with a separate CastedSparseEmbeddingSignSGD_Distributed optimizer,
    NOT AdamW.  Use build_puzzle_emb_optimizer() from this module.
    With puzzle_emb_ndim=0 (default), AdamW covers all parameters.
  * H_init / L_init are non-trainable buffers (matching the original), broadcast
    to (bsz, seq_len, hidden_size) as the initial carry.
  * The inner model handles its own bf16 casting via CastedEmbedding/CastedLinear.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from mnist_sudoku_models import SpatialBridge, _make_painter
from models_pt import SpatialEncoder, AttentiveBridge, TimestepMLP


# ── Sparse puzzle-embedding optimizer factory ──────────────────────────────────

def build_puzzle_emb_optimizer(
    model: "OriginalTRMSudoku",
    world_size: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> Optional[CastedSparseEmbeddingSignSGD_Distributed]:
    """
    Return a CastedSparseEmbeddingSignSGD_Distributed optimizer for the inner
    model's sparse puzzle embedding, or None when puzzle_emb_ndim=0.

    The returned optimizer must be stepped separately from (and after) AdamW.
    Pass the puzzle_emb parameters to AdamW with lr=0 so they are excluded from
    its update: use `get_non_puzzle_emb_params(model)` to build the AdamW group.

    Example usage in training script:
        puzzle_opt = build_puzzle_emb_optimizer(model, world_size=accelerator.num_processes)
        adamw = torch.optim.AdamW(get_non_puzzle_emb_params(model), lr=lr, ...)
        # in training loop:
        adamw.step(); adamw.zero_grad()
        if puzzle_opt: puzzle_opt.step()
    """
    inner = model.inner if isinstance(model, OriginalTRMSudoku) else model.thinker.inner
    if not hasattr(inner, "puzzle_emb"):
        return None
    emb = inner.puzzle_emb
    return CastedSparseEmbeddingSignSGD_Distributed(
        emb.buffers(),
        world_size=world_size,
        lr=lr,
        weight_decay=weight_decay,
    )


def get_non_puzzle_emb_params(model: nn.Module) -> list:
    """
    Return all parameters except the sparse puzzle embedding's local_weights /
    local_ids / weights buffers.  Use these as the AdamW parameter group so that
    AdamW does not touch the sparse embedding.
    """
    exclude_ids: set[int] = set()
    inner = model.inner if hasattr(model, "inner") else getattr(model, "thinker", model).inner
    if hasattr(inner, "puzzle_emb"):
        emb = inner.puzzle_emb
        for buf in (emb.local_weights, emb.local_ids, emb.weights):
            exclude_ids.add(id(buf))
    return [p for p in model.parameters() if id(p) not in exclude_ids]


# ── Spatial-input TRM inner model ─────────────────────────────────────────────

class _SpatialInputTRMInner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """Drop-in replacement for TinyRecursiveReasoningModel_ACTV1_Inner that also
    accepts pre-computed float embeddings as inputs.

    When batch["inputs"] is a *floating-point* tensor (B, seq_len, hidden_size),
    the embed_tokens call is skipped and the tensor is used directly as the input
    injection signal (scaled by embed_scale, plus learned pos enc if configured).

    When batch["inputs"] is an integer tensor the behaviour is identical to the
    parent class — fully backward-compatible.

    This enables image-conditioned variants (V0–V4) where a CNN encoder computes
    the input embeddings rather than an nn.Embedding lookup.
    """

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if input.is_floating_point():
            # Pre-computed embedding: (B, seq_len, hidden_size)
            embedding = input.to(self.forward_dtype)
            if self.config.pos_encodings == "learned":
                embedding = 0.707106781 * (
                    embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
                )
            return self.embed_scale * embedding
        return super()._input_embeddings(input, puzzle_identifiers)


# ── Core thinker wrapper ───────────────────────────────────────────────────────

class OriginalTRMSudoku(nn.Module):
    """
    Wraps TinyRecursiveReasoningModel_ACTV1_Inner for Sudoku training.

    Parameters
    ----------
    vocab_size          Token vocabulary size (0=PAD, 1=blank, 2-10=digits).
    seq_len             Sequence length (81 for 9×9 Sudoku).
    hidden_size         Transformer hidden dimension.
    n_heads             Number of attention heads.
    L_layers            Transformer layers per block call (L_layers in original).
    L_cycles            Inner iterations per H-cycle.
    H_cycles            Outer macroscopic cycles.
    n_sup               Supervision steps per batch (= halt_max_steps).
    expansion           SwiGLU hidden-dim multiplier (original uses 4).
    forward_dtype       Internal compute dtype ("bfloat16" | "float32").
    mlp_t               Use MLP-T block instead of attention (original default False).
    puzzle_emb_ndim     Dimensionality of sparse puzzle embeddings (0 = disabled).
                        When > 0, must also set num_puzzle_identifiers and
                        use build_puzzle_emb_optimizer() for a SignSGD optimizer.
    puzzle_emb_len      Number of prefix tokens for puzzle embeddings.
                        Ignored when puzzle_emb_ndim=0.
    num_puzzle_identifiers  Number of distinct puzzle IDs for sparse embeddings.
    halt_exploration_prob   ACT exploration probability (0 = fixed n_sup steps,
                            0.1 = original setting with adaptive halting).
    batch_size          Used only by CastedSparseEmbedding when puzzle_emb_ndim>0.
    """

    def __init__(
        self,
        vocab_size: int = 11,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 16,
        num_puzzle_identifiers: int = 1,
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
    ):
        super().__init__()
        self.n_sup = n_sup
        self.vocab_size = vocab_size
        self.freeze_weights = freeze_weights

        # puzzle_emb_len is only meaningful when puzzle_emb_ndim > 0
        effective_puzzle_emb_len = puzzle_emb_len if puzzle_emb_ndim > 0 else 0

        config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=batch_size,
            seq_len=seq_len,
            puzzle_emb_ndim=puzzle_emb_ndim,
            num_puzzle_identifiers=num_puzzle_identifiers,
            vocab_size=vocab_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            H_layers=0,                         # ignored by inner model
            L_layers=L_layers,
            hidden_size=hidden_size,
            expansion=expansion,
            num_heads=n_heads,
            pos_encodings=pos_encodings,
            halt_max_steps=n_sup,
            halt_exploration_prob=halt_exploration_prob,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            puzzle_emb_len=effective_puzzle_emb_len,
            no_ACT_continue=True,
        )
        self.inner = _SpatialInputTRMInner(config)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_initial_states(self, bsz: int):
        """
        Return (z_H, z_L) on the same device as the model, initialized from
        H_init / L_init buffers (1-D vectors broadcast to all sequence positions).
        """
        total_len = self.inner.config.seq_len + self.inner.puzzle_emb_len
        # H_init / L_init: (hidden_size,) → expand to (bsz, total_len, hidden_size)
        z_H = self.inner.H_init.view(1, 1, -1).expand(bsz, total_len, -1).clone()
        z_L = self.inner.L_init.view(1, 1, -1).expand(bsz, total_len, -1).clone()
        return z_H, z_L

    def reasoning_step(
        self,
        inputs: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        """
        One supervision step. Internally runs H_cycles-1 no-grad cycles then
        one full-grad cycle (matching the original training pattern).

        H_cycles / L_cycles: override the config values for this call only.

        Returns: (logits, z_H_detached, z_L_detached)
          logits: (B, seq_len, vocab_size) — gradients attached
        """
        bsz = inputs.shape[0]
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(bsz, dtype=torch.int32, device=inputs.device)

        orig_H = self.inner.config.H_cycles
        orig_L = self.inner.config.L_cycles
        if H_cycles is not None:
            self.inner.config.H_cycles = H_cycles
        if L_cycles is not None:
            self.inner.config.L_cycles = L_cycles
        try:
            carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
            new_carry, logits, _ = self.inner(carry, {"inputs": inputs, "puzzle_identifiers": puzzle_ids})
        finally:
            self.inner.config.H_cycles = orig_H
            self.inner.config.L_cycles = orig_L

        return logits, new_carry.z_H, new_carry.z_L

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        n_sup: Optional[int] = None,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full inference: n_sup × H_cycles × L_cycles uniformly (no grad split).
        Matches the original eval behaviour (always runs max steps).
        """
        n_sup = n_sup or self.n_sup
        bsz = inputs.shape[0]
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(bsz, dtype=torch.int32, device=inputs.device)

        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(inputs.device)
        z_L = z_L.to(inputs.device)

        seq_info = {"cos_sin": self.inner.rotary_emb() if hasattr(self.inner, "rotary_emb") else None}
        input_emb = self.inner._input_embeddings(inputs, puzzle_ids)

        for _ in range(n_sup):
            for _ in range(self.inner.config.H_cycles):
                for _ in range(self.inner.config.L_cycles):
                    z_L = self.inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = self.inner.L_level(z_H, z_L, **seq_info)

        return self.inner.lm_head(z_H)[:, self.inner.puzzle_emb_len:]


# ── Painter-thinker (V0Tok) ────────────────────────────────────────────────────

class OriginalTRMRatatouilleV0Tok(nn.Module):
    """
    Painter-thinker model using the original TRM as the thinker (token input).

    Thinker: OriginalTRMSudoku(vocab_size) — receives puzzle tokens directly,
             outputs (B, 81, vocab_size) logits over sudoku token IDs.
    Bridge:  SpatialBridge — bilinear upsample + 2 convs:
             (B, vocab_size, 9, 9) → (B, bridge_channels, painter_size, painter_size).
    Painter: UNet2DModel — denoises cat([noisy, bridge_feat], dim=1).

    The thinker logits are reshaped to (B, vocab_size, 9, 9) before the bridge:
      logits (B,81,V) → transpose → (B,V,81) → reshape → (B,V,9,9)

    token_input=True tells train_trm.py to use batch["puzzle_tokens"] as
    the condition, not batch["conditions"].

    Sudoku CE loss: labels must be in token format (2-10 for correct digit, -100
    for ignored). train_trm.py converts solution (0-8 digit classes) → (2-10)
    automatically.  See train_trm.py for details.

    diff_thinker_weight: scale of diffusion-loss gradient back into the thinker.
      0.0  → thinker trained only on sudoku CE loss (bridge sees detached spatial).
      1.0  → full diffusion gradient reaches thinker.
    """

    token_input: bool = True
    has_realsolution_eval: bool = True   # eval with full solution tokens as condition

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        vocab_size: int = 11,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 16,
        num_puzzle_identifiers: int = 1,
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        # How thinker logits are converted to spatial conditioning at inference time.
        #   "logits"  – raw logits (default, matches training with raw logit spatial)
        #   "onehot"  – argmax → one-hot (matches painter trained on real one-hot solutions)
        #   "softmax" – softmax probabilities (soft version of onehot)
        thinker_bridge_mode: str = "logits",
        # Autocast dtype for the bridge + painter UNet.  None = no autocast
        # (painter runs in whatever dtype the tensors arrive in).
        # "bfloat16" is the safe default: same exponent range as float32, so no
        # GradScaler needed.  "float16" also works but requires a GradScaler
        # (use accelerate mixed_precision="fp16" which handles this automatically).
        # The TRM thinker is NOT affected — it manages its own dtype via forward_dtype.
        painter_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.diff_thinker_weight = diff_thinker_weight
        self.thinker_bridge_mode = thinker_bridge_mode
        self._grid = painter_size // cell_size   # e.g. 144//16 = 9
        # Thinker vocab uses 0=PAD, 1=blank, 2-10=digits 1-9.
        # sample_grids compares argmax predictions against raw solution labels (0-8),
        # so it needs to know to shift by this offset when comparing.
        self.token_offset = 2
        self._painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype]
            if painter_dtype is not None else None
        )

        self.thinker = OriginalTRMSudoku(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            puzzle_emb_ndim=puzzle_emb_ndim,
            puzzle_emb_len=puzzle_emb_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
        )
        self.bridge = SpatialBridge(
            in_channels=vocab_size,
            out_channels=bridge_channels,
            painter_size=painter_size,
        )
        self.painter = _make_painter(
            painter_size=painter_size,
            bridge_channels=bridge_channels,
            painter_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
        )

    @property
    def n_sup(self) -> int:
        return self.thinker.n_sup

    def get_initial_states(self, bsz: int):
        return self.thinker.get_initial_states(bsz)

    def get_painter_params(self) -> list:
        """Parameters belonging to the bridge and painter UNet (for a separate optimizer)."""
        return list(self.bridge.parameters()) + list(self.painter.parameters())

    def get_thinker_params(self) -> list:
        """Parameters belonging to the thinker (excluding painter/bridge)."""
        painter_ids = {id(p) for p in self.get_painter_params()}
        return [p for p in self.parameters() if id(p) not in painter_ids]

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) logits → (B, C, grid, grid) spatial conditioning.

        Conversion respects self.thinker_bridge_mode:
          "logits"  – raw float logits (default)
          "onehot"  – argmax → one-hot
          "softmax" – softmax probabilities
        """
        B, _, C = logits.shape
        mode = getattr(self, "thinker_bridge_mode", "logits")
        if mode == "onehot":
            preds  = logits.argmax(dim=-1)
            onehot = F.one_hot(preds, num_classes=C).float()
            return onehot.transpose(1, 2).reshape(B, C, self._grid, self._grid)
        elif mode == "softmax":
            probs = logits.float().softmax(dim=-1)
            return probs.transpose(1, 2).reshape(B, C, self._grid, self._grid)
        else:
            return logits.float().transpose(1, 2).reshape(B, C, self._grid, self._grid)

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Autocast applies to bridge + UNet only; TRM handles its own dtype.
        # Loss is always computed in float32 by callers (.float() before MSE/CE).
        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )
        with ctx:
            bridge_feat = self.bridge(spatial_cond)
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample

    def reasoning_step(
        self,
        puzzle_tokens: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        """
        One supervision step: thinker → bridge → painter.

        diff_thinker_weight scales the gradient that diffusion loss sends back
        through the bridge into the thinker (1.0 = full, 0.0 = detached).
        The sudoku CE loss always flows through unscaled logits.

        H_cycles / L_cycles: override thinker config for this call only.

        Returns: (noise_pred, sudoku_logits, z_H_detached, z_L_detached)
        """
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            puzzle_tokens, z_H, z_L, puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles
        )
        # spatial_cond in float for bridge (inner model runs in bf16)
        spatial_cond = self._logits_to_spatial(logits.float())

        if self.diff_thinker_weight == 0.0:
            sc_for_painter = spatial_cond.detach()
        elif self.diff_thinker_weight != 1.0:
            # Partial gradient: interpolate between detached and full
            sc_for_painter = (
                self.diff_thinker_weight * spatial_cond
                + (1.0 - self.diff_thinker_weight) * spatial_cond.detach()
            )
        else:
            sc_for_painter = spatial_cond

        noise_pred = self._run_painter(noisy, sc_for_painter, timesteps)
        return noise_pred, logits, z_H_next, z_L_next

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        Full inference: run all n_sup thinker steps, then one painter pass.
        Used for eval (no gradient).
        """
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(noisy.device)
        z_L = z_L.to(noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(
                puzzle_tokens, z_H, z_L, puzzle_ids
            )

        spatial_cond = self._logits_to_spatial(logits.float())
        noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits


# ── Painter-thinker (V0: image-conditioned) ───────────────────────────────────

class OriginalTRMRatatouilleV0(OriginalTRMRatatouilleV0Tok):
    """
    Image-conditioned painter-thinker (V0).

    Identical to V0Tok except the thinker receives CNN-encoded puzzle image
    features instead of discrete puzzle tokens.  A SpatialEncoder + 1×1 Conv2d
    projects the condition image (B, 1, H, W) to float embeddings
    (B, 81, hidden_size) which are fed directly to _SpatialInputTRMInner
    (bypassing embed_tokens).

    token_input = False  → train_trm uses batch["conditions"] not puzzle_tokens
    token_offset = 0     → logits are already in 0-8 digit space
    """

    token_input: bool = False
    has_realsolution_eval: bool = True   # eval with full MNIST image as condition

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        num_classes: int = 9,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        # --- bridge & painter ---
        thinker_out_channels: int = None,   # if > num_classes, expands logits before bridge
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        _toc = thinker_out_channels if thinker_out_channels is not None else num_classes
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            vocab_size=num_classes,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            puzzle_emb_ndim=0,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            painter_dtype=painter_dtype,
        )
        self.token_offset = 0

        # condition image (B,1,H,W) → (B, enc_channels, grid, grid)
        self.image_encoder = SpatialEncoder(1, enc_channels, factor=cell_size)
        # project enc_channels → hidden_size per cell
        std = 1.0 / (math.sqrt(hidden_size) * math.sqrt(enc_channels))
        self.enc_proj = nn.Conv2d(enc_channels, hidden_size, 1)
        nn.init.normal_(self.enc_proj.weight, std=std)
        nn.init.zeros_(self.enc_proj.bias)

        # Optional expansion: project num_classes → thinker_out_channels before bridge.
        # CE loss still uses raw num_classes logits; only the bridge sees the expanded map.
        if _toc != num_classes:
            self.logit_expand = nn.Linear(num_classes, _toc, bias=False)
            self.bridge = SpatialBridge(
                in_channels=_toc,
                out_channels=bridge_channels,
                painter_size=painter_size,
            )
        else:
            self.logit_expand = None

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        return super()._logits_to_spatial(logits)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → float embeddings (B, 81, hidden_size)"""
        feat = self.image_encoder(x)               # (B, enc_channels, grid, grid)
        proj = self.enc_proj(feat)                 # (B, hidden_size, grid, grid)
        return proj.flatten(2).transpose(1, 2)     # (B, 81, hidden_size)

    def _get_enc_emb(
        self, condition: torch.Tensor, noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """V0: encode condition only, ignore noisy and timesteps.
        V1 overrides to use cat(condition, noisy) and optionally the timestep."""
        return self._encode_image(condition)

    def reasoning_step(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)
        return super().reasoning_step(
            enc_emb, noisy, z_H, z_L, timesteps,
            puzzle_ids=puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(noisy.device)
        z_L = z_L.to(noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(
                enc_emb, z_H, z_L, puzzle_ids
            )

        spatial_cond = self._logits_to_spatial(logits.float())
        noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits


# ── Painter-thinker (V1: image+noisy-conditioned) ────────────────────────────

class OriginalTRMRatatouilleV1(OriginalTRMRatatouilleV0):
    """
    Same as V0 but the encoder sees cat(condition, noisy_image) (2 channels).

    The thinker reasons from a noisy/corrupted signal, removing the clean-input
    training wheel present in V0.

    Inherits everything from V0; only differences:
      - image_encoder uses SpatialEncoder(2, ...) instead of SpatialEncoder(1, ...)
      - _get_enc_emb concatenates condition + noisy before encoding
    """

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        num_classes: int = 9,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        thinker_out_channels: int = None,
        # --- timestep conditioning (V1-specific) ---
        enc_timestep_cond: bool = False,     # FiLM scale+shift on encoder features
        thinker_timestep_cond: bool = False, # T2: broadcast temb added to thinker input tokens
        temb_dim: int = 256,                 # output dim of the shared TimestepMLP
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            num_classes=num_classes,
            thinker_out_channels=thinker_out_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            painter_dtype=painter_dtype,
        )
        # Replace 1-channel encoder with 2-channel (condition + noisy)
        self.image_encoder = SpatialEncoder(2, enc_channels, factor=cell_size)

        # Timestep conditioning.  Both projections are zero-init so the model
        # starts as the no-timestep identity and gradually learns to use t.
        self.enc_timestep_cond     = enc_timestep_cond
        self.thinker_timestep_cond = thinker_timestep_cond
        if enc_timestep_cond or thinker_timestep_cond:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=temb_dim)
        if enc_timestep_cond:
            self.enc_film = nn.Linear(temb_dim, 2 * enc_channels)
            nn.init.zeros_(self.enc_film.weight)
            nn.init.zeros_(self.enc_film.bias)
        if thinker_timestep_cond:
            self.thinker_temb_proj = nn.Linear(temb_dim, hidden_size)
            nn.init.zeros_(self.thinker_temb_proj.weight)
            nn.init.zeros_(self.thinker_temb_proj.bias)

    def _get_enc_emb(
        self, condition: torch.Tensor, noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute shared timestep embedding once (if any conditioning is active).
        temb = None
        if timesteps is not None and (self.enc_timestep_cond or self.thinker_timestep_cond):
            temb = self.timestep_mlp(timesteps)

        # Encode cat(condition, noisy) with optional encoder FiLM.
        feat = self.image_encoder(torch.cat([condition, noisy], dim=1))
        if temb is not None and self.enc_timestep_cond:
            scale, shift = self.enc_film(temb).chunk(2, dim=1)  # (B, enc_channels) each
            feat = feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        proj    = self.enc_proj(feat)
        enc_emb = proj.flatten(2).transpose(1, 2)          # (B, 81, hidden_size)

        # T2: broadcast timestep embedding into thinker token space.
        if temb is not None and self.thinker_timestep_cond:
            enc_emb = enc_emb + self.thinker_temb_proj(temb).unsqueeze(1)

        return enc_emb


# ── Painter-thinker (V2: no CE supervision) ───────────────────────────────────

class OriginalTRMRatatouilleV2(OriginalTRMRatatouilleV1):
    """
    Same as V1 but with no sudoku CE loss and unconstrained thinker output channels.

    Training wheel removed: the thinker gets no explicit digit-level supervision.
    The thinker output is a latent spatial map (thinker_out_channels, 9, 9) which
    the bridge upsamples to condition the painter, but its CE loss is suppressed by
    returning None logits so the training loop skips it.

    Use thinker_out_channels=16 (or any value) instead of num_classes=9.
    """

    has_realsolution_eval: bool = False   # latent thinker; no digit-level solution eval

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        thinker_out_channels: int = 16,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            num_classes=thinker_out_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            painter_dtype=painter_dtype,
        )

    def reasoning_step(self, condition, noisy, z_H, z_L, timesteps,
                       puzzle_ids=None, H_cycles=None, L_cycles=None):
        noise_pred, _logits, z_H_next, z_L_next = super().reasoning_step(
            condition, noisy, z_H, z_L, timesteps,
            puzzle_ids=puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles,
        )
        return noise_pred, None, z_H_next, z_L_next

    def forward(self, noisy, timesteps, condition, puzzle_ids=None):
        noise_pred, _logits = super().forward(noisy, timesteps, condition, puzzle_ids)
        return noise_pred, None


# ── Painter-thinker (V3: larger latent, same as V2) ───────────────────────────

class OriginalTRMRatatouilleV3(OriginalTRMRatatouilleV2):
    """
    Same as V2 but with a larger thinker latent (thinker_out_channels=64).

    The only difference from V2 is the default output dimensionality — the
    architecture, loss (no CE), and encoder (condition+noisy) are identical.
    Use this when you want a higher-capacity thinker latent for the bridge.
    """

    def __init__(self, thinker_out_channels: int = 64, **kwargs):
        super().__init__(thinker_out_channels=thinker_out_channels, **kwargs)


# ── Painter-thinker (V4: AttentiveBridge + decoupled compression factor) ──────

class OriginalTRMRatatouilleV4(OriginalTRMRatatouilleV3):
    """
    Same as V3 but the thinker grid topology is decoupled from the puzzle cell
    structure via an independent compression_factor, and SpatialBridge is
    replaced by AttentiveBridge (Perceiver-IO cross-attention upsampling).

    Key differences from V3:
      - compression_factor controls encoder downsampling (may differ from cell_size)
      - thinker seq_len = (painter_size // compression_factor)²
      - Bridge: AttentiveBridge with learned positional queries upsamples the
        low-res thinker output to painter_size × painter_size
      - bridge_num_heads: attention heads in AttentiveBridge
    """

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,            # only for _run_painter noisy input shape
        compression_factor: int = 16,   # encoder + thinker grid factor
        # --- thinker ---
        thinker_out_channels: int = 64,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        # --- bridge & painter ---
        bridge_channels: int = 16,
        bridge_num_heads: int = 4,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        grid_size = painter_size // compression_factor
        seq_len   = grid_size * grid_size

        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            thinker_out_channels=thinker_out_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            painter_dtype=painter_dtype,
        )

        # Thinker grid is compression_factor-based, not cell_size-based
        self._grid = grid_size

        # Replace 1→2 channel encoder (set by V1) with compression_factor version
        self.image_encoder = SpatialEncoder(2, enc_channels, factor=compression_factor)

        # Replace SpatialBridge with AttentiveBridge
        self.bridge = AttentiveBridge(
            in_channels=thinker_out_channels,
            out_channels=bridge_channels,
            out_resolution=painter_size,
            factor=compression_factor,
            num_heads=bridge_num_heads,
        )


# ── Standalone painter (sanity check: conditioned on real solutions) ───────────

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

    token_input: bool = False             # uses solution tokens, handled via _get_condition
    has_realsolution_eval: bool = True   # realsolution IS the only conditioning

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
        self.cfg_prob   = cfg_prob
        self.cfg_scale  = cfg_scale   # used at inference time; overridable externally
        self._painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype]
            if painter_dtype is not None else None
        )
        self.bridge = SpatialBridge(
            in_channels=vocab_size,
            out_channels=bridge_channels,
            painter_size=painter_size,
        )
        self.painter = _make_painter(
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
        condition: torch.Tensor,          # (B, 81) long solution tokens 2-10
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
            null   = torch.zeros_like(spatial)
            s_both = torch.cat([spatial, null], dim=0)
            n_both = noisy.repeat(2, 1, 1, 1)
            t_both = timesteps.repeat(2)
            with ctx:
                bf    = self.bridge(s_both)
                pred  = self.painter(torch.cat([n_both, bf], dim=1), t_both).sample
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            bridge_feat = self.bridge(spatial)
            noise_pred  = self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample
        return noise_pred, None


# ── Thinker with frozen painter ───────────────────────────────────────────────

class ThinkerWithFrozenPainter(OriginalTRMRatatouilleV0Tok):
    """
    Trains only the thinker; bridge + UNet are loaded from a pretrained
    StandalonePainter checkpoint and kept frozen throughout.

    Inherits everything from OriginalTRMRatatouilleV0Tok except:
      - bridge and painter come from a pre-built StandalonePainter (no new weights)
      - those parameters are frozen (requires_grad=False)
      - get_painter_params() returns [] so the optimizer never touches them

    Usage:
      python train_trm.py experiment=thinker_frozen_painter \\
        painter.painter_checkpoint=runs/standalone_painter/checkpoint_final.pt
    """

    def __init__(self, painter: StandalonePainter, **thinker_kwargs):
        super().__init__(**thinker_kwargs)
        # Replace the freshly-built bridge+painter with the pretrained frozen ones.
        self.bridge  = painter.bridge
        self.painter = painter.painter
        for p in self.bridge.parameters():
            p.requires_grad_(False)
        for p in self.painter.parameters():
            p.requires_grad_(False)

    def get_painter_params(self) -> list:
        return []  # frozen — excluded from all optimizers
