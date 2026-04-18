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

import torch
import torch.nn as nn

from typing import Optional

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from mnist_sudoku_models import SpatialBridge, _make_painter


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
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(config)

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
    ):
        """
        One supervision step. Internally runs H_cycles-1 no-grad cycles then
        one full-grad cycle (matching the original training pattern).

        Returns: (logits, z_H_detached, z_L_detached)
          logits: (B, seq_len, vocab_size) — gradients attached
        """
        bsz = inputs.shape[0]
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(bsz, dtype=torch.int32, device=inputs.device)
        carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
        new_carry, logits, _ = self.inner(carry, {"inputs": inputs, "puzzle_identifiers": puzzle_ids})
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
        self._grid = painter_size // cell_size   # e.g. 144//16 = 9
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
        """(B, 81, vocab_size) → (B, vocab_size, grid, grid)"""
        B, _, C = logits.shape
        return logits.transpose(1, 2).reshape(B, C, self._grid, self._grid)

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
    ):
        """
        One supervision step: thinker → bridge → painter.

        diff_thinker_weight scales the gradient that diffusion loss sends back
        through the bridge into the thinker (1.0 = full, 0.0 = detached).
        The sudoku CE loss always flows through unscaled logits.

        Returns: (noise_pred, sudoku_logits, z_H_detached, z_L_detached)
        """
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            puzzle_tokens, z_H, z_L, puzzle_ids
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
