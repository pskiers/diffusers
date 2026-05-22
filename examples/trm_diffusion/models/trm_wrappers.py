from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional

from models.trm.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from models.trm.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


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
