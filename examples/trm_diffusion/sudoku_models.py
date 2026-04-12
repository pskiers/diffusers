"""
SudokuTRM – Tiny Recursive Model for Sudoku solving.

Architecture mirrors TRMv2 (UNetTRMv2 in trm_models.py) but adapted for
discrete sequences instead of continuous images:

  Recursion structure (identical to UNetTRMv2):
    For each n_sup step:
        H_cycles-1 without grad:
            for L_cycles: z_L = norm_z_L( blocks( input_emb + z_H + z_L ) )
            z_H = norm_z_H( blocks( z_H + z_L ) )  [no input]
        1 final H_cycle with gradients (for backprop):
            for L_cycles: z_L = norm_z_L( blocks( input_emb + z_H + z_L ) )
            z_H = norm_z_H( blocks( z_H + z_L ) )
        logits = lm_head( out_norm( z_H ) )

  Mapping to diffusion TRM terms:
    L_cycles  ↔  n   (inner iterations, like _latent_recursion loop)
    H_cycles  ↔  T   (macroscopic steps, like _deep_recursion)
    n_sup     ↔  n_sup (outer supervision loop)
    z_H       ↔  y   (slow state)
    z_L       ↔  z   (fast state)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm GPT-style transformer block: LayerNorm → SelfAttn → residual,
    LayerNorm → MLP(GELU) → residual.

    This is the sequence analogue of the UNet core_model used inside TRMv2.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────────

class SudokuTRM(nn.Module):
    """
    Tiny Recursive Model for 9×9 Sudoku.

    Token convention (matches sudoku_dataset.py):
        0  – PAD / ignore
        1  – blank cell in the puzzle
        2–10 – given/solution digits 1–9

    Args:
        vocab_size: Total token vocabulary (default 11).
        seq_len:    Sequence length (default 81 for 9×9).
        d_model:    Hidden dimension.
        n_heads:    Attention heads (must divide d_model).
        n_layers:   Transformer layers per block call.
        L_cycles:   Inner (L-level) iterations per H-cycle.
        H_cycles:   Outer (H-level) macroscopic cycles.
        n_sup:      Supervision loop depth.
        dropout:    Dropout probability (0 for inference).
    """

    BLANK_TOKEN: int = 1  # token ID for blank cells

    def __init__(
        self,
        vocab_size: int = 11,
        seq_len: int    = 81,
        d_model: int    = 256,
        n_heads: int    = 4,
        n_layers: int   = 2,
        L_cycles: int   = 6,
        H_cycles: int   = 3,
        n_sup: int      = 4,
        dropout: float  = 0.0,
        num_puzzle_ids: int | None = None,
    ):
        super().__init__()
        self.L_cycles  = L_cycles
        self.H_cycles  = H_cycles
        self.n_sup     = n_sup
        self.seq_len   = seq_len
        self.vocab_size = vocab_size

        # ── Input processing ────────────────────────────────────────────────
        self.embedding     = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ── Optional puzzle-identifier embedding ─────────────────────────────
        # When enabled, a learned per-puzzle offset is added to the input embedding.
        # ID 0 is reserved as "unknown/no identifier" → padding_idx=0.
        if num_puzzle_ids is not None and num_puzzle_ids > 0:
            self.puzzle_id_embedding = nn.Embedding(num_puzzle_ids + 1, d_model, padding_idx=0)
            nn.init.normal_(self.puzzle_id_embedding.weight, std=0.02)
        else:
            self.puzzle_id_embedding = None

        # ── Shared core model ───────────────────────────────────────────────
        # The SAME blocks are used for both z_L and z_H updates (same as
        # UNetTRMv2 reusing its core_model for all recursive steps).
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── State normalisations (mirrors norm_y / norm_z in UNetTRMv2) ────
        self.norm_z_H = nn.LayerNorm(d_model)
        self.norm_z_L = nn.LayerNorm(d_model)

        # ── Output head ─────────────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # ── Initial states (zeros, like y_init / z_init in old TRM) ────────
        self.register_buffer("z_H_init", torch.zeros(1, seq_len, d_model))
        self.register_buffer("z_L_init", torch.zeros(1, seq_len, d_model))

        self._init_weights()

    # ── Weight initialisation ───────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight,   std=0.02)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ── Utilities ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed(self, inputs: torch.Tensor, puzzle_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        inputs:     (B, 81) long  →  (B, 81, d_model)
        puzzle_ids: (B,)    long  – per-sample puzzle identifier (optional).
                    Only used when num_puzzle_ids was set at construction time.
        """
        x = self.embedding(inputs) + self.pos_embedding
        if self.puzzle_id_embedding is not None and puzzle_ids is not None:
            # puzzle_id_embedding is (B, d_model); broadcast across seq_len
            x = x + self.puzzle_id_embedding(puzzle_ids).unsqueeze(1)
        return x

    def get_initial_states(self, bsz: int):
        """Zero-initialised (z_H, z_L) expanded to the given batch size."""
        z_H = self.z_H_init.expand(bsz, -1, -1).clone()
        z_L = self.z_L_init.expand(bsz, -1, -1).clone()
        return z_H, z_L

    # ── Core recursion ──────────────────────────────────────────────────────

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

        Runs (H_cycles-1) macroscopic cycles without gradients, then a single
        final cycle with full gradients – exactly matching UNetTRMv2's
        _deep_recursion + _latent_recursion pattern.

        Returns:
            logits  – (B, 81, vocab_size), gradients attached
            z_H_det – detached z_H for the next supervision step
            z_L_det – detached z_L for the next supervision step
        """
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.norm_z_L(self._run_blocks(input_emb + z_H + z_L))
                z_H = self.norm_z_H(self._run_blocks(z_H + z_L))

        # Final H_cycle – gradients flow here
        for _ in range(self.L_cycles):
            z_L = self.norm_z_L(self._run_blocks(input_emb + z_H + z_L))
        z_H = self.norm_z_H(self._run_blocks(z_H + z_L))

        logits = self.lm_head(self.out_norm(z_H))
        return logits, z_H.detach(), z_L.detach()

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        n_sup: int | None = None,
        puzzle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full inference pass.

        Args:
            inputs:     (B, 81) long token IDs
            n_sup:      Override number of supervision steps (defaults to self.n_sup)
            puzzle_ids: (B,) long per-sample puzzle identifiers (optional)

        Returns:
            logits: (B, 81, vocab_size)
        """
        n_sup     = n_sup if n_sup is not None else self.n_sup
        input_emb = self.embed(inputs, puzzle_ids=puzzle_ids)
        z_H, z_L  = self.get_initial_states(inputs.shape[0])
        z_H       = z_H.to(inputs.device)
        z_L       = z_L.to(inputs.device)

        logits = None
        for _ in range(n_sup):
            logits, z_H, z_L = self.reasoning_step(input_emb, z_H, z_L)
        return logits
