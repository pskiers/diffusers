"""
SudokuTRM – Tiny Recursive Model for Sudoku solving.

Architecture mirrors the reference TinyRecursiveReasoningModel (ACTV1):

  Recursion structure:
    For each n_sup step:
        H_cycles-1 without grad:
            for L_cycles: z_L = blocks( input_emb + z_H + z_L )
            z_H = blocks( z_H + z_L )  [no input]
        1 final H_cycle with gradients (for backprop):
            for L_cycles: z_L = blocks( input_emb + z_H + z_L )
            z_H = blocks( z_H + z_L )
        logits = lm_head( out_norm( z_H[:, puzzle_emb_len:] ) )

  Key design choices matching the reference:
    1. Learnable z_H / z_L initial states (trunc_normal std=1).
    2. Post-norm blocks: residual → RMSNorm  (vs. pre-norm LayerNorm).
    3. SwiGLU feed-forward network  (vs. GELU 4× FFN).
    4. Rotary positional embeddings (RoPE)  (vs. learned absolute).
    6. Puzzle-ID token prepended to sequence  (vs. broadcast addition).

  TransformerBlock.forward() accepts an optional freqs_cis so that
  SpatialTRM (which imports this block) can call it without RoPE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utilities ──────────────────────────────────────────────────────────────────

def _find_multiple(n: int, k: int) -> int:
    """Smallest multiple of k that is >= n."""
    return ((n + k - 1) // k) * k


# ── Normalization ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ── Rotary Positional Embedding ───────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Precomputed RoPE complex-exponential frequencies.

    Registers `freqs_cis` as a non-persistent buffer so it travels with the
    model's device but is not saved in checkpoints (recomputed on load).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 256, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t     = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)                        # (max_seq_len, head_dim//2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.freqs_cis[:seq_len]


def _apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings in-place to Q and K projections.

    q, k      : (B, T, n_heads, head_dim)
    freqs_cis : (T, head_dim//2)  complex
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_r = x.float().reshape(*x.shape[:-1], -1, 2)
        x_c = torch.view_as_complex(x_r.contiguous())
        x_c = x_c * freqs_cis.unsqueeze(0).unsqueeze(2)   # (B, T, heads, head_dim//2)
        return torch.view_as_real(x_c).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


# ── Feed-Forward: SwiGLU ──────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (no bias).
    Intermediate dim = find_multiple(round(d_model * mlp_ratio * 2/3), 64)
    — approximately iso-parametric to a GELU FFN with the same mlp_ratio.
    """

    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        inter = _find_multiple(round(d_model * mlp_ratio * 2 / 3), 64)
        self.gate_up = nn.Linear(d_model, inter * 2, bias=False)
        self.down    = nn.Linear(inter,   d_model,   bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


# ── Self-Attention ────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Multi-head self-attention with optional RoPE.  No bias; uses PyTorch SDPA.

    freqs_cis is None  →  standard (position-unaware) attention.
    freqs_cis provided →  Q and K are rotated before dot-product.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cis=None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            q, k = _apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)   # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(out.transpose(1, 2).contiguous().view(B, T, C))


# ── Building blocks ────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Post-norm transformer block: x = RMSNorm(x + attn(x)), x = RMSNorm(x + mlp(x)).
    Uses SwiGLU MLP and bias-free projections.  Matches the reference TRM block.

    freqs_cis (optional):
      Pass RoPE frequencies from a RotaryEmbedding to enable rotary positions.
      When None, attention is position-unaware — preserving backward compatibility
      with SpatialTRM, which uses learned 2-D positional embeddings instead.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn  = SelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.mlp   = SwiGLU(d_model, mlp_ratio)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, freqs_cis=None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, freqs_cis))
        x = self.norm2(x + self.mlp(x))
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
        vocab_size:      Total token vocabulary (default 11).
        seq_len:         Sequence length (default 81 for 9×9).
        d_model:         Hidden dimension.
        n_heads:         Attention heads (must divide d_model).
        n_layers:        Transformer layers per block call.
        L_cycles:        Inner (L-level) iterations per H-cycle.
        H_cycles:        Outer (H-level) macroscopic cycles.
        n_sup:           Supervision loop depth.
        dropout:         Dropout probability (0 for inference).
        num_puzzle_ids:  If set, prepend a per-puzzle token to the sequence.
        rope_theta:      RoPE base frequency (default 10000).
    """

    BLANK_TOKEN: int = 1

    def __init__(
        self,
        vocab_size:      int          = 11,
        seq_len:         int          = 81,
        d_model:         int          = 256,
        n_heads:         int          = 4,
        n_layers:        int          = 2,
        L_cycles:        int          = 6,
        H_cycles:        int          = 3,
        n_sup:           int          = 4,
        dropout:         float        = 0.0,
        num_puzzle_ids:  int | None   = None,
        rope_theta:      float        = 10000.0,
    ):
        super().__init__()
        self.L_cycles   = L_cycles
        self.H_cycles   = H_cycles
        self.n_sup      = n_sup
        self.seq_len    = seq_len
        self.vocab_size = vocab_size

        # ── Puzzle-identifier prefix token ───────────────────────────────────
        # When enabled, a single learned token (zero-initialised) is prepended to
        # the sequence.  ID 0 is "unknown / no identifier" → padding_idx=0.
        if num_puzzle_ids is not None and num_puzzle_ids > 0:
            self.puzzle_emb_len      = 1
            self.puzzle_id_embedding = nn.Embedding(
                num_puzzle_ids + 1, d_model, padding_idx=0
            )
        else:
            self.puzzle_emb_len      = 0
            self.puzzle_id_embedding = None

        total_len = seq_len + self.puzzle_emb_len   # full sequence length

        # ── Token embedding ──────────────────────────────────────────────────
        # Scale by sqrt(d_model) so per-element std ≈ 1.0 (matches reference).
        # No positional embedding — RoPE handles positions.
        self.embed_scale = math.sqrt(d_model)
        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # ── Rotary positional encoding ───────────────────────────────────────
        head_dim = d_model // n_heads
        self.rotary_emb = RotaryEmbedding(
            head_dim, max_seq_len=total_len + 4, theta=rope_theta
        )

        # ── Shared core model ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Output head ──────────────────────────────────────────────────────
        # Final RMSNorm before projection (last block is already post-normed,
        # but an extra norm stabilises the lm_head scale across training).
        self.out_norm = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # ── Learnable initial states ─────────────────────────────────────────
        # Reference uses trunc_normal(std=1) — non-zero init matters because the
        # model needs a good prior for the first reasoning step.  Stored as
        # nn.Parameter so the optimizer trains them via backprop.
        self.z_H_init = nn.Parameter(torch.empty(1, total_len, d_model))
        self.z_L_init = nn.Parameter(torch.empty(1, total_len, d_model))

        self._init_weights()

    # ── Weight initialisation ───────────────────────────────────────────────

    def _init_weights(self) -> None:
        embed_std = 1.0 / self.embed_scale
        nn.init.normal_(self.embedding.weight, std=embed_std)
        nn.init.normal_(self.lm_head.weight,   std=embed_std)
        # Learnable initial states: truncated normal std=1 (reference convention)
        nn.init.trunc_normal_(self.z_H_init, std=1.0)
        nn.init.trunc_normal_(self.z_L_init, std=1.0)
        # Puzzle prefix: zero-initialised so it starts as a no-op
        if self.puzzle_id_embedding is not None:
            nn.init.zeros_(self.puzzle_id_embedding.weight)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # ── Utilities ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed(
        self,
        inputs:     torch.Tensor,
        puzzle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs:     (B, 81) long  →  (B, [1+]81, d_model)
        puzzle_ids: (B,)    long  – per-sample puzzle identifier (optional).

        When num_puzzle_ids was set at construction time, a single prefix token
        is always prepended (using ID=0 / zero embedding when puzzle_ids=None).
        """
        x = self.embed_scale * self.embedding(inputs)   # (B, 81, d_model)

        if self.puzzle_id_embedding is not None:
            if puzzle_ids is None:
                puzzle_ids = torch.zeros(
                    inputs.shape[0], dtype=torch.long, device=inputs.device
                )
            puzzle_token = self.puzzle_id_embedding(puzzle_ids).unsqueeze(1)  # (B,1,d)
            x = torch.cat([puzzle_token, x], dim=1)                          # (B,82,d)

        return x

    def get_initial_states(self, bsz: int):
        """Expand learned initial states to the given batch size."""
        z_H = self.z_H_init.expand(bsz, -1, -1).clone()
        z_L = self.z_L_init.expand(bsz, -1, -1).clone()
        return z_H, z_L

    # ── Core recursion ──────────────────────────────────────────────────────

    def _run_blocks(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, freqs_cis)
        return x

    def reasoning_step(
        self,
        input_emb: torch.Tensor,
        z_H:       torch.Tensor,
        z_L:       torch.Tensor,
    ):
        """
        One n_sup supervision step.

        Runs (H_cycles-1) macroscopic cycles without gradients, then a single
        final cycle with full gradients — exactly matching the reference pattern.

        Returns:
            logits  – (B, 81, vocab_size), gradients attached
            z_H_det – detached z_H for the next supervision step
            z_L_det – detached z_L for the next supervision step
        """
        T         = input_emb.shape[1]
        freqs_cis = self.rotary_emb(T)

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self._run_blocks(input_emb + z_H + z_L, freqs_cis)
                z_H = self._run_blocks(z_H + z_L, freqs_cis)

        # Final H_cycle — gradients flow here
        for _ in range(self.L_cycles):
            z_L = self._run_blocks(input_emb + z_H + z_L, freqs_cis)
        z_H = self._run_blocks(z_H + z_L, freqs_cis)

        # Slice off the puzzle prefix token before projecting to logits
        logits = self.lm_head(self.out_norm(z_H[:, self.puzzle_emb_len:]))
        return logits, z_H.detach(), z_L.detach()

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        inputs:     torch.Tensor,
        n_sup:      int | None          = None,
        puzzle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full inference pass.

        Runs n_sup × H_cycles × L_cycles iterations uniformly (no grad split).
        input_emb is recomputed each n_sup step, matching the training loop.

        Args:
            inputs:     (B, 81) long token IDs
            n_sup:      Override number of supervision steps (defaults to self.n_sup)
            puzzle_ids: (B,) long per-sample puzzle identifiers (optional)

        Returns:
            logits: (B, 81, vocab_size)
        """
        n_sup    = n_sup if n_sup is not None else self.n_sup
        z_H, z_L = self.get_initial_states(inputs.shape[0])
        z_H      = z_H.to(inputs.device)
        z_L      = z_L.to(inputs.device)

        for _ in range(n_sup):
            input_emb = self.embed(inputs, puzzle_ids=puzzle_ids)
            T         = input_emb.shape[1]
            freqs_cis = self.rotary_emb(T)
            for _ in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    z_L = self._run_blocks(input_emb + z_H + z_L, freqs_cis)
                z_H = self._run_blocks(z_H + z_L, freqs_cis)

        return self.lm_head(self.out_norm(z_H[:, self.puzzle_emb_len:]))
