"""
Discrete TRM Diffusion — core model for masked discrete diffusion on Sokoban boards.

Architecture: Option III hybrid — a small number of diffusion steps (T=50-200),
each step runs a TRM reasoning loop with dynamic ACT halting, and the carry
(z_H, z_L) persists across diffusion timesteps.

The model predicts the clean board x_0 from a partially-masked input x_t using
cross-entropy loss over masked positions only.

Board: 12x12 = 144 tokens, vocab = {0=MASK, 1=WALL, ..., 7=PLAYER_ON_TARGET}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)


MASK_TOKEN_ID = 0


# ─── Timestep embedding ──────────────────────────────────────────────────────


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for diffusion timesteps.

    Args:
        timesteps: [B] int or float tensor
        dim: embedding dimension (must be even)

    Returns: [B, dim] float32 tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


# ─── adaLN-Zero TRM Block ────────────────────────────────────────────────────


class AdaLNTRMBlock(nn.Module):
    """Transformer block with adaLN-Zero timestep conditioning (DiT-style).

    Pre-norm + gated residual: identity at init because gate scalars start at 0.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps

        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )

        self.ffn = SwiGLU(hidden_size=hidden_size, expansion=expansion)

        # adaLN-Zero: produces (γ₁, β₁, α₁, γ₂, β₂, α₂) from timestep embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            CastedLinear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init the linear so gates start at 0 → identity at init
        linear = self.adaLN_modulation[1]
        assert isinstance(linear, CastedLinear)
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
        t_emb: torch.Tensor,          # [B, D]
        cos_sin: CosSin,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [B, 1, 6*D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Pre-attention adaLN
        h_norm = rms_norm(hidden_states, self.rms_norm_eps) * (1.0 + gamma1) + beta1
        h_attn = self.self_attn(cos_sin=cos_sin, hidden_states=h_norm)
        hidden_states = hidden_states + alpha1 * h_attn

        # Pre-FFN adaLN
        h_norm = rms_norm(hidden_states, self.rms_norm_eps) * (1.0 + gamma2) + beta2
        h_ffn = self.ffn(h_norm)
        hidden_states = hidden_states + alpha2 * h_ffn

        return hidden_states


# ─── Reasoning module (sequence of blocks) ────────────────────────────────────


class AdaLNReasoningModule(nn.Module):
    """L_level: stack of AdaLNTRMBlock with additive input injection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        n_layers: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            AdaLNTRMBlock(hidden_size, num_heads, expansion, rms_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, L, D]
        input_injection: torch.Tensor,  # [B, L, D]
        t_emb: torch.Tensor,            # [B, D]
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for block in self.blocks:
            hidden_states = block(hidden_states, t_emb, cos_sin)
        return hidden_states


# ─── Carry dataclass ──────────────────────────────────────────────────────────


@dataclass
class DiscreteTRMCarry:
    z_H: torch.Tensor  # [B, seq_len, D]
    z_L: torch.Tensor  # [B, seq_len, D]


# ─── Core model ───────────────────────────────────────────────────────────────


class DiscreteTRMDiffusion(nn.Module):
    """Discrete TRM denoiser for masked diffusion on Sokoban boards.

    Shape contract:
        x_t:      [B, 144]  long  tokens in {0,...,7}
        timestep: [B]        long  in {1,...,T}
        carry:    DiscreteTRMCarry | None

    Returns:
        logits:    [B, 144, vocab_size]
        q_logits:  (q_halt [B], q_continue [B])  both float32
        new_carry: DiscreteTRMCarry (detached)
    """

    def __init__(
        self,
        vocab_size: int = 8,
        seq_len: int = 144,
        hidden_size: int = 512,
        num_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        pos_encodings: str = "rope",
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        self.forward_dtype = getattr(torch, forward_dtype)
        self.rms_norm_eps = rms_norm_eps

        embed_scale = math.sqrt(hidden_size)
        self.embed_scale = embed_scale
        embed_init_std = 1.0 / embed_scale

        self.embed_tokens = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        self.pos_encodings = pos_encodings
        if pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=hidden_size // num_heads,
                max_position_embeddings=seq_len,
                base=rope_theta,
            )
        elif pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                seq_len, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
            )

        self.t_mlp = nn.Sequential(
            CastedLinear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size, hidden_size, bias=True),
        )

        self.L_level = AdaLNReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            n_layers=L_layers,
            rms_norm_eps=rms_norm_eps,
        )

        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q_head = CastedLinear(hidden_size, 2, bias=True)

        _h_init = trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1)
        _l_init = trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1)
        self.register_buffer("H_init", _h_init, persistent=True)
        self.register_buffer("L_init", _l_init, persistent=True)
        self.H_init: torch.Tensor
        self.L_init: torch.Tensor

        self.final_norm_weight = nn.Parameter(torch.ones(hidden_size))

        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5)

    def empty_carry(self, batch_size: int, device: torch.device) -> DiscreteTRMCarry:
        return DiscreteTRMCarry(
            z_H=self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1).clone(),
            z_L=self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1).clone(),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        carry: Optional[DiscreteTRMCarry] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], DiscreteTRMCarry]:
        B = x_t.shape[0]
        device = x_t.device

        # Input embeddings [B, 144, D]
        token_emb = self.embed_tokens(x_t.to(torch.int32))
        if self.pos_encodings == "learned":
            pos = self.embed_pos.embedding_weight.to(self.forward_dtype)
            x_emb = self.embed_scale * (token_emb + pos)
        else:
            x_emb = self.embed_scale * token_emb

        # Timestep embedding [B, D]
        t_emb_sin = sinusoidal_embedding(timestep, self.hidden_size).to(self.forward_dtype)
        t_emb = self.t_mlp(t_emb_sin)

        # RoPE
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # Initialize carry
        if carry is None:
            carry = self.empty_carry(B, device)
        z_H = carry.z_H.to(device)
        z_L = carry.z_L.to(device)

        # TRM reasoning loop: H_cycles-1 without grad, final with grad
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + x_emb, t_emb, cos_sin)
                z_H = self.L_level(z_H, z_L, t_emb, cos_sin)

        for _ in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + x_emb, t_emb, cos_sin)
        z_H = self.L_level(z_H, z_L, t_emb, cos_sin)

        # Output
        z_H_normed = rms_norm(z_H, self.rms_norm_eps) * self.final_norm_weight.to(z_H.dtype)
        logits = self.lm_head(z_H_normed)  # [B, 144, V]

        q_input = z_H.mean(dim=1)  # [B, D]
        q_logits = self.q_head(q_input).to(torch.float32)  # [B, 2]

        new_carry = DiscreteTRMCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return logits, (q_logits[..., 0], q_logits[..., 1]), new_carry


# ─── ACT wrapper ──────────────────────────────────────────────────────────────


class ACTMaskedDiffusionWrapper(nn.Module):
    """ACT wrapper for Discrete TRM Diffusion.

    Runs the inner model up to halt_max_steps times per diffusion timestep.
    Each step: forward full batch → check halt → store final logits for
    newly halted samples. Loss is computed once at the end on the stored logits.

    The full batch is forwarded every step (halted samples included) to keep
    tensor shapes static and avoid dynamic-batch overhead. Halted samples
    don't contribute to loss; their extra FLOPs are the cost of simplicity.
    """

    def __init__(
        self,
        inner: DiscreteTRMDiffusion,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        halt_loss_weight: float = 0.5,
        use_carry_recycling: bool = False,
        carry_recycle_prob: float = 0.5,
    ):
        super().__init__()
        self.inner = inner
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.halt_loss_weight = halt_loss_weight
        self.use_carry_recycling = use_carry_recycling
        self.carry_recycle_prob = carry_recycle_prob

    def forward(
        self,
        x_t: torch.Tensor,          # [B, 144] masked tokens
        x_0: torch.Tensor,          # [B, 144] clean tokens (target)
        timestep: torch.Tensor,      # [B] diffusion timestep
        diffusion_carry: Optional[DiscreteTRMCarry] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object], DiscreteTRMCarry]:
        """
        Returns:
            total_loss:  scalar (differentiable)
            metrics:     dict of python floats
            final_carry: DiscreteTRMCarry (detached)
        """
        B = x_t.shape[0]
        device = x_t.device
        is_masked = (x_t == MASK_TOKEN_ID)  # [B, 144]

        # Labels: -100 for unmasked → CE ignores them natively
        labels = x_0.clone()  # [B, 144]
        labels[~is_masked] = -100

        mask_counts = is_masked.sum(dim=1).clamp_min(1).float()  # [B]

        # ── Carry recycling ──
        if diffusion_carry is not None:
            carry = diffusion_carry
        elif (
            self.training
            and self.use_carry_recycling
            and torch.rand(1, device=device).item() < self.carry_recycle_prob
        ):
            with torch.no_grad():
                _, _, cold_carry = self.inner(x_t, timestep, carry=None)
            carry = cold_carry
        else:
            carry = None

        # ── ACT loop ──
        steps = torch.zeros(B, dtype=torch.int32, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        # We'll store the logits/q_halt from each sample's halt step.
        # Use the last iteration's logits for the loss (they participate in
        # the computation graph because we only store on the final iteration
        # that matters). For simplicity, we keep the logits from the very
        # last forward pass and compute loss on the full batch — the
        # halt-exploration logic just encourages the q_head to learn when
        # to stop, but the CE loss is always on the final logits.
        last_logits = None
        last_q_halt = None

        for k in range(self.halt_max_steps):
            logits, (q_halt, _), new_carry = self.inner(x_t, timestep, carry)

            steps = steps + (~halted).int()
            is_last = (steps >= self.halt_max_steps)

            with torch.no_grad():
                halt_now = is_last.clone()
                if self.training and self.halt_max_steps > 1:
                    halt_now = halt_now | (q_halt > 0)
                    if self.halt_exploration_prob > 0:
                        explore = torch.rand(B, device=device) < self.halt_exploration_prob
                        min_steps = torch.randint(2, self.halt_max_steps + 1, (B,), device=device)
                        halt_now = halt_now & ((steps >= min_steps) | ~explore)
                else:
                    halt_now = is_last
                halt_now = halt_now & ~halted

            halted = halted | halt_now
            carry = DiscreteTRMCarry(z_H=new_carry.z_H, z_L=new_carry.z_L)
            last_logits = logits
            last_q_halt = q_halt

            if halted.all():
                break

        # ── Loss on final logits (through computation graph) ──
        assert last_logits is not None and last_q_halt is not None

        per_token_loss = F.cross_entropy(
            last_logits.float().reshape(-1, self.inner.vocab_size),  # [B*144, V]
            labels.reshape(-1),                                      # [B*144]
            ignore_index=-100,
            reduction="none",
        ).reshape(B, self.inner.seq_len)  # [B, 144]

        per_sample_loss = per_token_loss.sum(dim=1) / mask_counts  # [B]
        lm_loss = per_sample_loss.mean()

        # Halt loss: BCE(q_halt, all_masked_correct)
        with torch.no_grad():
            preds = last_logits.detach().argmax(dim=-1)  # [B, 144]
            correct_at_mask = (preds == x_0) | ~is_masked
            seq_correct = correct_at_mask.all(dim=1).float()  # [B]

            n_correct = ((preds == x_0) & is_masked).float().sum().item()
            n_masked = is_masked.float().sum().item()

        halt_loss = F.binary_cross_entropy_with_logits(
            last_q_halt, seq_correct, reduction="mean"
        )

        total_loss = lm_loss + self.halt_loss_weight * halt_loss

        # ── Metrics (all python floats — safe for logging) ──
        metrics: Dict[str, object] = {
            "lm_loss": lm_loss.detach().item(),
            "halt_loss": halt_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "token_accuracy": n_correct / max(n_masked, 1.0),
            "exact_accuracy": seq_correct.mean().item(),
            "avg_halt_steps": steps.float().mean().item(),
            "halt_step_std": steps.float().std().item() if B > 1 else 0.0,
            "q_halt_mean": last_q_halt.detach().mean().item(),
            "q_halt_pos_frac": (last_q_halt.detach() > 0).float().mean().item(),
            "mask_ratio": n_masked / max(B * self.inner.seq_len, 1),
        }

        return total_loss, metrics, carry  # type: ignore[return-value]
