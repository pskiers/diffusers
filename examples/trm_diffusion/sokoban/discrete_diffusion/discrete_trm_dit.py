"""
Discrete TRM Diffusion — core model for masked discrete diffusion on Sokoban boards.

Small number of diffusion steps (T=50-200).
Each step runs a TRM reasoning loop with dynamic ACT halting.
The carry (y, z) can be persisted across diffusion timesteps (use_carry_recycling flag).

The model predicts the clean board x_0 from a partially-masked input x_t using cross-entropy loss over masked positions only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

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
        / (half-1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)    # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


@dataclass
class DiscreteTRMCarry:
    z: torch.Tensor  # [B, seq_len, D]
    y: torch.Tensor  # [B, seq_len, D]


class AdaLNTRMBlock(nn.Module):
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

        self.adaLN_modulation = nn.Sequential(  # adaLN-Zero, outputs (γ₁, β₁, α₁, γ₂, β₂, α₂)
            nn.SiLU(),
            CastedLinear(hidden_size, 6 * hidden_size, bias=True),
        )
        linear = self.adaLN_modulation[1]   # zero init for training stability
        nn.init.zeros_(linear.weight)       # type: ignore [CastedLinear]
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)     # type: ignore [CastedLinear]

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
        t_emb: torch.Tensor,          # [B, D]
        cos_sin: CosSin,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [B, 1, 6*D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # adaLN + Attention layer
        h_norm = rms_norm(hidden_states, self.rms_norm_eps) * (1.0 + gamma1) + beta1
        h_attn = self.self_attn(cos_sin=cos_sin, hidden_states=h_norm)
        hidden_states = hidden_states + alpha1 * h_attn

        # adaLN + FC layer
        h_norm = rms_norm(hidden_states, self.rms_norm_eps) * (1.0 + gamma2) + beta2
        h_ffn = self.ffn(h_norm)
        hidden_states = hidden_states + alpha2 * h_ffn

        return hidden_states


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
        hidden_states: torch.Tensor,    # [B, L, D]
        input_injection: torch.Tensor,  # [B, L, D]
        t_emb: torch.Tensor,            # [B, D]
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for block in self.blocks:
            hidden_states = block(hidden_states, t_emb, cos_sin)
        return hidden_states


class DiscreteTRMDiffusion(nn.Module):
    """Discrete TRM denoiser for masked diffusion on Sokoban boards.
    Attributes:
        timestep:              [B]              in {1,...,T}
        x_t:                   [B, seq_length]  in {0,...,vocab_size} embedded to [B, seq_len, D]
        carry (z_t, y_t):      [B, seq_len, D]  in real number, embedding
    Returns:
        logits:    [B, 144, vocab_size]
        q_logits:  tuple[q_halt [B], q_continue [B]] , both float32
        new_carry: DiscreteTRMCarry (detached)
    """
    def __init__(
        self,
        vocab_size: int = 8,
        seq_len: int = 144,
        hidden_size: int = 512,
        num_heads: int = 8,
        layers: int = 2,
        n: int = 6,
        T: int = 3,
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
        self.layers = layers
        self.T = T
        self.n = n
        self.forward_dtype = getattr(torch, forward_dtype)
        self.rms_norm_eps = rms_norm_eps

        # Embedding
        embed_init_std = 1.0 / math.sqrt(hidden_size)
        self.embed_tokens = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Positional encoding
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

        # Time
        self.t_mlp = nn.Sequential(
            CastedLinear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size, hidden_size, bias=True),
        )

        # Transformer network
        self.transformer = AdaLNReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            n_layers=layers,
            rms_norm_eps=rms_norm_eps,
        )

        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q_head = CastedLinear(hidden_size, 2, bias=True)

        self.norm_z = nn.LayerNorm(hidden_size, dtype=self.forward_dtype) # prevent hidden state from exploading
        self.norm_y = nn.LayerNorm(hidden_size, dtype=self.forward_dtype)
        self.final_norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=self.forward_dtype))

        # Layers initialization
        _h_init = torch.zeros(hidden_size, dtype=self.forward_dtype)
        _l_init = torch.zeros(hidden_size, dtype=self.forward_dtype)
        self.register_buffer("H_init", _h_init, persistent=True)
        self.register_buffer("L_init", _l_init, persistent=True)
        self.H_init: torch.Tensor
        self.L_init: torch.Tensor

        with torch.no_grad():   # start from q_logits = -5, q_halt = sigmoid(-5) = 0.005
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5)

    def empty_carry(self, batch_size: int, device: torch.device) -> DiscreteTRMCarry:
        return DiscreteTRMCarry(
            y=self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1).clone(),
            z=self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1).clone(),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        carry: Optional[DiscreteTRMCarry] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], DiscreteTRMCarry]:
        B = x_t.shape[0]
        device = x_t.device

        token_emb = self.embed_tokens(x_t.to(torch.int32))   # [B, 144, D]
        if self.pos_encodings == "learned": # absolute positional encoding
            pos = self.embed_pos.embedding_weight.to(self.forward_dtype)
            x_emb = token_emb + pos
        else:
            x_emb = token_emb   # rope applied in attention

        t_emb_sin = sinusoidal_embedding(timestep, self.hidden_size).to(self.forward_dtype)
        t_emb = self.t_mlp(t_emb_sin)   # [B, D]

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None    # pre-calculate RoPE embedding for efficiency

        if carry is None:   # init with zeros if carry was not persisted from last iteration
            carry = self.empty_carry(B, device)
        y = carry.y.to(device)
        z = carry.z.to(device)

        # TRM reasoning loop: T-1 without grad, final with grad
        with torch.no_grad():
            for _ in range(self.T - 1):
                for _ in range(self.n):
                    z = self.norm_z(self.transformer(z, y + x_emb, t_emb, cos_sin))
                y = self.norm_y(self.transformer(y, z, t_emb, cos_sin))

        for _ in range(self.n):
            z = self.norm_z(self.transformer(z, y + x_emb, t_emb, cos_sin))
        y = self.norm_y(self.transformer(y, z, t_emb, cos_sin))

        # Output
        y_normed = rms_norm(y, self.rms_norm_eps) * self.final_norm_weight.to(y.dtype)
        logits = self.lm_head(y_normed)  # [B, 144, V]

        q_input = y_normed.mean(dim=1)   # [B, D]
        q_logits = self.q_head(q_input).to(torch.float32)  # [B, 2]

        new_carry = DiscreteTRMCarry(y=y.detach(), z=z.detach())
        return logits, (q_logits[..., 0], q_logits[..., 1]), new_carry


class ACTMaskedDiffusionWrapper(nn.Module):
    """ACT wrapper for Discrete TRM Diffusion. Runs the inner model up to halt_max_steps times per diffusion timestep.
    Loss = token loss (from last n_sup  step) + q_halt loss (avg accross all n_sup steps)

    Halted samples don't contribute to loss.
    Using warm-up: forcing los n_sup values at the begining to prevent making n_sup iteration from total noise.
    """
    def __init__(
        self,
        inner: DiscreteTRMDiffusion,
        class_weights: torch.Tensor,
        halt_max_steps: int = 8,
        halt_exploration_prob: float = 0.1,
        halt_loss_weight: float = 0.5,
        use_carry_recycling: bool = False,
        carry_recycle_prob: float = 0.5,
        halt_warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1
    ):
        super().__init__()
        self.inner = inner
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.halt_loss_weight = halt_loss_weight
        self.use_carry_recycling = use_carry_recycling
        self.carry_recycle_prob = carry_recycle_prob
        self.halt_warmup_steps = halt_warmup_steps * gradient_accumulation_steps

        self.register_buffer("_training_step", torch.tensor(0, dtype=torch.long))

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights_tensor)

    def get_current_max_halt(self) -> int:
        """Linearly ramp halt_max_steps from 1 to halt_max_steps over halt_warmup_steps."""
        step = self._training_step.item()   # type: ignore
        if self.halt_warmup_steps <= 0 or step >= self.halt_warmup_steps:
            return self.halt_max_steps
        frac = step / self.halt_warmup_steps
        return max(1, int(1 + frac * (self.halt_max_steps - 1)))

    def forward(
        self,
        x_t: torch.Tensor,              # [B, 144] masked tokens
        x_0: torch.Tensor,              # [B, 144] clean tokens (target)
        timestep: torch.Tensor,         # [B] diffusion timestep
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
        mask_counts = is_masked.sum(dim=1).clamp_min(1).float()  # [B]

        current_max_halt = self.get_current_max_halt()
        if self.training:
            self._training_step.add_(1)

        labels = x_0.clone()  # [B, 144]
        labels[~is_masked] = -100  # -100 label for unmasked, CE ignores them natively

        # CARRY
        if diffusion_carry is not None: # forward pass
            carry = diffusion_carry
        elif (                          # carry recycling
            self.training
            and self.use_carry_recycling
            and torch.rand(1, device=device).item() < self.carry_recycle_prob
        ):
            with torch.no_grad():
                _, _, cold_carry = self.inner(x_t, timestep, carry=None)
            carry = cold_carry
        else:                           # carry reseted in each trm loop
            carry = None

        # ACT INITIALIZATION
        steps = torch.zeros(B, dtype=torch.int32, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        halt_loss_accum = torch.tensor(0.0, device=device)
        lm_loss_accum = torch.tensor(0.0, device=device)
        total_active_steps = torch.tensor(0.0, device=device)

        final_logits = None
        final_q_halt = None

        for _ in range(current_max_halt):
            # FORWARD PASS THROUGH INNER NETWORK
            logits, (q_halt, _), new_carry = self.inner(x_t, timestep, carry)
            carry = DiscreteTRMCarry(y=new_carry.y, z=new_carry.z)

            with torch.no_grad():
                step_preds = logits.detach().argmax(dim=-1)
                step_correct_at_mask = (step_preds == x_0) | ~is_masked
                step_seq_correct = step_correct_at_mask.all(dim=1).float()

            active_mask = (~halted).float()
            total_active_steps += active_mask.sum()

            # TOKEN LOSS
            step_per_token_loss = F.cross_entropy(
                logits.float().reshape(-1, self.inner.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                weight=self.class_weights,
                reduction="none",
            ).reshape(B, self.inner.seq_len)
            step_lm_loss = step_per_token_loss.sum(dim=1) / mask_counts  # [B]
            lm_loss_accum += (step_lm_loss * active_mask).sum()

            # HALT LOSS
            step_halt_loss = F.binary_cross_entropy_with_logits(
                q_halt, step_seq_correct, reduction="none"
            )
            halt_loss_accum += (step_halt_loss * active_mask).sum()

            # MASK UPDATE
            steps = steps + (~halted).int()
            is_last = (steps >= current_max_halt)

            with torch.no_grad():
                halt_now = is_last.clone()
                if self.training and current_max_halt > 1:  # TRAINING
                    halt_now |= (q_halt > 0)
                    if self.halt_exploration_prob > 0:
                        explore = torch.rand(B, device=device) < self.halt_exploration_prob
                        min_steps = torch.randint(2, current_max_halt + 1, (B,), device=device)
                        halt_now &= ((steps >= min_steps) | ~explore)

                halt_now &= ~halted

            halt_now_expanded = halt_now.view(B, 1, 1)
            if final_logits is None:
                final_logits = logits
                final_q_halt = q_halt.detach()
            else:
                final_logits = torch.where(halt_now_expanded, logits, final_logits)
                final_q_halt = torch.where(halt_now, q_halt.detach(), final_q_halt)

            halted |= halt_now

            if halted.all():
                break

        with torch.no_grad():
            final_preds = final_logits.detach().argmax(dim=-1)
            n_correct = ((final_preds == x_0) & is_masked).float().sum().item()
            final_seq_correct = ((final_preds == x_0) | ~is_masked).all(dim=1).float()

        safe_steps = total_active_steps.clamp_min(1.0)
        halt_loss = halt_loss_accum / safe_steps
        lm_loss = lm_loss_accum / safe_steps

        # TOTAL LOSS
        total_loss = lm_loss + (self.halt_loss_weight * halt_loss)

        metrics: Dict[str, object] = {
            "lm_loss": lm_loss.detach().item(),
            "halt_loss": halt_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "token_accuracy": n_correct / max(mask_counts.sum().item(), 1.0),
            "exact_accuracy": final_seq_correct.mean().item(),
            "avg_halt_steps": steps.float().mean().item(),
            "halt_step_std": steps.float().std().item() if B > 1 else 0.0,
            "q_halt_mean": final_q_halt.detach().mean().item(),
            "q_halt_pos_frac": (final_q_halt.detach() > 0).float().mean().item(),
            "mask_ratio": mask_counts.sum().item() / max(B * self.inner.seq_len, 1),
        }

        return total_loss, metrics, carry  # type: ignore
