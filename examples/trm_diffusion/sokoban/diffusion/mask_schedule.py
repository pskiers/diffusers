"""
Absorbing-state mask schedule for discrete diffusion on token sequences.

Forward process: each token is independently replaced with [MASK] (ID=0)
at rate γ(t) = t/T (linear schedule).

Reverse process: at each step, a fraction of masked tokens is unmasked by
sampling from the model's predicted x_0 distribution.
"""

from __future__ import annotations

import torch
from typing import Optional


MASK_TOKEN_ID = 0


class AbsorbingMaskSchedule:
    """Linear absorbing-state masking schedule for discrete token sequences.

    Forward: q(x_t | x_0) — each position masked independently with prob γ(t).
    Reverse: p(x_{t-1} | x_t, x̂_0) — unmask fraction of masked tokens per step.
    """

    def __init__(self, num_steps: int = 100):
        self.num_steps = num_steps

    def mask_rate(self, t: torch.Tensor) -> torch.Tensor:
        """γ(t) = t / T. Input t: [B] int in {0, ..., T}. Returns [B] float."""
        return t.float() / self.num_steps

    # ── Forward process (training) ──────────────────────────────────────────

    def forward_mask(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply absorbing mask: x_0 [B, L] → x_t [B, L].

        Each token replaced with MASK_TOKEN_ID independently with probability γ(t).
        """
        gamma = self.mask_rate(t).unsqueeze(1)  # [B, 1]
        rand = torch.rand_like(x_0.float(), generator=generator)
        mask = rand < gamma  # [B, L] True → replace with MASK
        mask_val = torch.tensor(MASK_TOKEN_ID, dtype=x_0.dtype, device=x_0.device)
        x_t = torch.where(mask, mask_val, x_0)
        return x_t

    # ── Reverse process (sampling) ──────────────────────────────────────────

    def reverse_step(
        self,
        x_t: torch.Tensor,
        probs: torch.Tensor,
        t: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """One reverse step: x_t → x_{t-1}.

        For masked positions: unmask with probability (γ(t) - γ(t-1)) / γ(t),
        sampling the token from the predicted distribution `probs`.
        Unmasked positions are preserved.

        Args:
            x_t:   [B, L] current tokens (with MASKs)
            probs: [B, L, V] predicted p(x_0 | x_t) from model
            t:     current timestep (int, > 0)
        """
        if t <= 0:
            # Last step: unmask everything greedily
            sampled = torch.argmax(probs, dim=-1)
            is_masked = (x_t == MASK_TOKEN_ID)
            return torch.where(is_masked, sampled, x_t)

        gamma_t = t / self.num_steps
        gamma_t_minus_1 = (t - 1) / self.num_steps

        # Probability of unmasking a masked token at this step
        unmask_prob = (gamma_t - gamma_t_minus_1) / gamma_t

        is_masked = (x_t == MASK_TOKEN_ID)  # [B, L]

        # Decide which masked positions to unmask
        rand = torch.rand_like(x_t.float(), generator=generator)
        do_unmask = is_masked & (rand < unmask_prob)  # [B, L]

        # Sample tokens from predicted distribution
        B, L, V = probs.shape
        flat_probs = probs.reshape(B * L, V).clamp_min(1e-8)
        # torch.multinomial on CUDA with Generator can be unreliable;
        # sample on CPU and move back for reliability.
        sampled = torch.multinomial(flat_probs.cpu(), num_samples=1).to(probs.device)
        sampled = sampled.reshape(B, L)

        x_t_minus_1 = torch.where(do_unmask, sampled, x_t)
        return x_t_minus_1

    # ── Full sampling loop ──────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        carry=None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
    ):
        """Full reverse sampling: x_T → x_0.

        The model's carry persists across diffusion timesteps (Option III).

        Args:
            model: DiscreteTRMDiffusion instance
            batch_size: number of boards to generate
            seq_len: 144 for 12x12 Sokoban
            device: target device
            carry: optional initial carry (None → fresh)
            generator: optional RNG
            temperature: sampling temperature (applied to logits)

        Returns:
            x_0: [B, L] generated token sequence
            carry: final carry state
        """
        # Start fully masked
        x = torch.full((batch_size, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)

        for t in range(self.num_steps, 0, -1):
            timestep = torch.full((batch_size,), t, dtype=torch.long, device=device)

            logits, _, carry = model(x, timestep, carry=carry)

            # Block MASK token (class 0) from being sampled
            logits[:, :, 0] = -float("inf")

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits.float(), dim=-1)  # [B, L, V]
            x = self.reverse_step(x, probs, t, generator=generator)

        return x, carry
