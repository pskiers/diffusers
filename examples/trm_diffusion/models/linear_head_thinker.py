"""Encoder + linear head: the no-reasoning control for the PaTh thinker.

Drop-in replacement for SpatialTRM (models/trm_wrappers.py) inside
ThinkerFrozenPainterBase (models/painter_thinkers.py): the same condition encoder,
translator and frozen painter are used, only the recursive transformer is removed.
Each token's logits are a linear function of that token's own encoder features —
no attention between tokens, no recurrence. The output layer is the same as the
TRM's lm_head (CastedLinear, no bias), applied to RMS-normalised features as the
TRM applies it to its post-norm z_H.

The model is stateless, but it keeps the TRM's n_sup: ThinkerFrozenPainterBase
takes one optimizer step per reasoning step (train_step), so the same n_sup gives
this control exactly as many optimizer updates as the TRM run it is compared with.
z_H carries the normalised features (what the head reads) so callers that record
z_H keep working; z_L is passed through unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from adam_atan2 import AdamATan2

from configs.schemas import ThinkerOptimConfig
from models.optim_utils import ScheduledOptimizer
from models.trm.layers import CastedLinear, rms_norm


class LinearHeadThinker(nn.Module):
    with_halt_head = False

    def __init__(
        self,
        optim_cfg: ThinkerOptimConfig,
        vocab_size: int = 11,
        seq_len: int = 144,
        hidden_size: int = 512,
        n_sup: int = 16,
        rms_norm_eps: float = 1e-5,
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 0,
    ):
        super().__init__()
        # Accepted only because experiment configs set them for the TRM; a puzzle
        # embedding would add positions the linear head has no way to handle.
        if puzzle_emb_ndim or puzzle_emb_len:
            raise ValueError("LinearHeadThinker does not support puzzle embeddings "
                             f"(got puzzle_emb_ndim={puzzle_emb_ndim}, puzzle_emb_len={puzzle_emb_len})")
        self.optim_cfg = optim_cfg
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_sup = n_sup
        self.rms_norm_eps = rms_norm_eps
        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_optimizers(self, world_size: int, num_steps) -> list[ScheduledOptimizer]:
        """Same optimizer and schedule as SpatialTRM's main thinker optimizer."""
        optim = AdamATan2(
            list(self.parameters()),
            lr=0,  # set per-step by the scheduler
            weight_decay=self.optim_cfg.weight_decay,
            betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
        )
        return [
            ScheduledOptimizer(
                optim,
                base_lr=self.optim_cfg.lr,
                warmup_steps=self.optim_cfg.warmup_steps,
                num_steps=num_steps,
                min_ratio=self.optim_cfg.lr_min_ratio,
            )
        ]

    def get_initial_states(self, bs: int):
        device = self.lm_head.weight.device
        z = torch.zeros(bs, self.seq_len, self.hidden_size, device=device)
        return z, z.clone()

    def reasoning_step(
        self,
        inputs: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        **_,
    ):
        """inputs: (B, seq_len, hidden_size) encoder features.

        Returns (logits, z_H, z_L) like SpatialTRM.reasoning_step:
        logits (B, seq_len, vocab_size) with gradients, z_H the detached normalised
        features, z_L unchanged. H_cycles / L_cycles / keep_carry_grad overrides
        are accepted and ignored — there is no loop to change.
        """
        features = rms_norm(inputs.float(), variance_epsilon=self.rms_norm_eps)
        logits = self.lm_head(features)
        return logits, features.detach(), z_L
