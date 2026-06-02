from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from adam_atan2 import AdamATan2
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from typing import Any, Optional

from configs.schemas import ThinkerOptimConfig
from datasets.sudoku_dataset import IGNORE_LABEL_ID
from models.base import BaseModel
from models.trm.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from models.trm.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.optim_utils import ScheduledOptimizer


class SpatialTRMInner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """
    Drop-in replacement for TinyRecursiveReasoningModel_ACTV1_Inner that also
    accepts float embeddings as inputs.
    """

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if input.is_floating_point():
            # Pre-computed embedding: (B, seq_len, hidden_size)
            embedding = input.to(self.forward_dtype)
            if self.config.pos_encodings == "learned":
                embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
            return self.embed_scale * embedding
        return super()._input_embeddings(input, puzzle_identifiers)


class SpatialTRM(BaseModel):
    """
    Wrapper for SpatialTRMInner for training.
    """

    def __init__(
        self,
        optim_cfg: ThinkerOptimConfig,
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
        self.optim_cfg = optim_cfg

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
            H_layers=0,  # ignored by inner model
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
        self.inner = SpatialTRMInner(config)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_optimizers(self, world_size: int, num_steps) -> list[ScheduledOptimizer]:
        """
        Builds optimizers.
        """
        optims = []

        exclude_ids: set[int] = set()

        # puzzle emb optim
        if hasattr(self.inner, "puzzle_emb"):
            emb = self.inner.puzzle_emb
            emb_optim = CastedSparseEmbeddingSignSGD_Distributed(
                emb.buffers(),
                world_size=world_size,
                lr=0,  # set per-step by scheduler
                weight_decay=self.optim_cfg.puzzle_emb.weight_decay,
            )
            optims.append(
                ScheduledOptimizer(
                    emb_optim,
                    base_lr=self.optim_cfg.puzzle_emb.lr,
                    warmup_steps=self.optim_cfg.puzzle_emb.warmup_steps,
                    num_steps=num_steps,
                    min_ratio=self.optim_cfg.puzzle_emb.lr_min_ratio,
                )
            )
            for buf in (emb.local_weights, emb.local_ids, emb.weights):
                exclude_ids.add(id(buf))

        if not self.freeze_weights:
            optim = AdamATan2(
                [p for p in self.parameters() if id(p) not in exclude_ids],
                lr=0,  # set per-step by scheduler
                weight_decay=self.optim_cfg.weight_decay,
                betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
            )
            optims.append(
                ScheduledOptimizer(
                    optim,
                    base_lr=self.optim_cfg.lr,
                    warmup_steps=self.optim_cfg.warmup_steps,
                    num_steps=num_steps,
                    min_ratio=self.optim_cfg.lr_min_ratio,
                )
            )
        return optims

    def get_initial_states(self, bs: int):
        """
        Return (z_H, z_L) on the same device as the model, initialized from
        H_init / L_init buffers (1-D vectors broadcast to all sequence positions).
        """
        total_len = self.inner.config.seq_len + self.inner.puzzle_emb_len
        # H_init / L_init: (hidden_size,) → expand to (bs, total_len, hidden_size)
        z_H = self.inner.H_init.view(1, 1, -1).expand(bs, total_len, -1).clone()
        z_L = self.inner.L_init.view(1, 1, -1).expand(bs, total_len, -1).clone()
        return z_H, z_L

    def reasoning_step(
        self,
        inputs: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        keep_carry_grad: bool = False,
    ):
        """
        One supervision step. Internally runs H_cycles-1 no-grad cycles then one full-grad cycle.

        H_cycles / L_cycles: override the config values for this call only.
        keep_carry_grad: if True, z_H/z_L in the returned carry are NOT detached,
          so gradients flow back to `inputs` (used for classifier guidance in eval).
          Leave False during training to avoid accumulating graph across n_sup steps.

        Returns: (logits, z_H, z_L) — z_H/z_L detached unless keep_carry_grad=True.
          logits: (B, seq_len, vocab_size) — gradients always attached.
        """
        bs = inputs.shape[0]
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(bs, dtype=torch.int32, device=inputs.device)

        orig_H = self.inner.config.H_cycles
        orig_L = self.inner.config.L_cycles
        if H_cycles is not None:
            self.inner.config.H_cycles = H_cycles
        if L_cycles is not None:
            self.inner.config.L_cycles = L_cycles
        try:
            carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
            self.inner._keep_carry_grad = keep_carry_grad
            new_carry, logits, _ = self.inner(carry, {"inputs": inputs, "puzzle_identifiers": puzzle_ids})
        finally:
            self.inner.config.H_cycles = orig_H
            self.inner.config.L_cycles = orig_L
            self.inner._keep_carry_grad = False

        return logits, new_carry.z_H, new_carry.z_L

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        n_sup: Optional[int] = None,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full inference: n_sup * H_cycles * L_cycles uniformly (no grad split). Always runs max steps.
        """
        n_sup = n_sup or self.n_sup
        bs = inputs.shape[0]
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(bs, dtype=torch.int32, device=inputs.device)

        z_H, z_L = self.get_initial_states(bs)
        z_H = z_H.to(inputs.device)
        z_L = z_L.to(inputs.device)

        seq_info = {"cos_sin": self.inner.rotary_emb() if hasattr(self.inner, "rotary_emb") else None}
        input_emb = self.inner._input_embeddings(inputs, puzzle_ids)

        for _ in range(n_sup):
            for _ in range(self.inner.config.H_cycles):
                for _ in range(self.inner.config.L_cycles):
                    z_L = self.inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = self.inner.L_level(z_H, z_L, **seq_info)

        return self.inner.lm_head(z_H)[:, self.inner.puzzle_emb_len :]

    def _sudoku_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        preds = logits.argmax(-1)
        blank = labels != IGNORE_LABEL_ID
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1).clamp(min=0),
            ignore_index=IGNORE_LABEL_ID,
            reduction="mean",
        )
        cell_acc = (preds == labels)[blank].float().mean()
        correct = (preds == labels) & blank
        puzzle_acc = (correct.sum(-1) == blank.sum(-1)).float().mean()
        return {"sudoku_loss": loss, "cell_acc": cell_acc, "puzzle_acc": puzzle_acc}

    def train_step(
        self,
        micro_batches: list[dict],
        accelerator: Any,
        optimizers: list,
        ema: Any,
        global_batch_size: int,
        global_step: int,
        **kwargs,
    ) -> tuple[dict, float, int]:
        from models.optim_utils import apply_lr_and_step

        K = len(micro_batches)
        device = accelerator.device

        mb_data = []
        for mb in micro_batches:
            bsz = mb["inputs"].shape[0]
            z_H, z_L = self.get_initial_states(bsz)
            puzzle_ids = mb.get("puzzle_id")
            mb_data.append(
                {
                    "inputs": mb["inputs"].to(device),
                    "labels": mb["labels"].to(device),
                    "puzzle_ids": puzzle_ids.to(device) if puzzle_ids is not None else None,
                    "z_H": z_H.to(device),
                    "z_L": z_L.to(device),
                }
            )

        total_loss = 0.0
        lr = None
        for _ in range(self.n_sup):
            for d in mb_data:
                logits, d["z_H"], d["z_L"] = self.reasoning_step(d["inputs"], d["z_H"], d["z_L"], d["puzzle_ids"])
                step_loss = F.cross_entropy(
                    logits.float().view(-1, self.vocab_size),
                    d["labels"].view(-1).clamp(min=0),
                    ignore_index=IGNORE_LABEL_ID,
                )
                accelerator.backward(step_loss / (global_batch_size * K))
                total_loss += step_loss.item()

            accelerator.clip_grad_norm_(self.parameters(), 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            if ema is not None:
                ema.update(self)
            global_step += 1

        avg_loss = total_loss / (self.n_sup * K)
        return {"loss": avg_loss}, lr, global_step

    def compile_submodules(self):
        self.inner.L_level = torch.compile(self.inner.L_level, fullgraph=False)

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator: Any, **kwargs) -> dict:
        self.eval()
        max_batches = kwargs.get("max_batches", 10)
        accum: dict[str, list] = {"sudoku_loss": [], "cell_acc": [], "puzzle_acc": []}
        for i, batch in tqdm(enumerate(dataloader), desc="Evaluating", total=max_batches):
            if i >= max_batches:
                break
            inputs = batch["inputs"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            puzzle_ids = batch.get("puzzle_id")
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(accelerator.device)
            logits = self.predict(inputs, puzzle_ids=puzzle_ids)
            m = self._sudoku_metrics(logits.float(), labels)
            for k, v in m.items():
                accum[k].append(v.item())
        self.train()
        return {k: float(np.mean(v)) for k, v in accum.items()}
