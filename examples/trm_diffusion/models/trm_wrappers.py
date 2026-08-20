from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from adam_atan2 import AdamATan2
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from hydra.utils import instantiate
from typing import Any, Optional

from configs.schemas import ThinkerOptimConfig
from datasets.data_sample import DataSample
from datasets.sudoku_dataset import IGNORE_LABEL_ID
from models.base import BaseModel
from models.condition_encoders import _build_spatial_enc
from models.interfaces import DiffusionPrediction
from models.losses import build_loss
from models.painter_base import PainterBase
from models.utility_models import TimestepMLP
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
            emb = self.embed_scale * embedding
        else:
            emb = super()._input_embeddings(input, puzzle_identifiers)
        # Optional addend injected by SpatialTRM.reasoning_step (e.g. timestep conditioning).
        # Uses the same set-then-reset pattern as _keep_carry_grad.
        emb_bias = getattr(self, "_emb_bias", None)
        if emb_bias is not None:
            emb = emb + emb_bias.to(emb.dtype)
        return emb


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
        with_timestep_emb: bool = False,
        with_halt_head: bool = False,
        halt_head_lr: float = 1e-3,
    ):
        super().__init__()
        self.n_sup = n_sup
        self.vocab_size = vocab_size
        self.freeze_weights = freeze_weights
        self.with_timestep_emb = with_timestep_emb
        self.with_halt_head = with_halt_head
        self.optim_cfg = optim_cfg

        if with_timestep_emb:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=256)
            self.thinker_temb_proj = nn.Linear(256, hidden_size)
            nn.init.zeros_(self.thinker_temb_proj.weight)
            nn.init.zeros_(self.thinker_temb_proj.bias)

        if with_halt_head:
            # Predicts, from z_H at reasoning step t, the expected future loss
            # reduction still available from continuing (see train_halt_head).
            # Zero-init: starts out predicting "no expected improvement" (a
            # neutral halt-early prior), mirroring the vendored q_head's own
            # init trick. Trained by its own optimizer (below), decoupled from
            # the main thinker optimizer/LR schedule/global_step — the aux
            # loss is only computable once per full n_sup trajectory, not
            # once per n_sup iteration like everything else.
            self.halt_head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.halt_head.weight)
            nn.init.zeros_(self.halt_head.bias)
            self.halt_optimizer = torch.optim.Adam(self.halt_head.parameters(), lr=halt_head_lr)

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

        # halt_head has its own dedicated optimizer (self.halt_optimizer, set
        # up in __init__) — exclude it from the main thinker optimizer.
        if self.with_halt_head:
            for p in self.halt_head.parameters():
                exclude_ids.add(id(p))

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
        input_emb_bias: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        One supervision step. Internally runs H_cycles-1 no-grad cycles then one full-grad cycle.

        H_cycles / L_cycles: override the config values for this call only.
        keep_carry_grad: if True, z_H/z_L in the returned carry are NOT detached,
          so gradients flow back to `inputs` (used for classifier guidance in eval).
          Leave False during training to avoid accumulating graph across n_sup steps.
        input_emb_bias: optional (B, 1, hidden_size) or (B, N, hidden_size) tensor added
          to the input embeddings after embed_scale.
        timesteps: optional (B,) timestep tensor; if with_timestep_emb=True, projected
          into an embedding bias added to all input tokens.

        Returns: (logits, z_H, z_L) — z_H/z_L detached unless keep_carry_grad=True.
          logits: (B, seq_len, vocab_size) — gradients always attached.
        """
        if self.with_timestep_emb and timesteps is not None:
            t_bias = self.thinker_temb_proj(self.timestep_mlp(timesteps)).unsqueeze(1)
            input_emb_bias = (input_emb_bias + t_bias) if input_emb_bias is not None else t_bias

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
            self.inner._emb_bias = input_emb_bias
            new_carry, logits, _ = self.inner(carry, {"inputs": inputs, "puzzle_identifiers": puzzle_ids})
        finally:
            self.inner.config.H_cycles = orig_H
            self.inner.config.L_cycles = orig_L
            self.inner._keep_carry_grad = False
            self.inner._emb_bias = None

        return logits, new_carry.z_H, new_carry.z_L

    # ── Adaptive-halting head (continuous-space substitute for the vendored
    #    q_head, which needs a binary correctness reward this codebase's
    #    diffusion targets don't have) ─────────────────────────────────────

    def predict_halt_value(self, z_H: torch.Tensor) -> torch.Tensor:
        """Predicted expected future loss reduction from continuing past the
        reasoning step that produced this z_H (reads the same pooled token
        the vendored q_head uses). Near/below zero → diminishing returns,
        safe to halt; call sites decide the actual threshold.
        """
        if not self.with_halt_head:
            raise RuntimeError("predict_halt_value requires with_halt_head=True at construction.")
        return self.halt_head(z_H[:, 0].float()).squeeze(-1)

    def train_halt_head(self, z_H0_list: list[torch.Tensor], loss_list: list[torch.Tensor]) -> float:
        """Train the halt head on one micro-batch's full n_sup trajectory.

        z_H0_list / loss_list: length-n_sup lists of (B, hidden_size) / (B,)
        tensors, one per reasoning step, already detached — z_H0_list[t] is
        z_H[:, 0] right after step t, loss_list[t] is that step's per-sample
        diffusion loss (both computed by the caller, e.g.
        ThinkerFrozenPainterBase.train_step).

        Target for step t is the future loss reduction still available:
        loss_t - min(loss_{t+1 .. n_sup-1}). The last step is skipped (no
        future to compare against). Trained via its own optimizer
        (self.halt_optimizer) — does not touch the main thinker optimizer or
        global_step.
        """
        n_sup = len(loss_list)
        if n_sup < 2:
            return 0.0

        losses = torch.stack(loss_list, dim=0)  # (n_sup, B)
        # suffix_min[t] = min(losses[t:]) via cummin on the reversed sequence
        suffix_min = torch.flip(torch.cummin(torch.flip(losses, dims=[0]), dim=0).values, dims=[0])
        future_min = suffix_min[1:]  # future_min[t] = min(losses[t+1:]), t = 0..n_sup-2
        targets = (losses[:-1] - future_min).reshape(-1)  # ((n_sup-1)*B,)

        # z_H0_list entries come straight from the TRM's forward_dtype (bfloat16
        # in every real config) — halt_head is a plain fp32 Linear, same as
        # predict_halt_value's own cast.
        z_H0_stack = torch.stack(z_H0_list[:-1], dim=0).float()  # (n_sup-1, B, hidden_size)
        preds = self.halt_head(z_H0_stack.reshape(-1, z_H0_stack.shape[-1])).squeeze(-1)

        loss = F.mse_loss(preds, targets)
        self.halt_optimizer.zero_grad()
        loss.backward()
        self.halt_optimizer.step()
        return loss.item()

    @staticmethod
    def _future_min_targets(loss_seq: torch.Tensor) -> torch.Tensor:
        """loss_seq: (L, ...) → (L-1, ...) targets loss_t - min(loss_{t+1:}),
        t = 0..L-2. Same formula train_halt_head uses inline; factored out
        here only for train_halt_head_ragged's per-sample use (train_halt_head
        itself is left untouched)."""
        suffix_min = torch.flip(torch.cummin(torch.flip(loss_seq, dims=[0]), dim=0).values, dims=[0])
        return loss_seq[:-1] - suffix_min[1:]

    def train_halt_head_ragged(self, histories: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Like train_halt_head, but for a set of INDEPENDENT per-sample
        trajectories of possibly different lengths — used by
        ThinkerFrozenPainterACT's persistent-carry training, where different
        samples halt (and so flush their own history) at different calls,
        rather than one shared n_sup-length trajectory for the whole batch.

        histories: list of (z_H0_seq, loss_seq) pairs, one per sample that
        just halted — (L, hidden_size) / (L,) tensors, already detached, for
        THAT sample's own trajectory since its last reset. Histories with
        L < 2 are skipped (no future to compare against, same as
        train_halt_head). All qualifying per-step (input, target) pairs are
        pooled into ONE batched MSE update via self.halt_optimizer.
        """
        inputs, targets = [], []
        for z_H0_seq, loss_seq in histories:
            if loss_seq.shape[0] < 2:
                continue
            targets.append(self._future_min_targets(loss_seq))
            inputs.append(z_H0_seq[:-1].float())

        if not inputs:
            return 0.0

        preds = self.halt_head(torch.cat(inputs, dim=0)).squeeze(-1)
        target = torch.cat(targets, dim=0)

        loss = F.mse_loss(preds, target)
        self.halt_optimizer.zero_grad()
        loss.backward()
        self.halt_optimizer.step()
        return loss.item()

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
        if self.with_timestep_emb:
            self.timestep_mlp = torch.compile(self.timestep_mlp, fullgraph=False)
            self.thinker_temb_proj = torch.compile(self.thinker_temb_proj, fullgraph=False)

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


class TRMDiffusionBackbone(PainterBase, SpatialTRM):
    """SpatialTRM repurposed as the FULL pixel-space diffusion backbone,
    instead of a reasoning module that steers a separate frozen painter.

    The noisy image (channel-concatenated with the puzzle condition image) is
    patch-embedded into seq_len tokens — one token per patch_size x patch_size
    patch, via the same CNN patchifier NoisySpatialConditionEncoder already
    uses (_build_spatial_enc). For MNIST-Sudoku, patch_size=cell_size=16
    happens to give exactly the 9x9=81 grid the TRM already uses for one
    token per sudoku cell, so no sequence-length/attention-cost problem
    needs solving: attention stays over 81 tokens regardless of the 144x144
    pixel resolution.

    Reinterprets the vocab_size-sized "logits" SpatialTRM already produces as
    flattened per-patch pixels (vocab_size = patch_size**2 * out_channels)
    and folds them back into the full image — no separate frozen UNet, no
    discrete classification head. Pure diffusion pixel-reconstruction loss
    only (train_cfg.sudoku_loss_weight must be 0; there is no thinker logits
    output to attach a CE loss to here).

    Trains with the same deep-supervision scheme as the thinker side of the
    two-stage pipeline: every one of the n_sup outer iterations gets its own
    full forward + backward + optimizer step (see train_step), all against
    the SAME (x_noisy, timesteps, target) sampled once per micro-batch —
    only the recurrent carry (z_H, z_L) evolves across the n_sup loop.
    """

    condition_keys: list[str] = ["spatial_conditions"]
    has_realsolution_eval: bool = False

    def __init__(
        self,
        optim_cfg: ThinkerOptimConfig,
        scheduler,
        train_cfg,
        eval_cfg,
        patch_size: int = 16,
        out_channels: int = 1,
        cond_channels: int = 1,
        enc_channels: int = 128,
        enc_hidden_channels: tuple = (128, 256, 256),
        image_size: int = 144,
        eval_callbacks=None,
        sampling_pipeline=None,
        **trm_kwargs,
    ):
        patch_pixels = patch_size * patch_size * out_channels
        grid = image_size // patch_size
        SpatialTRM.__init__(
            self,
            optim_cfg=optim_cfg,
            vocab_size=patch_pixels,
            seq_len=grid * grid,
            with_timestep_emb=True,
            **trm_kwargs,
        )
        self.scheduler = scheduler
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.grid = grid

        in_channels = out_channels + cond_channels
        self.patch_enc, self.patch_proj = _build_spatial_enc(
            in_channels, enc_channels, list(enc_hidden_channels), self.inner.config.hidden_size, patch_size
        )
        self.loss_fn = build_loss(train_cfg, scheduler)
        self.eval_callbacks: list = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None

    @property
    def noise_shape(self) -> tuple:
        return (self.out_channels, self.image_size, self.image_size)

    def decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.clamp(0.0, 1.0)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        return images.clamp(0.0, 1.0)

    # ── Patchify / unpatchify ────────────────────────────────────────────────

    def _tokens_from_image(self, x_noisy: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([x_noisy, cond], dim=1) if cond is not None else x_noisy
        feat = self.patch_proj(self.patch_enc(x))
        return feat.flatten(2).transpose(1, 2)  # (B, seq_len, hidden_size)

    def _image_from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.shape[0]
        p, g, c = self.patch_size, self.grid, self.out_channels
        patches = patches.view(B, g, g, c, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, g, p, g, p)
        return patches.reshape(B, c, g * p, g * p)

    # ── Forward / sampling ───────────────────────────────────────────────────

    def forward(self, sample: DataSample, **kwargs) -> DiffusionPrediction:
        """Full inference: fresh carry, n_sup reasoning steps, fold back to
        an image. No guidance is applied here (matches ThinkerFrozenPainterBase.forward);
        use CFGPredictor from models.sampling during SamplingPipeline.sample()."""
        bsz = sample.x_noisy.shape[0]
        tokens = self._tokens_from_image(sample.x_noisy, sample.spatial_conditions)
        z_H, z_L = self.get_initial_states(bsz)
        z_H, z_L = z_H.to(sample.x_noisy.device), z_L.to(sample.x_noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.reasoning_step(tokens, z_H, z_L, timesteps=sample.timesteps)
        noise_pred = self._image_from_patches(logits.float())

        return DiffusionPrediction(pred=noise_pred, pred_type=self.scheduler.config.prediction_type)

    # ── Training ─────────────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        images = mb.images.to(device)
        bsz = images.shape[0]
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy = self.scheduler.add_noise(images, noise, timesteps)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else images
        cond = mb.spatial_conditions.to(device) if mb.spatial_conditions is not None else None
        return DataSample(
            images=images,
            spatial_conditions=cond,
            x_noisy=noisy,
            timesteps=timesteps,
            target=target,
            solution=mb.solution.to(device) if mb.solution is not None else None,
            solution_mask=mb.solution_mask.to(device) if mb.solution_mask is not None else None,
        )

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size: int,
        global_step: int,
        **kwargs,
    ) -> tuple[dict, float, int]:
        from models.optim_utils import apply_lr_and_step

        K = len(micro_batches)
        device = accelerator.device

        mb_data = []
        for mb in micro_batches:
            sample = self._prepare_training_sample(mb, device)
            bsz = sample.x_noisy.shape[0]
            z_H, z_L = self.get_initial_states(bsz)
            mb_data.append({"sample": sample, "z_H": z_H.to(device), "z_L": z_L.to(device)})

        total_losses: dict[str, float] = {}
        lr = 0.0
        for _ in range(self.n_sup):
            for d in mb_data:
                tokens = self._tokens_from_image(d["sample"].x_noisy, d["sample"].spatial_conditions)
                logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    tokens, d["z_H"], d["z_L"], timesteps=d["sample"].timesteps
                )
                noise_pred = self._image_from_patches(logits)
                step_loss, loss_dict = self.loss_fn(noise_pred, None, d["sample"])
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                if step_loss.requires_grad:
                    accelerator.backward(step_loss / (global_batch_size * K))

            accelerator.clip_grad_norm_(self.parameters(), 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            global_step += 1
            if ema is not None:
                ema.update(self)

        n = self.n_sup * K
        losses = {k: v / n for k, v in total_losses.items()}
        return losses, lr, global_step

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator: Any, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)

        self.train()
        metric_accum: dict[str, float] = {}
        n_batches = 0
        for i, batch in tqdm(enumerate(dataloader), desc="Eval", total=max_batches):
            if i >= max_batches:
                break
            sample = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            _, loss_dict = self.loss_fn(result.pred, None, sample)
            for k, v in loss_dict.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v
            n_batches += 1

        result = {k: v / n_batches for k, v in metric_accum.items()} if n_batches > 0 else {}

        self.eval()
        for cb in self.eval_callbacks:
            result.update(cb(self, dataloader, accelerator, **kwargs))
        self.train()
        return result
