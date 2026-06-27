"""
models/clevr_painter_thinkers.py — TRM thinker classes for a frozen CLEVR DiT painter.

Two conditioning variants (V0 and V1 each):

  CrossAttn  — single logit_to_cond linear + unfrozen caption_projection.
               All 12 DiT blocks see the same projected condition tokens via their
               existing (frozen) cross-attention weights.

  IPAdapter  — same CrossAttn baseline PLUS per-block trainable cross-attention
               modules (IP-Adapter style) that add residuals after each transformer
               block.  Each block independently learns to use the TRM tokens.

V0: TRM input = object embeddings only  (B, max_objects, hidden_size).
V1: TRM input = object embeddings  cat  noisy-latent tokens  (B, max_objects+G², hidden).

Both inherit from PainterThinkerV0Tok so the tested MNIST training loop
(_train_step_standard, _compute_step_loss, …) is reused unchanged.

Key override list vs base PainterThinkerV0Tok:
  reasoning_step  — adds _get_enc_emb encoding step; fixes CFG-dropout to 3-D mask.
  forward         — same pattern as OriginalTRMRatatouilleV0.forward.
  _logits_to_spatial — passthrough (DiT uses sequences, not spatial maps).
  _prep_mb_data   — VAE-encode images to latents; no solution/CE labels.
  eval_step       — CLEVR-specific: validation MSE + DDIM image samples.
  get_painter_params — returns [] (DiT is frozen).
  build_optimizers — thinker optims + one encoder optim for trainable non-thinker params.
  compile_submodules — only compiles thinker (frozen DiT skipped; hooks need it unfused).
"""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from tqdm.auto import tqdm

from configs.schemas import (
    ClevrDiTConfig,
    ClevrDiTOptimConfig,
    ClevrPainterThinkerConfig,
    EvalConfig,
    PainterThinkerConfig,
    ThinkerModelConfig,
    ThinkerOptimConfig,
    TrainConfig,
)
from models.clevr_painters import StandaloneClevrDiT
from models.condition_encoders import ConditionEncoderBase
from models.eval_callbacks import EvalCallbackBase, ImageGenEvalCallback
from models.optim_utils import DelayedScheduledOptimizer, ScheduledOptimizer, apply_lr_and_step
from models.painter_thinkers import PainterThinkerV0Tok

# ── Per-block cross-attention module (IP-Adapter) ────────────────────────────


class ClevrIPAdapterCrossAttn(nn.Module):
    """
    Single cross-attention block for IP-Adapter style per-block conditioning.

    Q: from DiT hidden state  (B, n_patches, query_dim)
    K,V: from TRM tokens      (B, seq_len,   kv_dim)
    Output: residual added to DiT hidden state after each transformer block.

    out_proj is zero-initialised so residuals start at zero and training begins
    from the frozen DiT baseline.
    """

    def __init__(self, query_dim: int, kv_dim: int, n_heads: int, head_dim: int):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(kv_dim, inner, bias=False)
        self.v_proj = nn.Linear(kv_dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, query_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        S = kv.shape[1]
        Q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


# ── Shared base class ─────────────────────────────────────────────────────────


class ThinkerWithFrozenClevrDiTBase(PainterThinkerV0Tok):
    """
    Base CLEVR thinker.  Creates SpatialTRM via PainterThinkerV0Tok, then
    replaces the dummy MNIST bridge/UNet with the frozen CLEVR DiT + VAE.

    Subclasses implement _run_painter (and register any conditioning modules).
    """

    has_realsolution_eval: bool = False  # no discrete-label solution eval for CLEVR

    def __init__(
        self,
        clevr_painter: StandaloneClevrDiT,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: ClevrPainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: ClevrDiTOptimConfig,
        scheduler,
        condition_encoder: ConditionEncoderBase,
        eval_callbacks=None,
    ):
        # PainterThinkerV0Tok needs a PainterThinkerConfig.  The bridge and UNet
        # it creates are tiny dummies that get replaced immediately below.
        _dummy_cfg = PainterThinkerConfig(
            painter_size=144,
            cell_size=16,
            bridge_channels=16,
            painter_channels=(32,),
            painter_layers_per_block=1,
            diff_thinker_weight=model_cfg.diff_thinker_weight,
            thinker_bridge_mode="logits",
            painter_dtype=model_cfg.painter_dtype,
        )
        super().__init__(
            thinker_cfg=thinker_cfg,
            model_cfg=_dummy_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
        )

        # Store optim cfg so build_optimizers can access it.
        self.painter_optim_cfg = painter_optim_cfg

        # ── Replace dummy bridge+UNet with frozen CLEVR DiT + VAE ────────────
        del self.bridge
        self.bridge = None

        # Frozen VAE — never trained
        self.vae = clevr_painter.vae
        self.scaling_factor: float = clevr_painter.scaling_factor
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Frozen DiT — subclasses may selectively unfreeze modules (e.g. caption_projection)
        self.painter = clevr_painter.dit
        for p in self.painter.parameters():
            p.requires_grad_(False)

        # Trainable condition encoder — must be provided; instantiated in factory via Hydra.
        self.condition_encoder: ConditionEncoderBase = condition_encoder

        # Cache DiT config for subclasses
        self._dit_cfg = clevr_painter.model_cfg

        # caption_projection delayed-unfreeze: lr stays 0 until this step.
        # Subclasses (CrossAttn, IPAdapter) call requires_grad_(True) on caption_projection;
        # the base build_optimizers puts those params in a DelayedScheduledOptimizer.
        self._caption_proj_freeze_steps: int = model_cfg.caption_proj_freeze_steps

        # _painter_dtype from PainterThinkerV0Tok is already set; expose a ctx helper.
        self._autocast_ctx = lambda device: (
            torch.autocast(device_type=device, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else contextlib.nullcontext()
        )

        self.eval_callbacks: list[EvalCallbackBase] = (
            list(eval_callbacks) if eval_callbacks is not None else [ImageGenEvalCallback()]
        )

    # ── Decode / noise-shape helpers (used by ImageGenEvalCallback) ───────────

    @property
    def noise_shape(self) -> tuple:
        """Shape of one noise sample (no batch dim). Used by ImageGenEvalCallback."""
        return (self._dit_cfg.latent_channels, self._dit_cfg.latent_size, self._dit_cfg.latent_size)

    def _decode_for_eval(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latents → pixel images in [0, 1]. Used by ImageGenEvalCallback."""
        return ((self.vae.decode(z / self.scaling_factor).sample + 1.0) / 2.0).clamp(0.0, 1.0)

    # ── Condition helpers ─────────────────────────────────────────────────────

    def _get_condition(self, mb: dict, device) -> torch.Tensor:
        key = self.condition_encoder.condition_keys[0]
        return mb[key].to(device)

    def _get_enc_emb(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.condition_encoder(condition, noisy, timesteps)

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """Passthrough — DiT uses token sequences, not spatial maps."""
        return logits  # (B, seq_len, vocab_size)

    # ── Reasoning / forward (mirrors OriginalTRMRatatouilleV0) ───────────────

    def reasoning_step(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy, timesteps)
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            enc_emb, z_H, z_L, puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles
        )
        spatial_cond = self._logits_to_spatial(logits.float())  # (B, seq_len, vocab_size)

        if self.diff_thinker_weight == 0.0:
            sc = spatial_cond.detach()
        elif self.diff_thinker_weight != 1.0:
            sc = self.diff_thinker_weight * spatial_cond + (1.0 - self.diff_thinker_weight) * spatial_cond.detach()
        else:
            sc = spatial_cond

        if self.training and self.train_cfg.cfg_prob > 0:
            # 3-D mask: (B, 1, 1) broadcasts over (B, seq_len, vocab_size)
            drop = torch.rand(sc.shape[0], 1, 1, device=sc.device) < self.train_cfg.cfg_prob
            sc = sc * (~drop)

        noise_pred = self._run_painter(noisy, sc, timesteps)
        return noise_pred, logits, z_H_next, z_L_next

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy)
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H, z_L = z_H.to(noisy.device), z_L.to(noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(enc_emb, z_H, z_L, puzzle_ids)

        spatial_cond = self._logits_to_spatial(logits.float())

        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred = pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits

    # ── Abstract — subclasses implement ──────────────────────────────────────

    def _run_painter(
        self,
        noisy_z: torch.Tensor,
        thinker_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # ── Data preparation ──────────────────────────────────────────────────────

    def _prep_mb_data(self, micro_batches, device):
        """Encode images to VAE latents; add diffusion noise in latent space."""
        mb_data = []
        for mb in micro_batches:
            images = mb["images"].to(device)  # (B, 3, H, W)  in [-1, 1]
            conditions = mb["conditions"].to(device)
            bsz = images.shape[0]

            with torch.no_grad():
                z = self.vae.encode(images).latent_dist.sample() * self.scaling_factor

            noise = torch.randn_like(z)
            timesteps_t = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_z = self.scheduler.add_noise(z, noise, timesteps_t)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z

            z_H, z_L = self.get_initial_states(bsz)
            mb_data.append(
                {
                    "images": images,
                    "condition": conditions,
                    "solution": None,
                    "ce_labels": None,
                    "puzzle_ids": None,
                    "noisy": noisy_z,
                    "timesteps": timesteps_t,
                    "target": target,
                    "z_H": z_H.to(device),
                    "z_L": z_L.to(device),
                }
            )
        return mb_data

    # ── Eval step ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 50)
        device = accelerator.device
        self.eval()

        # Validation MSE
        val_losses = []
        for i, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            if i >= max_batches:
                break
            images = batch["images"].to(device)
            conditions = batch["conditions"].to(device)
            bsz = images.shape[0]
            z = self.vae.encode(images).latent_dist.sample() * self.scaling_factor
            noise = torch.randn_like(z)
            ts = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_z = self.scheduler.add_noise(z, noise, ts)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else z
            noise_pred, _ = self(noisy_z, ts, conditions)
            val_losses.append(F.mse_loss(noise_pred.float(), target.float()).item())

        result = {"diff_loss": float(np.mean(val_losses))} if val_losses else {}
        for cb in self.eval_callbacks:
            result.update(cb(self, dataloader, accelerator, **kwargs))

        self.train()
        return result

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def get_painter_params(self) -> list:
        return []  # DiT is frozen; nothing for the painter optimizer

    def get_thinker_params(self) -> list:
        frozen_ids = {id(p) for p in self.painter.parameters() if not p.requires_grad}
        frozen_ids.update(id(p) for p in self.vae.parameters())
        return [p for p in self.parameters() if id(p) not in frozen_ids]

    def _get_encoder_params(self) -> list:
        """Trainable params not in SpatialTRM and not in caption_projection.

        caption_projection is handled separately in build_optimizers with a
        DelayedScheduledOptimizer so its training can be deferred.
        """
        thinker_ids = {id(p) for p in self.thinker.parameters()}
        caption_proj_ids = {id(p) for p in self.painter.caption_projection.parameters()}
        return [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in thinker_ids and id(p) not in caption_proj_ids
        ]

    def build_optimizers(self, world_size, num_steps):
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        optims = list(thinker_optims)

        encoder_params = self._get_encoder_params()
        if encoder_params:
            enc_optim = torch.optim.AdamW(encoder_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
            optims.append(
                ScheduledOptimizer(
                    enc_optim,
                    base_lr=self.painter_optim_cfg.lr,
                    warmup_steps=self.painter_optim_cfg.warmup_steps,
                    num_steps=num_steps,
                    min_ratio=self.painter_optim_cfg.lr_min_ratio,
                )
            )

        # caption_projection: if unfrozen (by CrossAttn / IPAdapter subclasses), give it
        # its own DelayedScheduledOptimizer.  Setting _caption_proj_freeze_steps to a
        # large value (e.g. 999999) effectively keeps it frozen for the whole run.
        cap_params = [p for p in self.painter.caption_projection.parameters() if p.requires_grad]
        if cap_params:
            cap_optim = torch.optim.AdamW(cap_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
            optims.append(
                DelayedScheduledOptimizer(
                    cap_optim,
                    base_lr=self.painter_optim_cfg.lr,
                    warmup_steps=self.painter_optim_cfg.warmup_steps,
                    num_steps=num_steps,
                    min_ratio=self.painter_optim_cfg.lr_min_ratio,
                    delay_steps=self._caption_proj_freeze_steps,
                )
            )

        return optims

    def compile_submodules(self):
        # Only compile the thinker inner module.
        # The frozen DiT is not compiled: forward hooks (IPAdapter) require an
        # un-fused graph, and compiling a frozen model has no training benefit.
        self.thinker.inner.L_level = torch.compile(self.thinker.inner.L_level, fullgraph=False)


# ── Option 3: single cross-attention  (V0) ───────────────────────────────────


class ThinkerWithFrozenClevrDiTV0CrossAttn(ThinkerWithFrozenClevrDiTBase):
    """
    V0 thinker with single cross-attention conditioning.

    TRM logits → logit_to_cond → encoder_hidden_states for ALL DiT blocks.
    caption_projection is unfrozen so it can adapt to the new token distribution.

    Trainable: SpatialTRM, condition_encoder, logit_to_cond, caption_projection.
    Frozen:    all other DiT weights, VAE.
    """

    def __init__(
        self,
        clevr_painter: StandaloneClevrDiT,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: ClevrPainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: ClevrDiTOptimConfig,
        scheduler,
        condition_encoder: ConditionEncoderBase,
    ):
        super().__init__(
            clevr_painter=clevr_painter,
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )
        cfg = clevr_painter.model_cfg

        # Single projection: TRM logits → conditioning tokens for DiT cross-attention.
        # Zero-bias init → null condition (zeros in) starts at exactly zero.
        self.logit_to_cond = nn.Linear(thinker_cfg.vocab_size, cfg.cond_embed_dim)
        nn.init.normal_(self.logit_to_cond.weight, std=0.02)
        nn.init.zeros_(self.logit_to_cond.bias)

        # Unfreeze caption_projection so it participates in the optimizer.
        # Training is delayed via DelayedScheduledOptimizer: lr stays 0 until
        # caption_proj_freeze_steps, then normal warmup+cosine begins.
        # Set caption_proj_freeze_steps=999999 to keep it effectively frozen.
        for p in self.painter.caption_projection.parameters():
            p.requires_grad_(True)

    def _run_painter(self, noisy_z, thinker_cond, timesteps):
        cond_tokens = self.logit_to_cond(thinker_cond.float())  # (B, seq_len, cond_embed_dim)
        with self._autocast_ctx(noisy_z.device.type):
            return self.painter(noisy_z, timestep=timesteps, encoder_hidden_states=cond_tokens).sample


# ── Option C: IP-Adapter per-block cross-attention  (V0) ─────────────────────


class ThinkerWithFrozenClevrDiTV0IPAdapter(ThinkerWithFrozenClevrDiTBase):
    """
    V0 thinker with IP-Adapter per-block cross-attention conditioning.

    CrossAttn baseline (logit_to_cond → encoder_hidden_states, unfrozen
    caption_projection) is active, AND each DiT transformer block gets an
    additional independent cross-attention module that attends to the TRM tokens
    and adds a residual to the block's hidden state.

    The per-block modules use a shared trm_to_ip projection for K,V (same tokens,
    different per-block attention weights) so the IP part has:
      num_layers x (Q_k, K_k, V_k, out_k)  +  1 x trm_to_ip

    All new modules are zero-init on their output projection so training starts
    exactly from the frozen DiT + CrossAttn baseline.

    Trainable: SpatialTRM, condition_encoder, logit_to_cond, caption_projection,
               trm_to_ip, ip_cross_attn (all blocks).
    Frozen:    all other DiT weights, VAE.
    """

    def __init__(
        self,
        clevr_painter: StandaloneClevrDiT,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: ClevrPainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: ClevrDiTOptimConfig,
        scheduler,
        condition_encoder: ConditionEncoderBase,
    ):
        super().__init__(
            clevr_painter=clevr_painter,
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )
        cfg = clevr_painter.model_cfg
        inner_dim = cfg.num_attention_heads * cfg.attention_head_dim

        # CrossAttn baseline (same as CrossAttn variant)
        self.logit_to_cond = nn.Linear(thinker_cfg.vocab_size, cfg.cond_embed_dim)
        nn.init.normal_(self.logit_to_cond.weight, std=0.02)
        nn.init.zeros_(self.logit_to_cond.bias)
        for p in self.painter.caption_projection.parameters():
            p.requires_grad_(True)

        # IP-Adapter: shared projection for cross-attention K,V
        self.trm_to_ip = nn.Linear(thinker_cfg.vocab_size, inner_dim)
        nn.init.normal_(self.trm_to_ip.weight, std=0.02)
        nn.init.zeros_(self.trm_to_ip.bias)

        # Per-block cross-attention modules
        self.ip_cross_attn = nn.ModuleList(
            [
                ClevrIPAdapterCrossAttn(
                    query_dim=inner_dim,
                    kv_dim=inner_dim,
                    n_heads=cfg.num_attention_heads,
                    head_dim=cfg.attention_head_dim,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # Scratch storage for KV tokens (set before each DiT forward, cleared after)
        self._ip_kv: Optional[torch.Tensor] = None
        self._register_ip_hooks()

    def _register_ip_hooks(self):
        self._ip_hook_handles = []
        for k, block in enumerate(self.painter.transformer_blocks):

            def make_hook(idx: int):
                def hook(module, inputs, output):
                    if self._ip_kv is None:
                        return output
                    h = output[0] if isinstance(output, tuple) else output
                    delta = self.ip_cross_attn[idx](h, self._ip_kv)
                    h = h + delta
                    return (h,) + output[1:] if isinstance(output, tuple) else h

                return hook

            self._ip_hook_handles.append(block.register_forward_hook(make_hook(k)))

    def _run_painter(self, noisy_z, thinker_cond, timesteps):
        cond_tokens = self.logit_to_cond(thinker_cond.float())
        self._ip_kv = self.trm_to_ip(thinker_cond.float())  # (B, seq_len, inner_dim)
        try:
            with self._autocast_ctx(noisy_z.device.type):
                result = self.painter(noisy_z, timestep=timesteps, encoder_hidden_states=cond_tokens).sample
        finally:
            self._ip_kv = None
        return result


# ── V1 variants (add noisy latent encoder) ───────────────────────────────────


class _ClevrV1Mixin:
    """
    Marker mixin for V1 variants (object + noisy-latent encoder).

    The encoder itself is fully specified by the condition_encoder config key
    (typically ClevrNoisyObjectFeatureEncoder).  This mixin exists for factory
    dispatch and documentation only — no code is added.
    """


class ThinkerWithFrozenClevrDiTV1CrossAttn(_ClevrV1Mixin, ThinkerWithFrozenClevrDiTV0CrossAttn):
    """
    V1 + CrossAttn: TRM sees object embeddings concatenated with noisy-latent tokens.

    Encoder type is specified via config (typically ClevrNoisyObjectFeatureEncoder).
    thinker_cfg.seq_len must equal max_objects + grid_size² of the latent encoder.
    """

    def __init__(
        self,
        clevr_painter: StandaloneClevrDiT,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: ClevrPainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: ClevrDiTOptimConfig,
        scheduler,
        condition_encoder: ConditionEncoderBase,
    ):
        super().__init__(
            clevr_painter=clevr_painter,
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )


class ThinkerWithFrozenClevrDiTV1IPAdapter(_ClevrV1Mixin, ThinkerWithFrozenClevrDiTV0IPAdapter):
    """
    V1 + IPAdapter: TRM sees object embeddings concatenated with noisy-latent tokens,
    and per-block IP-Adapter modules provide additional per-block conditioning.

    Encoder type is specified via config (typically ClevrNoisyObjectFeatureEncoder).
    thinker_cfg.seq_len must equal max_objects + grid_size² of the latent encoder.
    """

    def __init__(
        self,
        clevr_painter: StandaloneClevrDiT,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: ClevrPainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: ClevrDiTOptimConfig,
        scheduler,
        condition_encoder: ConditionEncoderBase,
    ):
        super().__init__(
            clevr_painter=clevr_painter,
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )
