"""
models/action_painters.py — Base diffusion-model painters for action-sequence
data (PushT / BlockPush / ToolHang), wrapping the backbones in
action_backbones.py.

Structurally these mirror UNetPainter/DiTPainter in painter_base.py (same
PainterBase/BaseModel training/eval/optimizer machinery), but differ in the
particulars that come with raw action-sequence data instead of pixels:
  - no VAE (actions are trained directly, not through a latent space)
  - noise_shape is (horizon, action_dim), not a square (C, H, W)
  - conditioning is a single flat `global_cond` vector (FiLM / prefix-token,
    produced by a GlobalCondEncoderBase from condition_encoders.py) instead of
    the cross-attention `encoder_hidden_states` convention used elsewhere
  - predict_action() drives closed-loop rollouts for dp_eval_callbacks.py by
    reusing the existing SamplingPipeline denoising loop
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from tqdm.auto import tqdm

from configs.schemas import EvalConfig, PainterOptimConfig, TrainConfig
from datasets.data_sample import DataSample
from models.action_backbones import ControlPainterUNet1D
from models.base import BaseModel
from models.diffusion_utils import apply_noisy_swap
from models.interfaces import DiffusionPrediction, IPAdapterSteering, ThinkerSteering
from models.losses import LossBase, build_loss
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.painter_base import PainterBase
from models.painter_thinkers import ThinkerFrozenPainterBase
from models.sampling import SamplingPipeline
from models.utility_models import strip_compiled_prefix


class ClosedLoopActionMixin:
    """Adds predict_action() — the closed-loop rollout entry point
    models/dp_eval_callbacks.py needs — to any model exposing
    self.condition_encoder (with an optional .n_obs_steps attribute),
    self.sampling_pipeline, self.eval_cfg, and callable as self(sample) via
    SamplingPipeline.sample_one_batch.

    Mixed into both ActionPainterBase (plain action-sequence painters) and
    ActionThinkerFrozenPainterBase (TRM thinker + frozen action painter) —
    the rollout logic itself doesn't care which one "self" actually is, so
    it lives here once instead of being duplicated or living in either
    class specifically.
    """

    @torch.no_grad()
    def predict_action(self, obs_dict: dict, n_action_steps: Optional[int] = None) -> dict:
        """Closed-loop rollout entry point used by dp_eval_callbacks.py.

        obs_dict keys mirror DataSample fields ('spatial_conditions',
        'embedding_conditions'); values already carry a batch dimension.
        Returns {'action': (B, T, action_dim)} still in the model's
        normalized action space — the caller unnormalizes.

        The backbone predicts a full (horizon, action_dim) chunk, but the
        first (n_obs_steps - 1) steps of that chunk correspond to already-
        observed history, not future actions — matching upstream diffusion_policy
        (e.g. diffusion_unet_lowdim_policy.predict_action), we slice
        action_pred[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps] rather
        than returning the raw chunk from index 0. Executing from index 0
        instead silently offsets every closed-loop action by (n_obs_steps-1)
        steps — invisible to training loss (teacher-forced regression over
        the whole horizon doesn't care about this slicing), but enough to
        tank success rate on contact-rich/precision tasks.
        """
        device = next(self.parameters()).device
        conditions = DataSample(
            spatial_conditions=obs_dict.get('spatial_conditions'),
            embedding_conditions=obs_dict.get('embedding_conditions'),
        ).to(device)

        pipeline = self.sampling_pipeline
        if pipeline is None:
            pipeline = SamplingPipeline(num_inference_steps=self.eval_cfg.num_ddim_steps, batch_size=1)

        action_pred = pipeline.sample_one_batch(self, conditions, device)

        n_obs_steps = getattr(self.condition_encoder, 'n_obs_steps', 1) if self.condition_encoder is not None else 1
        start = n_obs_steps - 1
        end = start + n_action_steps if n_action_steps is not None else action_pred.shape[1]
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}


class ActionPainterBase(PainterBase, BaseModel, ClosedLoopActionMixin):
    """Shared implementation for ActionUNet1DPainter / ActionTransformerPainter.

    Subclasses only need to set ``self.backbone`` in ``__init__``; everything
    else (training loop, CFG dropout, optimizer, closed-loop predict_action
    via ClosedLoopActionMixin) is shared here.

    Args:
        backbone:          Hydra config for a models.action_backbones module
                            (ConditionalUnet1D or TransformerForDiffusion).
        scheduler:          diffusion noise scheduler (DDPMScheduler etc.).
        action_dim:         dimensionality of the action vector at each step.
        horizon:            number of timesteps in the generated action chunk.
        optim_cfg:          optimizer hyperparams.
        train_cfg:          training hyperparams.
        eval_cfg:           eval hyperparams.
        condition_encoder:  optional Hydra config for a GlobalCondEncoderBase
                            (e.g. LowdimObsConditionEncoder / ImageObsConditionEncoder).
        eval_callbacks:     optional list of Hydra configs for closed-loop
                            eval callbacks (e.g. dp_eval_callbacks.py).
    """

    def __init__(
        self,
        scheduler,
        action_dim: int,
        horizon: int,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        condition_encoder=None,
        eval_callbacks=None,
        sampling_pipeline=None,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.action_dim = action_dim
        self.horizon = horizon
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None
        self.condition_encoder: Optional[nn.Module] = (
            instantiate(condition_encoder) if condition_encoder is not None else None
        )
        self.eval_callbacks: list = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.loss_fn: LossBase = build_loss(train_cfg, scheduler)

    @property
    def condition_keys(self) -> list[str]:
        return self.condition_encoder.condition_keys if self.condition_encoder is not None else []

    @property
    def noise_shape(self) -> tuple:
        """(horizon, action_dim) — matches the backbones' (B, T, action_dim) convention."""
        return (self.horizon, self.action_dim)

    # ── Forward ─────────────────────────────────────────────────────────────

    def _global_cond(self, sample: DataSample) -> Optional[torch.Tensor]:
        if self.condition_encoder is None:
            return None
        args = [getattr(sample, k) for k in self.condition_encoder.condition_keys]
        return self.condition_encoder(*args, timesteps=sample.timesteps)

    def forward(
        self,
        sample: DataSample,
        steering: Optional[ThinkerSteering] = None,
    ) -> DiffusionPrediction:
        global_cond = self._global_cond(sample)
        kwargs: dict = {}
        if steering is not None:
            kwargs.update(steering.to_painter_kwargs())
        noise_pred = self.backbone(sample.x_noisy, sample.timesteps, global_cond=global_cond, **kwargs)
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    # ── Training helpers (identical structure to UNetPainter) ────────────────

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        actions = mb.images.to(device)
        bsz = actions.shape[0]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy = self.scheduler.add_noise(actions, noise, timesteps)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else actions

        noisy, target = apply_noisy_swap(
            images=actions,
            noisy=noisy,
            target=target,
            timesteps=timesteps,
            scheduler=self.scheduler,
            swap_cfg=self.train_cfg.noisy_swap,
        )

        kwargs: dict = {}
        for f in dataclasses.fields(DataSample):
            if f.name in ("x_noisy", "timesteps", "target", "enc_x_noisy"):
                continue
            val = getattr(mb, f.name)
            if val is not None:
                kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        kwargs.update({"x_noisy": noisy, "timesteps": timesteps, "target": target})
        return DataSample(**kwargs)

    def _apply_cfg_dropout(self, sample: DataSample, drop: torch.Tensor) -> DataSample:
        updates: dict = {}
        for k in self.condition_keys:
            val = getattr(sample, k, None)
            if val is not None:
                mask = drop.view((drop.shape[0],) + (1,) * (val.ndim - 1)).to(val.device)
                updates[k] = val * (~mask).to(val.dtype)
        return dataclasses.replace(sample, **updates) if updates else sample

    # ── Training / eval / optimizer (identical structure to UNetPainter) ────

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=0,
            betas=tuple(self.optim_cfg.betas),
            weight_decay=self.optim_cfg.weight_decay,
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

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size,
        global_step,
        **kwargs,
    ):
        K = len(micro_batches)
        device = accelerator.device
        loss_sums: dict[str, float] = {}

        for mb in micro_batches:
            sample = self._prepare_training_sample(mb, device)

            if self.train_cfg.cfg_prob > 0:
                drop = torch.rand(sample.x_noisy.shape[0], device=device) < self.train_cfg.cfg_prob
                sample = self._apply_cfg_dropout(sample, drop)

            result = self(sample)
            total_loss, components = self.loss_fn(result.pred, result.logits, sample)
            accelerator.backward(total_loss / (global_batch_size * K))
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        accelerator.clip_grad_norm_(self.parameters(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)

        return {k: v / K for k, v in loss_sums.items()}, lr, global_step + 1

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()
        loss_sums: dict[str, float] = {}
        n_batches = 0
        dl_iter = iter(dataloader)
        for i in tqdm(range(max_batches), "Eval"):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            sample = self._prepare_training_sample(batch, accelerator.device)
            result = self(sample)
            _, components = self.loss_fn(result.pred, result.logits, sample)
            for k, v in components.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            n_batches += 1
        del dl_iter
        self.train()
        metrics = {k: v / n_batches for k, v in loss_sums.items()} if n_batches else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        return metrics

    def compile_submodules(self):
        self.backbone = torch.compile(self.backbone, fullgraph=False)
        if self.condition_encoder is not None:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)


class ActionUNet1DPainter(ActionPainterBase):
    """Base painter wrapping ConditionalUnet1D (FiLM-conditioned conv1d U-Net)."""

    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone: nn.Module = instantiate(backbone)


class ActionTransformerPainter(ActionPainterBase):
    """Base painter wrapping TransformerForDiffusion (GPT-style causal decoder)."""

    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone: nn.Module = instantiate(backbone)


# ── Frozen action-sequence painters for TRM thinker steering ─────────────────


class ControlNetSteeredActionUNet1DPainter(ActionUNet1DPainter):
    """Frozen ActionUNet1DPainter loaded from a checkpoint for ControlNet-style
    thinker steering — the action-sequence analog of
    models.painter_base.ControlNetSteeredUNetPainter.

    Builds the same architecture as the training run (same Hydra config), loads
    weights from a checkpoint produced by train_trm.py, and freezes the backbone
    and condition_encoder. The backbone is converted to ControlPainterUNet1D so
    that steering residuals (from models.translators.ControlNetTranslator1D)
    can be passed directly as kwargs — no forward() override needed here since
    ActionPainterBase.forward already unpacks steering.to_painter_kwargs()
    generically.

    Args:
        checkpoint: path to a checkpoint_*.pt file saved by train_trm.py
                    (must contain a "model_state" key).
        **kwargs:   forwarded verbatim to ActionUNet1DPainter.__init__.
    """

    def __init__(self, checkpoint: str, **kwargs):
        super().__init__(**kwargs)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=True)
        self.backbone.__class__ = ControlPainterUNet1D
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.condition_encoder is not None:
            for p in self.condition_encoder.parameters():
                p.requires_grad_(False)


# ── IP-Adapter frozen action-transformer wrapper ─────────────────────────────


class _IPAdapterEncoderLayer(nn.Module):
    """Wraps a single frozen nn.TransformerEncoderLayer with trainable,
    index-matched IP conditioning — the action-sequence analog of
    models.painter_base._DirectIPAdapterDiTBlock.

    ip_hidden_states must already be aligned 1:1 with the action-token
    positions (same length, same order as the action horizon) — token i's
    projected value is added directly to action-token i's output, not
    attended to via a separate Q/K softmax lookup (the same guarantee
    ControlNet residuals get from being added at matching positions).

    Zero-init on to_out_ip means no effect at the start of training — the
    model begins identical to the frozen painter.
    """

    def __init__(self, layer: nn.TransformerEncoderLayer, hidden_dim: int) -> None:
        super().__init__()
        self.layer = layer
        self.to_v_ip = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out_ip = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.to_out_ip.weight)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[torch.Tensor] = None,
        n_cond: int = 0,
    ) -> torch.Tensor:
        out = self.layer(src, src_mask=src_mask)
        if ip_hidden_states is not None:
            act = out[:, n_cond:]
            if ip_hidden_states.shape[1] != act.shape[1]:
                raise ValueError(
                    "IPAdapterSteeredActionTransformerPainter requires ip_hidden_states "
                    f"to be aligned 1:1 with the action token sequence (got "
                    f"{ip_hidden_states.shape[1]} tokens vs {act.shape[1]} action steps)."
                )
            delta = self.to_out_ip(self.to_v_ip(ip_hidden_states))
            out = torch.cat([out[:, :n_cond], act + delta], dim=1)
        return out


class IPAdapterSteeredActionTransformerPainter(ActionTransformerPainter):
    """Frozen ActionTransformerPainter with trainable, index-matched IP
    conditioning — the action-sequence analog of
    models.painter_base.DirectIPAdapterSteeredDiTPainter.

    nn.TransformerEncoder.forward() has no hook for threading extra per-layer
    kwargs, so this class's forward() reimplements TransformerForDiffusion's
    small token-assembly + causal-mask logic inline (rather than delegating
    to self.backbone(...) as a black box) and loops over the wrapped layers
    directly, passing ip_hidden_states explicitly each call. This keeps
    models/action_backbones.py untouched and avoids any hidden/stateful
    kwarg passing.

    Args:
        checkpoint: path to a checkpoint_*.pt file saved by train_trm.py
                    (must contain a "model_state" key).
        **kwargs:   forwarded verbatim to ActionTransformerPainter.__init__.
    """

    def __init__(self, checkpoint: str, **kwargs) -> None:
        super().__init__(**kwargs)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.condition_encoder is not None:
            for p in self.condition_encoder.parameters():
                p.requires_grad_(False)
        self.backbone.decoder.layers = nn.ModuleList([
            _IPAdapterEncoderLayer(layer, self.backbone.n_emb) for layer in self.backbone.decoder.layers
        ])

    def forward(self, sample: DataSample, steering: Optional[ThinkerSteering] = None) -> DiffusionPrediction:
        global_cond = self._global_cond(sample)
        bb = self.backbone
        x_noisy, timestep = sample.x_noisy, sample.timesteps
        B, T, _ = x_noisy.shape
        device = x_noisy.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.expand(B)
        time_token = bb.time_emb(bb.time_proj(timestep).to(dtype=x_noisy.dtype)).unsqueeze(1)  # (B, 1, n_emb)

        cond_tokens = [time_token]
        if bb.has_cond:
            if global_cond is None:
                raise ValueError("This backbone was built with cond_dim > 0.")
            cond_tokens.append(bb.cond_obs_emb(global_cond))  # (B, n_obs_steps, n_emb)
        cond = torch.cat(cond_tokens, dim=1) + bb.cond_pos_emb
        n_cond = cond.shape[1]

        act = bb.input_emb(x_noisy) + bb.pos_emb[:, :T]
        x = bb.drop(torch.cat([cond, act], dim=1))

        mask = bb._causal_mask(n_cond, T, device) if bb.causal_attn else None

        ip_hidden_states = None
        if steering is not None and isinstance(steering, IPAdapterSteering):
            ip_hidden_states = steering.ip_hidden_states

        for layer in bb.decoder.layers:
            x = layer(x, src_mask=mask, ip_hidden_states=ip_hidden_states, n_cond=n_cond)

        x = bb.ln_f(x)
        noise_pred = bb.head(x[:, n_cond:])  # drop cond/time positions, keep action tokens

        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )


# ── Thinker base with closed-loop predict_action ──────────────────────────────


class ActionThinkerFrozenPainterBase(ClosedLoopActionMixin, ThinkerFrozenPainterBase):
    """TRM thinker + frozen action-diffusion painter.

    Combines ThinkerFrozenPainterBase (thinker/translator/painter wiring,
    training, eval_step — unchanged, shared with the sudoku/CLEVR thinkers)
    with ClosedLoopActionMixin (predict_action, the closed-loop rollout entry
    point models/dp_eval_callbacks.py needs). n_obs_steps is read from the
    thinker's own condition_encoder (LowdimObsTRMConditionEncoder /
    ImageObsTRMConditionEncoder from condition_encoders.py), not the frozen
    painter's — the thinker and the frozen painter each have their own,
    separate condition encoders.
    """