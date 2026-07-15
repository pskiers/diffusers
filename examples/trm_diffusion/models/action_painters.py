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
from models.base import BaseModel
from models.diffusion_utils import apply_noisy_swap
from models.interfaces import DiffusionPrediction, ThinkerSteering
from models.losses import LossBase, build_loss
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.painter_base import PainterBase
from models.sampling import SamplingPipeline


class ActionPainterBase(PainterBase, BaseModel):
    """Shared implementation for ActionUNet1DPainter / ActionTransformerPainter.

    Subclasses only need to set ``self.backbone`` in ``__init__``; everything
    else (training loop, CFG dropout, optimizer, closed-loop predict_action)
    is shared here.

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
        noise_pred = self.backbone(sample.x_noisy, sample.timesteps, global_cond=global_cond)
        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.scheduler.config.prediction_type,
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> dict:
        """Closed-loop rollout entry point used by dp_eval_callbacks.py.

        obs_dict keys mirror DataSample fields ('spatial_conditions',
        'embedding_conditions'); values already carry a batch dimension.
        Returns {'action': (B, horizon, action_dim)} still in the model's
        normalized action space — the caller unnormalizes.
        """
        device = next(self.parameters()).device
        conditions = DataSample(
            spatial_conditions=obs_dict.get('spatial_conditions'),
            embedding_conditions=obs_dict.get('embedding_conditions'),
        ).to(device)

        pipeline = self.sampling_pipeline
        if pipeline is None:
            pipeline = SamplingPipeline(num_inference_steps=self.eval_cfg.num_ddim_steps, batch_size=1)

        action = pipeline.sample_one_batch(self, conditions, device)
        return {'action': action}

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