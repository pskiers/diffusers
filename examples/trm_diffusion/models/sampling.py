"""
models/sampling.py — SamplingPipeline and NoisePredictor hierarchy.

SamplingPipeline owns the DDIM/DDPM denoising loop and builds its scheduler
from a SchedulerConfig.  Guidance is pluggable via NoisePredictor subclasses:

    DirectPredictor              — single model call, no guidance
    CFGPredictor(scale)          — classifier-free guidance (2 calls per step)
    NoisyGuidancePredictor(...)  — noisy-image guidance wrapping any predictor

These compose: NoisyGuidancePredictor(CFGPredictor(...)) gives CFG + noisy-image
guidance at 4 model calls per step.

The model must expose:
    model(sample: DataSample) -> DiffusionPrediction       (forward)
    model.null_condition_sample(sample: DataSample) -> DataSample

V0 / V1 noisy-guidance separation
----------------------------------
DataSample.x_noisy is always the painter's denoising target (actual x_t).
DataSample.enc_x_noisy is an optional override for what the *encoder* sees as
its noisy conditioning input.  When enc_x_noisy is None the encoder falls back
to x_noisy.  NoisyGuidancePredictor sets enc_x_noisy=zeros for the V0-equivalent
pass so the painter is unaffected.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from datasets.data_sample import DataSample
from models.interfaces import DiffusionPrediction

# ── NoisePredictor hierarchy ──────────────────────────────────────────────────


class NoisePredictor(ABC):
    """Strategy for producing a noise prediction at one denoising step.

    All guidance blending lives here; the SamplingPipeline only calls this
    interface and handles the scheduler loop.
    """

    @abstractmethod
    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        """
        Args:
            model:  exposes predict_noise(DataSample) -> DiffusionPrediction
                    and null_condition_sample(DataSample) -> DataSample
            sample: DataSample with x_noisy and timesteps already set for step t
            t:      current integer timestep value
            T:      total number of training timesteps

        Returns:
            (B, C, H, W) noise (or x0 / v) prediction tensor
        """


class DirectPredictor(NoisePredictor):
    """Single model call — no guidance."""

    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        return model(sample).pred


class CFGPredictor(NoisePredictor):
    """Classifier-free guidance.

    pred = pred_uncond + scale * (pred_cond - pred_uncond)

    The model provides null_condition_sample() to zero conditions for the
    unconditional pass.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        pred_cond = model(sample).pred
        pred_uncond = model(model.null_condition_sample(sample)).pred
        return pred_uncond + self.scale * (pred_cond - pred_uncond)


class NoisyGuidancePredictor(NoisePredictor):
    """Noisy-image guidance wrapping any inner predictor.

    At each step the inner predictor is called twice:
      1. Normal pass — encoder sees actual x_t via x_noisy (standard V1).
      2. Clean  pass — encoder sees zeros via enc_x_noisy (V0-equivalent),
                       while x_noisy (painter input) remains x_t.

    The results are blended by schedule(t, T):

        pred = pred_v1 + s * (pred_v0 - pred_v1)

    s=0 → pure V1 baseline, s=1 → pure V0, s>1 → extrapolation beyond V0.

    Wrapping a CFGPredictor yields 4 model calls per step.

    schedule convention:
        t decreases from T-1 to 0 during denoising.
        schedule(t=T-1, T) is the noisy-end value.
        schedule(t=0,   T) is the clean-end value.
    """

    def __init__(self, inner: NoisePredictor, schedule: Callable[[int, int], float]):
        self.inner = inner
        self.schedule = schedule

    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        pred_v1 = self.inner.predict(model, sample, t, T)

        # Override only what the encoder sees; x_noisy (painter input) unchanged.
        clean_sample = dataclasses.replace(sample, enc_x_noisy=torch.zeros_like(sample.x_noisy))
        pred_v0 = self.inner.predict(model, clean_sample, t, T)

        s = self.schedule(t, T)
        return pred_v1 + s * (pred_v0 - pred_v1)


# ── Built-in guidance schedules ───────────────────────────────────────────────


def constant_schedule(scale: float) -> Callable[[int, int], float]:
    """Constant guidance weight at every step."""
    return lambda t, T: scale


def linear_schedule(noisy_end: float, clean_end: float) -> Callable[[int, int], float]:
    """Linear schedule from noisy_end (t≈T) to clean_end (t≈0).

    Equivalent to the _lin(a, b) helper in eval_noisy_guidance.py:
        _lin(a, b)  ↔  linear_schedule(noisy_end=a+b, clean_end=a)
    """
    return lambda t, T: noisy_end + (clean_end - noisy_end) * (1.0 - t / T)


# ── Scheduler config ──────────────────────────────────────────────────────────


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    num_inference_steps: int = 50
    sampler: str = "ddim"


# ── SamplingPipeline ──────────────────────────────────────────────────────────


class SamplingPipeline:
    """Owns the DDIM/DDPM denoising loop and builds its scheduler from config.

    The pipeline is model-agnostic: it only calls
        model.predict_noise(DataSample) -> DiffusionPrediction
        model.null_condition_sample(DataSample) -> DataSample

    Example — DDIM with CFG:
        pipeline = SamplingPipeline(SchedulerConfig(sampler="ddim", ...))
        predictor = CFGPredictor(scale=2.0)
        images = pipeline.sample(model, sample, predictor, shape=(B,C,H,W), device=device)

    Example — DDIM with CFG + noisy-image guidance:
        predictor = NoisyGuidancePredictor(
            inner=CFGPredictor(scale=2.0),
            schedule=linear_schedule(noisy_end=0.0, clean_end=1.0),
        )
        images = pipeline.sample(model, sample, predictor, shape=..., device=...)
    """

    def __init__(self, scheduler_cfg: SchedulerConfig):
        self.cfg = scheduler_cfg
        self._scheduler = self._build_scheduler()

    def _build_scheduler(self):
        from diffusers import DDIMScheduler, DDPMScheduler

        kwargs = dict(
            num_train_timesteps=self.cfg.num_train_timesteps,
            beta_schedule=self.cfg.beta_schedule,
            prediction_type=self.cfg.prediction_type,
        )
        if self.cfg.sampler == "ddim":
            return DDIMScheduler(**kwargs)
        elif self.cfg.sampler == "ddpm":
            return DDPMScheduler(**kwargs)
        else:
            raise ValueError(f"Unknown sampler {self.cfg.sampler!r}. Choose 'ddim' or 'ddpm'.")

    @torch.no_grad()
    def sample(
        self,
        model,
        sample: DataSample,
        predictor: NoisePredictor,
        shape: tuple,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Run the full denoising loop.

        Args:
            model:               model with predict_noise / null_condition_sample
            sample:              DataSample with static condition fields pre-filled;
                                 x_noisy and timesteps are overwritten each step.
            predictor:           NoisePredictor controlling guidance strategy.
            shape:               (B, C, H, W) shape of the tensor to generate.
            device:              target device.
            num_inference_steps: overrides cfg.num_inference_steps for this call.
            generator:           optional RNG for reproducible initial noise.

        Returns:
            (B, C, H, W) denoised output tensor on device.
        """
        steps = num_inference_steps or self.cfg.num_inference_steps
        self._scheduler.set_timesteps(steps, device=device)
        T = self.cfg.num_train_timesteps

        x = torch.randn(shape, device=device, generator=generator)

        for t in self._scheduler.timesteps:
            t_batch = t.expand(shape[0]).to(device)
            step_sample = dataclasses.replace(sample, x_noisy=x, timesteps=t_batch)

            noise_pred = predictor.predict(model, step_sample, int(t.item()), T)
            x = self._scheduler.step(noise_pred, t, x).prev_sample

        return x
