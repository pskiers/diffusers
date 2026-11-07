"""
models/sampling.py — SamplingPipeline and NoisePredictor hierarchy.

SamplingPipeline owns the DDIM/DDPM denoising loop and uses the model's own
scheduler (model.scheduler) for inference.  Guidance is pluggable via
NoisePredictor subclasses:

    DirectPredictor              — single model call, no guidance
    CFGPredictor(scale)          — classifier-free guidance (2 calls per step)
    NoisyGuidancePredictor(...)  — noisy-image guidance wrapping any predictor

These compose: NoisyGuidancePredictor(CFGPredictor(...)) gives CFG + noisy-image
guidance at 4 model calls per step.

Schedule callables (for NoisyGuidancePredictor.schedule):
    ConstantSchedule(scale)              — constant weight at every step
    LinearSchedule(noisy_end, clean_end) — linear interpolation t≈T → t≈0

The model must expose:
    model(sample: DataSample) -> DiffusionPrediction       (forward)
    model.null_condition_sample(sample: DataSample) -> DataSample
    model.scheduler  (DDPMScheduler / DDIMScheduler with .config and .set_timesteps)
    model.noise_shape  (C, H, W) tuple, no batch dim

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
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from datasets.data_sample import DataSample, collate_data_samples
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
            model:  exposes forward(DataSample) -> DiffusionPrediction
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


class SteeringCFGPredictor(NoisePredictor):
    """Classifier-free guidance over the thinker's steering only.

    pred = pred_unsteered + scale * (pred_steered - pred_unsteered)

    Unlike CFGPredictor (which zeros the *painter's own* conditioning too,
    via null_condition_sample), this contrasts "with thinker steering" vs
    "frozen painter's own conditioning untouched, zero steering" — calling
    model(sample, null_steering=True) instead. See
    ThinkerFrozenPainterBase.forward in models/painter_thinkers.py.

    Requires the model to support forward(sample, null_steering=True) —
    i.e. a ThinkerFrozenPainterBase (or subclass) model.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        pred_steered = model(sample).pred
        pred_unsteered = model(sample, null_steering=True).pred
        return pred_unsteered + self.scale * (pred_steered - pred_unsteered)


class SteeringReusePredictor(NoisePredictor):
    """PaTh with reasoning stopped after the first `reason_frac` of the denoising steps (paper Fig. 20, "reuse"):
    on those steps it is the plain (CFG) predictor; on every later step the thinker is not run and the painter is
    run on the last reasoning step's logits (`model.run_painter`, both CFG branches), as experiments/
    ablate_trm_loop_budget.py's reuse axis. reason_frac=1.0 is identical to DirectPredictor / CFGPredictor(scale).

    Needs a ThinkerFrozenPainterBase-style model (forward returns the thinker logits, run_painter(sample, logits))
    that reasons from a fresh state at every step (model.forward; no carry across steps).
    """

    def __init__(self, reason_frac: float, scale: float = 1.0):
        assert 0.0 < reason_frac <= 1.0, reason_frac
        self.reason_frac = reason_frac
        self.scale = scale
        self._logits: dict[str, Tensor] = {}

    def n_reason_steps(self, num_steps: int) -> int:
        return max(1, round(self.reason_frac * num_steps))

    def predict(self, model, sample: DataSample, t: int, T: int) -> Tensor:
        timesteps = model.scheduler.timesteps
        step_idx = int((timesteps == t).nonzero()[0])
        if step_idx == 0:
            self._logits = {}
        reason = step_idx < self.n_reason_steps(len(timesteps))

        def branch(s: DataSample, key: str) -> Tensor:
            if reason:
                p = model(s)
                self._logits[key] = p.logits
                return p.pred
            return model.run_painter(s, self._logits[key])

        pred_c = branch(sample, "cond")
        if self.scale == 1.0:
            return pred_c
        pred_u = branch(model.null_condition_sample(sample), "uncond")
        return pred_u + self.scale * (pred_c - pred_u)


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


@dataclass
class ConstantSchedule:
    """Constant guidance weight at every step."""
    scale: float = 1.0

    def __call__(self, t: int, T: int) -> float:
        return self.scale


@dataclass
class LinearSchedule:
    """Linear schedule from noisy_end (t≈T) to clean_end (t≈0).

    Equivalent to the _lin(a, b) helper in eval_noisy_guidance.py:
        _lin(a, b)  ↔  LinearSchedule(noisy_end=a+b, clean_end=a)
    """
    noisy_end: float = 0.0
    clean_end: float = 1.0

    def __call__(self, t: int, T: int) -> float:
        return self.noisy_end + (self.clean_end - self.noisy_end) * (1.0 - t / T)


# ── Trajectory capture ────────────────────────────────────────────────────────


@dataclass
class TrajectoryRecorder:
    """Opt-in capture of intermediate denoising states during sampling (from origin/amaze/interpretability).

    steps:      inference-step indices (0-based, in scheduler order) to record;
                None records every step.
    capture_xt: also store the noisy latent x_t (prev_sample), not just the
                running x0 estimate (pred_original_sample).
    records:    filled during sampling — one dict per captured step with keys
                {"step", "t", "x0_pred"[, "x_t"]}; tensors are (B, C, H, W) on CPU.
    """

    steps: Optional[set] = None
    capture_xt: bool = False
    records: list = field(default_factory=list)

    def wants(self, i: int) -> bool:
        return self.steps is None or i in self.steps


# ── Correctors (predictor-corrector sampling) ─────────────────────────────────


def _alpha_bar(scheduler, t: int, device: torch.device) -> Tensor:
    """ᾱ_t of the training schedule; 1.0 below t=0 (DDPMScheduler.step's own convention for the final step)."""
    return scheduler.alphas_cumprod[t].to(device) if t >= 0 else torch.tensor(1.0, device=device)


def _pred_to_x0(scheduler, pred: Tensor, x: Tensor, t: int) -> Tensor:
    """The model output (prediction_type sample or epsilon) as an x0 prediction."""
    kind = scheduler.config.prediction_type
    if kind == "sample":
        return pred
    assert kind == "epsilon", f"corrector supports prediction_type sample/epsilon, got {kind}"
    a = _alpha_bar(scheduler, t, x.device)
    return (x - (1 - a).sqrt() * pred) / a.sqrt()


class Corrector(ABC):
    """K extra model evaluations per predictor step (predictor-corrector sampling). `step` replaces one predictor
    step t -> previous timestep; `predict(x, t)` returns the (guided) model output at x, t."""

    def __init__(self, n_steps: int):
        assert n_steps >= 1, n_steps
        self.n_steps = n_steps

    @abstractmethod
    def step(self, predict: Callable[[Tensor, Tensor], Tensor], scheduler, x: Tensor, t: Tensor) -> Tensor:
        """One predictor step from x_t plus n_steps corrector steps; returns x at the previous timestep."""


class LangevinCorrector(Corrector):
    """Song et al. 2021's LangevinCorrector for the VP SDE (score_sde `sampling.py`), run n_steps times at t before
    the predictor step, as in their PC sampler:

        grad = score(x, t);  noise ~ N(0, I)
        step = 2 * alpha_t * (snr * mean_b ||noise_b|| / mean_b ||grad_b||)^2       (alpha_t = 1 - beta_t)
        x <- x + step * grad + sqrt(2 * step) * noise

    with the score from the x0 prediction: score = -(x - sqrt(ᾱ_t) x0) / (1 - ᾱ_t).
    """

    def __init__(self, n_steps: int, snr: float):
        super().__init__(n_steps)
        self.snr = snr

    def step(self, predict, scheduler, x: Tensor, t: Tensor) -> Tensor:
        ti = int(t.item())
        a_bar = _alpha_bar(scheduler, ti, x.device)
        alpha = scheduler.alphas[ti].to(x.device)
        B = x.shape[0]
        for _ in range(self.n_steps):
            x0 = _pred_to_x0(scheduler, predict(x, t), x, ti)
            grad = -(x - a_bar.sqrt() * x0) / (1 - a_bar)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(B, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x = x + step_size * grad + torch.sqrt(step_size * 2) * noise
        return scheduler.step(predict(x, t), t, x).prev_sample


class RestartCorrector(Corrector):
    """Per-step Restart (Xu et al. 2023) / RePaint resampling: after the predictor step t -> t', re-noise x_t' back to
    t with the forward process, x_t = sqrt(ᾱ_t / ᾱ_t') x_t' + sqrt(1 - ᾱ_t / ᾱ_t') z, and redo the step; n_steps
    times."""

    def step(self, predict, scheduler, x: Tensor, t: Tensor) -> Tensor:
        ti = int(t.item())
        t_prev = int(scheduler.previous_timestep(ti))
        ratio = _alpha_bar(scheduler, ti, x.device) / _alpha_bar(scheduler, t_prev, x.device)
        x_prev = scheduler.step(predict(x, t), t, x).prev_sample
        for _ in range(self.n_steps):
            x = ratio.sqrt() * x_prev + (1 - ratio).sqrt() * torch.randn_like(x_prev)
            x_prev = scheduler.step(predict(x, t), t, x).prev_sample
        return x_prev


# ── SamplingPipeline ──────────────────────────────────────────────────────────


class SamplingPipeline:
    """Owns the DDIM/DDPM denoising loop using the model's own scheduler.

    The pipeline is model-agnostic: it only calls
        model(DataSample) -> DiffusionPrediction
        model.null_condition_sample(DataSample) -> DataSample
        model.scheduler  (set_timesteps + step)
        model.noise_shape

    Two entry points:
        sample_one_batch — for pre-batched DataSamples (e.g. from a dataloader)
        generate         — for a list of DataSamples; batches them internally

    Example — DDIM with CFG:
        pipeline = SamplingPipeline(num_inference_steps=20, batch_size=8,
                                    predictor=CFGPredictor(scale=2.0))
        images = pipeline.sample_one_batch(model, conditions, device)

    Example — CFG + noisy-image guidance:
        pipeline = SamplingPipeline(
            num_inference_steps=20, batch_size=8,
            predictor=NoisyGuidancePredictor(
                inner=CFGPredictor(scale=2.0),
                schedule=LinearSchedule(noisy_end=0.0, clean_end=1.0),
            ),
        )
    """

    def __init__(
        self,
        num_inference_steps: int = 20,
        batch_size: int = 8,
        predictor: Optional[NoisePredictor] = None,
        corrector: Optional[Corrector] = None,
    ):
        self.num_inference_steps = num_inference_steps
        self.batch_size = batch_size
        self.predictor = predictor if predictor is not None else DirectPredictor()
        self.corrector = corrector  # None: plain predictor steps (default)

    @property
    def cfg_scale(self) -> float:
        """CFG scale if predictor is (or wraps) a CFGPredictor/SteeringCFGPredictor, else 1.0."""
        p = self.predictor
        if isinstance(p, NoisyGuidancePredictor):
            p = p.inner
        return p.scale if isinstance(p, (CFGPredictor, SteeringCFGPredictor)) else 1.0

    @torch.no_grad()
    def sample_one_batch(
        self,
        model,
        conditions: DataSample,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[Tensor] = None,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> Tensor:
        """Denoise one pre-batched DataSample using the model's scheduler.

        Args:
            model:      model with scheduler, noise_shape, and forward
            conditions: DataSample with static condition fields; must already
                        have a batch dimension.  x_noisy and timesteps are
                        set per step by this method.
            device:     target device.
            generator:  optional RNG for reproducible initial noise.
            initial_noise: optional (B, *noise_shape) starting x, already at
                        the scheduler's noise scale — e.g. per-sample seeded
                        noise that must not depend on how samples are batched.
                        Overrides `generator`.
            recorder:   optional TrajectoryRecorder; when given, intermediate
                        denoising states are appended to recorder.records
                        (not with a corrector).

        Returns:
            (B, C, H, W) denoised output tensor on device.
        """
        B = _batch_size_of(conditions)
        shape = (B, *model.noise_shape)
        T = model.scheduler.config.num_train_timesteps

        model.scheduler.set_timesteps(self.num_inference_steps, device=device)
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
            # init_noise_sigma: 1.0 on diffusers' DDPM/DDIM schedulers; 2.0 for
            # MiniT2IFlowScheduler (noise ~ N(0, 4I)).
            x = torch.randn(shape, device=device, generator=generator) * getattr(model.scheduler, "init_noise_sigma", 1.0)

        def predict(x_t: Tensor, t: Tensor) -> Tensor:
            step_sample = dataclasses.replace(conditions, x_noisy=x_t, timesteps=t.expand(B).to(device))
            return self.predictor.predict(model, step_sample, int(t.item()), T)

        assert recorder is None or self.corrector is None, "TrajectoryRecorder is not supported with a corrector"
        for i, t in enumerate(model.scheduler.timesteps):
            if self.corrector is None:
                step_out = model.scheduler.step(predict(x, t), t, x)
                x = step_out.prev_sample
                if recorder is not None and recorder.wants(i):
                    x0_pred = getattr(step_out, "pred_original_sample", None)
                    recorder.records.append({
                        "step": i,
                        "t": int(t.item()),
                        "x0_pred": (x0_pred if x0_pred is not None else x).detach().cpu(),
                        **({"x_t": x.detach().cpu()} if recorder.capture_xt else {}),
                    })
            else:
                x = self.corrector.step(predict, model.scheduler, x, t)

        return x

    @torch.no_grad()
    def sample_best_of_n(
        self,
        model,
        conditions: DataSample,
        device: torch.device,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample `n` independent candidates per instance (different noise
        seeds), batched together in a single denoising pass.

        Every condition field is repeated `n` times along the batch dim
        (interleaved: instance 0's `n` candidates are contiguous, then
        instance 1's, ...) and run through one `sample_one_batch` call of
        size B*n — cheaper than n separate calls since the scheduler/model
        overhead is shared, and torch.randn already draws independent noise
        per row of the B*n batch.

        Args:
            conditions: DataSample with batch dim B.
            n:          candidates per instance.

        Returns:
            (B, n, C, H, W) tensor.
        """
        B = _batch_size_of(conditions)
        repeated = _repeat_interleave_sample(conditions, n)
        flat = self.sample_one_batch(model, repeated, device, generator=generator)
        return flat.view(B, n, *flat.shape[1:])

    @torch.no_grad()
    def generate(
        self,
        model,
        conditions: list[DataSample],
        device: torch.device,
    ) -> Tensor:
        """Generate images for a list of DataSamples, batching internally.

        Args:
            model:      model with scheduler, noise_shape, and forward
            conditions: list of single-sample DataSamples (no batch dim);
                        batched internally in chunks of self.batch_size.
            device:     target device.

        Returns:
            (N, C, H, W) denoised output tensor where N = len(conditions).
        """
        results = []
        for start in range(0, len(conditions), self.batch_size):
            chunk = conditions[start : start + self.batch_size]
            batch = collate_data_samples(chunk).to(device)
            results.append(self.sample_one_batch(model, batch, device))
        return torch.cat(results, dim=0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _repeat_interleave_sample(sample: DataSample, n: int) -> DataSample:
    """Repeat every set tensor field of `sample` n times along dim 0,
    interleaved so each instance's n copies stay contiguous:
    [inst0, inst0, ..., inst0, inst1, inst1, ...] (n copies each)."""
    updates = {}
    for f in dataclasses.fields(sample):
        val = getattr(sample, f.name)
        if val is not None and isinstance(val, Tensor):
            updates[f.name] = val.repeat_interleave(n, dim=0)
    return dataclasses.replace(sample, **updates)


def _batch_size_of(sample: DataSample) -> int:
    for f in dataclasses.fields(sample):
        val = getattr(sample, f.name)
        if val is not None and isinstance(val, Tensor):
            return val.shape[0]
    raise ValueError("Cannot determine batch size: all DataSample fields are None.")
