# ABOUTME: Tests for the predictor-corrector option of SamplingPipeline (models/sampling.py): the Langevin and per-step
# ABOUTME: Restart correctors target the exact marginal with an exact model, and corrector=None is unchanged.
import dataclasses
import os
import sys
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_sample import DataSample  # noqa: E402
from models.interfaces import DiffusionPrediction  # noqa: E402
from models.sampling import LangevinCorrector, RestartCorrector, SamplingPipeline  # noqa: E402

SIGMA = 0.5  # data ~ N(0, SIGMA^2 I)


def _scheduler(prediction_type="sample"):
    s = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type=prediction_type,
                      clip_sample=False)
    s.set_timesteps(20)
    return s


def _marginal_var(s, t: int) -> float:
    """Var of p_t for N(0, SIGMA^2) data."""
    a = s.alphas_cumprod[t].item() if t >= 0 else 1.0
    return a * SIGMA**2 + 1 - a


def _exact_predict(s):
    """Exact model output for N(0, SIGMA^2) data: E[x0 | x_t] (or the matching epsilon)."""

    def predict(x, t):
        a = s.alphas_cumprod[int(t)]
        x0 = a.sqrt() * SIGMA**2 / (a * SIGMA**2 + 1 - a) * x
        return x0 if s.config.prediction_type == "sample" else (x - a.sqrt() * x0) / (1 - a).sqrt()

    return predict


class IdentityStep:
    """The scheduler with an identity predictor step, to test a corrector in isolation."""

    def __init__(self, s):
        self.s, self.alphas_cumprod, self.alphas, self.config = s, s.alphas_cumprod, s.alphas, s.config

    def previous_timestep(self, t):
        return self.s.previous_timestep(t)

    def step(self, pred, t, x):
        return SimpleNamespace(prev_sample=x)


def test_langevin_converges_to_the_marginal():
    torch.manual_seed(0)
    for kind in ("sample", "epsilon"):
        s = _scheduler(kind)
        t = s.timesteps[10]
        x = torch.randn(8192, 16)  # var 1, not the marginal's
        out = LangevinCorrector(n_steps=300, snr=0.16).step(_exact_predict(s), IdentityStep(s), x, t)
        target = _marginal_var(s, int(t))
        assert abs(out.var().item() - target) / target < 0.05, (kind, out.var().item(), target)


def test_langevin_follows_the_model():
    """Sanity check of the test: a model that assumes data variance 1 pulls the chain to that model's marginal
    (variance 1), away from the true one."""
    torch.manual_seed(0)
    s = _scheduler()
    t = s.timesteps[10]

    def wrong(x, tt):
        a = s.alphas_cumprod[int(tt)]
        return a.sqrt() * x  # E[x0 | x_t] for N(0, I) data

    out = LangevinCorrector(n_steps=300, snr=0.16).step(wrong, IdentityStep(s), torch.randn(8192, 16) * 0.5, t)
    assert abs(out.var().item() - 1.0) < 0.05 and abs(out.var().item() - _marginal_var(s, int(t))) > 0.3, out.var()


def test_restart_renoise_maps_marginal_t_prev_to_t():
    torch.manual_seed(0)
    s = _scheduler()
    for i in (0, 10, 19):  # 19: the last step, t' < 0 (clean)
        t = s.timesteps[i]
        t_prev = int(s.previous_timestep(int(t)))
        x_prev = torch.randn(8192, 16) * _marginal_var(s, t_prev) ** 0.5
        # identity predictor step: the output is one re-noising of x_prev from t' to t
        out = RestartCorrector(n_steps=1).step(_exact_predict(s), IdentityStep(s), x_prev, t)
        target = _marginal_var(s, int(t))
        assert abs(out.var().item() - target) / target < 0.03, (i, out.var().item(), target)


def test_no_corrector_matches_plain_loop():
    class Model:
        noise_shape = (16,)
        scheduler = _scheduler()

        def __call__(self, sample):
            return DiffusionPrediction(pred=_exact_predict(self.scheduler)(sample.x_noisy, sample.timesteps[0]),
                                       pred_type="sample")

        def null_condition_sample(self, sample):
            return sample

    model, conditions, x = Model(), DataSample(images=torch.zeros(64, 1)), torch.randn(64, 16)
    torch.manual_seed(1)
    out = SamplingPipeline(num_inference_steps=20, batch_size=64).sample_one_batch(
        model, conditions, torch.device("cpu"), initial_noise=x)
    torch.manual_seed(1)  # the loop before the corrector option existed
    model.scheduler.set_timesteps(20)
    ref = x.clone()
    for t in model.scheduler.timesteps:
        s = dataclasses.replace(conditions, x_noisy=ref, timesteps=t.expand(64))
        ref = model.scheduler.step(model(s).pred, t, ref).prev_sample
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


class StubPaTh:
    """Minimal PaTh-like model: the thinker 'logits' depend on x_t and the condition; the painter output on x_t and
    the logits. Counts thinker (forward) and painter-only (run_painter) calls."""

    noise_shape = (16,)

    def __init__(self):
        self.scheduler = _scheduler()
        self.calls = {"forward": 0, "run_painter": 0}

    def _paint(self, sample, logits):
        return 0.5 * sample.x_noisy + 0.1 * logits

    def __call__(self, sample):
        self.calls["forward"] += 1
        logits = torch.tanh(sample.x_noisy + sample.images)
        return DiffusionPrediction(pred=self._paint(sample, logits), pred_type="sample", logits=logits)

    def run_painter(self, sample, logits):
        self.calls["run_painter"] += 1
        return self._paint(sample, logits)

    def null_condition_sample(self, sample):
        return dataclasses.replace(sample, images=torch.zeros_like(sample.images))


def _run(predictor, recorder=None, seed=3):
    model = StubPaTh()
    conditions = DataSample(images=torch.randn(8, 16, generator=torch.Generator().manual_seed(0)))
    torch.manual_seed(seed)
    out = SamplingPipeline(num_inference_steps=20, batch_size=8, predictor=predictor).sample_one_batch(
        model, conditions, torch.device("cpu"), initial_noise=torch.randn(8, 16), recorder=recorder)
    return out, model.calls


def test_steering_reuse_full_budget_equals_plain_predictors():
    from models.sampling import CFGPredictor, DirectPredictor, SteeringReusePredictor

    for plain, reuse in ((DirectPredictor(), SteeringReusePredictor(1.0)),
                         (CFGPredictor(2.0), SteeringReusePredictor(1.0, scale=2.0))):
        a, _ = _run(plain)
        b, calls = _run(reuse)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert calls["run_painter"] == 0


def test_steering_reuse_stops_reasoning_after_k_steps():
    from models.sampling import SteeringReusePredictor

    out, calls = _run(SteeringReusePredictor(0.25, scale=2.0))
    assert calls == {"forward": 5 * 2, "run_painter": 15 * 2}, calls  # 20 steps, 25% reasoning, both CFG branches
    full, _ = _run(SteeringReusePredictor(1.0, scale=2.0))
    assert not torch.allclose(out, full)  # reusing stale steering changes the sample


def test_trajectory_recorder_records_x0():
    from models.sampling import TrajectoryRecorder

    rec = TrajectoryRecorder(steps={0, 10, 19})
    out, _ = _run(None, recorder=rec)
    assert [r["step"] for r in rec.records] == [0, 10, 19]
    torch.testing.assert_close(rec.records[-1]["x0_pred"], out)  # last step: x0 estimate = the sample (DDPM, t'<0)
