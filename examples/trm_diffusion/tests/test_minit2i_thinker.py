"""MiniT2I painter + thinker (models/minit2i.py) against the reference code it
reproduces, on the real released weights:

- the split, fused-attention forward vs the vendored MMJiT.forward
  (third_party/minit2i/mmdit.py);
- the training draws and loss vs Looped-DiT's training_loss, and sampling
  (scheduler + CFG) vs their euler_sample (third_party/looped_dit/looped_dit/
  diffusion.py), with identical random draws;
- steering: exactly a no-op at initialization, and gradients reaching the thinker;
- a short train_trm.py run of the experiment config.

Needs a GPU. Run from examples/trm_diffusion:  pytest tests/test_minit2i_thinker.py -s
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.data_sample import DataSample  # noqa: E402
from models.minit2i import MiniT2IFlowScheduler, MiniT2IPainter  # noqa: E402
from models.sampling import CFGPredictor, DirectPredictor, SamplingPipeline  # noqa: E402
from third_party.looped_dit.looped_dit.diffusion import euler_sample, training_loss  # noqa: E402
from third_party.minit2i import mmdit  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICE = torch.device("cuda")
PROMPTS = ["a red cube on top of a blue sphere", "four doors in a castle hallway, door 2 is bronze"]


@pytest.fixture(scope="module")
def painter():
    """fp32 (painter_dtype=None) so comparisons are not dominated by bf16 rounding."""
    p = MiniT2IPainter(scheduler=MiniT2IFlowScheduler(), train_cfg=None, eval_cfg=None, optim_cfg=None, painter_dtype=None)
    return p.to(DEVICE)


@pytest.fixture(scope="module")
def tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/flan-t5-large", model_max_length=256)
    t = tok(PROMPTS, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    return t["input_ids"].to(DEVICE), t["attention_mask"].to(DEVICE)


class _X0Model(nn.Module):
    """Their `model(x_t, text, text_mask, ...) -> x0_hat` interface over our painter."""

    def __init__(self, painter, token_ids):
        super().__init__()
        self.painter, self.token_ids = painter, token_ids

    def forward(self, x_t, text, text_mask, num_loops=None):
        return self.painter.decode_tail(self.painter.encode_head(x_t, self.token_ids, text_mask))


@torch.no_grad()
def test_forward_matches_vendored_mmjit(painter, tokens):
    ids, mask = tokens
    mask = mask.clone()
    mask[1] = 0  # second sample unconditional, as in a CFG / text-dropout pass
    ref = mmdit.MMJiT(painter.net.cfg).to(DEVICE).eval()  # original einsum attention
    ref.load_state_dict(painter.net.state_dict())
    x_t = torch.randn(2, 3, 512, 512, device=DEVICE) * 2
    context = painter.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    expected = ref(x_t, torch.full((2,), 0.3, device=DEVICE), context, mask)
    ours = painter.decode_tail(painter.encode_head(x_t, ids, mask))
    err = (ours - expected).abs().max().item()
    print(f"\nmax |ours - MMJiT.forward| = {err:.2e} (output std {expected.std():.2f})")
    assert err < 1e-3


def test_training_loss_matches_looped_dit(painter, tokens):
    ids, mask = tokens
    images = torch.rand(2, 3, 512, 512, device=DEVICE) * 2 - 1
    painter.label_drop_rate = 0.5  # exercise the text dropout too
    try:
        with torch.no_grad():
            torch.manual_seed(1)
            context = painter.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            theirs, _ = training_loss(_X0Model(painter, ids), images, context, mask, label_drop_rate=0.5)
            torch.manual_seed(1)
            sample = painter._prepare_training_sample(DataSample(images=images, token_conditions=ids, token_mask=mask), DEVICE)
            ours = F.mse_loss(painter(sample).pred, sample.target)
    finally:
        painter.label_drop_rate = 0.1
    print(f"\nloss: ours {ours.item():.6f}, Looped-DiT training_loss {theirs.item():.6f}")
    assert torch.allclose(ours, theirs, rtol=1e-5)


@torch.no_grad()
def test_sampling_matches_looped_dit_euler_sample(painter, tokens):
    """Without guidance the samplers agree bit for bit. With it, ours combines
    the two passes in velocity space and theirs in x0 space — the same algebra
    (guidance is linear), different fp32 rounding (~4e-6 in one step), which
    CFG x6 then amplifies from step to step; it stays far below one 8-bit level."""
    ids, mask = tokens
    context = painter.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    def both(steps, scale):
        torch.manual_seed(2)
        theirs = euler_sample(_X0Model(painter, ids), context, mask, 512, steps=steps, cfg_scale=scale)
        torch.manual_seed(2)
        # euler_sample skips the unconditional pass when cfg_scale == 1
        predictor = CFGPredictor(scale) if scale != 1.0 else DirectPredictor()
        pipeline = SamplingPipeline(num_inference_steps=steps, predictor=predictor)
        ours = pipeline.sample_one_batch(painter, DataSample(token_conditions=ids, token_mask=mask), DEVICE)
        return ours.clamp(-1, 1), theirs

    ours, theirs = both(4, 1.0)
    assert torch.equal(ours, theirs)
    ours, theirs = both(1, 6.0)
    assert (ours - theirs).abs().max() < 1e-5
    ours, theirs = both(4, 6.0)
    err = (ours - theirs).abs()
    print(f"\n4 CFG steps: max |ours - euler_sample| {err.max():.1e}, mean {err.mean():.1e} (8-bit level: {2 / 255:.1e})")
    assert err.max() < 1 / 255 and err.mean() < 1e-4


@pytest.fixture(scope="module")
def thinker_model(painter):
    """The experiment config's model with a tiny thinker, reusing the loaded painter."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from factory import build_model

    overrides = [
        "experiment=std120k_thinker_minit2i",
        "thinker.hidden_size=64", "thinker.n_heads=2", "thinker.vocab_size=16",
        "thinker.n_sup=2", "thinker.H_cycles=1", "thinker.L_cycles=1", "thinker.L_layers=1",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    import models.minit2i as minit2i

    real_load, real_t5 = minit2i.load_minit2i, None
    minit2i.load_minit2i = lambda *a, **k: painter.net  # don't load the weights a second time
    try:
        model = build_model(cfg, instantiate(cfg.diffusion))
    finally:
        minit2i.load_minit2i = real_load
    model.painter.text_encoder = painter.text_encoder
    model.painter.painter_dtype = None
    return model.to(DEVICE)


def _batch(tokens):
    ids, mask = tokens
    return DataSample(images=torch.rand(2, 3, 512, 512) * 2 - 1, token_conditions=ids.cpu(), token_mask=mask.cpu())


def test_steering_starts_as_noop(thinker_model, tokens):
    [d] = thinker_model.prep_mb_data([_batch(tokens)], DEVICE)
    with torch.no_grad():
        steered = thinker_model(d["sample"]).pred
        plain = thinker_model.painter(d["sample"]).pred
    assert torch.equal(steered, plain)


def test_gradients_reach_thinker(thinker_model, tokens):
    model = thinker_model
    with torch.no_grad():  # move off the zero init, as after a few optimizer steps
        for proj in model.thinker_painter_translator.projs:
            proj.weight.normal_(std=1e-2)
    try:
        [d] = model.prep_mb_data([_batch(tokens)], DEVICE)
        model.zero_grad(set_to_none=True)
        noise_pred, logits, _, _ = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
        loss, _ = model.loss_fn(noise_pred, logits, d["sample"])
        loss.backward()
        norms = {
            name: torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item()
            for name, params in [
                ("thinker", model.get_thinker_params()),
                ("encoder", list(model.condition_encoder.parameters())),
                ("translator", list(model.thinker_painter_translator.parameters())),
            ]
        }
        print(f"\nloss {loss.item():.4f}, grad norms {norms}")
        assert all(n > 1e-8 for n in norms.values()), norms
        assert all(p.grad is None for p in model.painter.parameters())  # painter + T5 stay frozen
    finally:
        with torch.no_grad():
            for proj in model.thinker_painter_translator.projs:
                proj.weight.zero_()


def test_eval_loss_seed(thinker_model, tokens):
    """eval_step(loss_seed=...): the same val loss at every eval, and the
    training RNG continues as if no eval had run."""
    from types import SimpleNamespace

    model, callbacks = thinker_model, thinker_model.eval_callbacks
    model.eval_callbacks = []  # loss only, no sampling
    batch, acc = [_batch(tokens)], SimpleNamespace(device=DEVICE)
    try:
        torch.manual_seed(1)
        expected = torch.rand(3), torch.rand(3, device=DEVICE)
        torch.manual_seed(1)
        a = model.eval_step(batch, acc, loss_seed=0)["diff_loss"]
        b = model.eval_step(batch, acc, loss_seed=0)["diff_loss"]
        after = torch.rand(3), torch.rand(3, device=DEVICE)
        c = model.eval_step(batch, acc, loss_seed=1)["diff_loss"]
        d, e = (model.eval_step(batch, acc)["diff_loss"] for _ in range(2))  # unseeded, as before
    finally:
        model.eval_callbacks = callbacks
    print(f"\nseed 0: {a:.6f} {b:.6f}, seed 1: {c:.6f}, unseeded: {d:.6f} {e:.6f}")
    assert a == pytest.approx(b, rel=1e-6) and all(torch.equal(x, y) for x, y in zip(after, expected))
    assert abs(a - c) > 1e-4 * a and abs(d - e) > 1e-4 * d


def test_train_script_smoke(tmp_path):
    """A few real train_trm.py steps of the experiment, incl. an eval with sampling."""
    fixture = ROOT / "data" / "std120k_testfixture"
    if not fixture.exists():
        pytest.skip("run tests/test_std120k_dataset.py first (builds the Std-120K fixture)")
    from datasets.std120k_dataset import LocalRawFiles, Std120KDataset

    data_root = tmp_path / "std120k"
    Std120KDataset(root=data_root, raw_files=LocalRawFiles(fixture), build_workers=4)
    cmd = [
        sys.executable, "train_trm.py", "experiment=std120k_thinker_minit2i",
        f"run.output_dir={tmp_path / 'run'}", f"data.root={data_root}", "data.num_workers=2",
        "thinker.hidden_size=64", "thinker.n_heads=2", "thinker.vocab_size=16",
        "thinker.n_sup=2", "thinker.H_cycles=1", "thinker.L_cycles=1", "thinker.L_layers=1",
        "train.batch_size=2", "train.gradient_accumulation_steps=1", "train.num_steps=4",
        "eval.eval_every=4", "eval.save_every=100", "eval.log_every=2",
        "sampling.num_inference_steps=2", "eval_callbacks.0.num_log_images=2",
    ]
    env = {**os.environ, "WANDB_MODE": "disabled"}
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "train/diff_loss" in res.stdout + res.stderr
    assert (tmp_path / "run" / "samples" / "samples_step4.png").exists()
    assert (tmp_path / "run" / "checkpoint_final.pt").exists()
