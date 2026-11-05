# ABOUTME: Tests mode=thinker_act_gram (GRAM stochastic thinker): the KL formula, the hooked forward equals the plain
# ABOUTME: thinker at eps = 0, gradient reaches posterior/prior/encoders in training, inference uses only the prior.
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.data_sample import collate_data_samples  # noqa: E402
from factory import build_model  # noqa: E402
from models.gram_thinkers import GramSpatialTRM, gauss_kl  # noqa: E402
from models.trm_wrappers import SpatialTRM  # noqa: E402


R = "runs/local_rope_debug"
PAINTER = f"{R}/ckpt/painter_p16_all.pt"
needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(PAINTER)),
    reason="needs a GPU and the p16_all painter (runs/local_rope_debug)",
)
GRAM = [
    "mode=thinker_act_gram",
    "thinker._target_=models.gram_thinkers.GramSpatialTRM",
    "+train.gram_beta=0.1",
    "+train.act_halt_label_clean=true",
]


def _build(extra):
    ov = [
        l.strip()
        for l in open(f"{R}/ov_all_act.txt")
        if l.strip() and not l.startswith(("painter.checkpoint=", "eval"))
    ]
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(
            config_name="config", overrides=ov + extra + [f"painter.checkpoint={PAINTER}", "run.wandb_project=null"]
        )
    cfg.eval_callbacks = None
    return build_model(cfg, instantiate(cfg.diffusion)).cuda(), cfg


def test_gauss_kl_matches_torch():
    g = torch.Generator().manual_seed(0)
    mq, mp = torch.randn(5, 7, generator=g), torch.randn(5, 7, generator=g)
    sq, sp = torch.rand(5, 7, generator=g) + 0.1, torch.rand(5, 1, generator=g) + 0.1
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mq, sq), torch.distributions.Normal(mp, sp))
    assert torch.allclose(gauss_kl(mq, sq, mp, sp), ref, atol=1e-6)


@needs_gpu
@pytest.mark.parametrize("sigma", ["token", "channel"])
def test_zero_noise_equals_plain_thinker(sigma):
    model, cfg = _build(GRAM + [f"+thinker.gram_sigma={sigma}"])
    th = model.thinker
    assert isinstance(th, GramSpatialTRM)
    plain_cfg = {k: v for k, v in cfg.thinker.items() if not k.startswith("gram_")}
    plain = instantiate({**plain_cfg, "_target_": "models.trm_wrappers.SpatialTRM"}).cuda()
    assert type(plain) is SpatialTRM
    plain.load_state_dict(
        {k: v for k, v in th.state_dict().items() if not k.startswith(("prior_head", "posterior_head"))}
    )
    th.eval(), plain.eval()
    th.prior_mode = "mean"  # zero-init heads: mean eps = 0
    enc = torch.randn(4, 81, th.inner.config.hidden_size, device="cuda")
    z_H, z_L = (z.cuda() for z in th.get_initial_states(4))
    t = torch.randint(0, 100, (4,), device="cuda")
    with torch.no_grad():
        a = th.reasoning_step(enc, z_H, z_L, None, timesteps=t)
        b = plain.reasoning_step(enc, z_H, z_L, None, timesteps=t)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    th.prior_mode = "sample"
    with torch.no_grad():
        c = th.reasoning_step(enc, z_H, z_L, None, timesteps=t)
    assert not torch.equal(a[0], c[0])  # init std 0.1: sampled eps changes the logits


@needs_gpu
@pytest.mark.parametrize("free_bits", [0.5, 0.0])
def test_train_step_gradient_flow(free_bits):
    model, cfg = _build(GRAM + [f"+train.gram_free_bits={free_bits}"])
    with torch.no_grad():  # off the zero-init output layers (translator, heads), which pass no gradient back
        for proj in model.thinker_painter_translator.projs:
            proj[-1].weight.normal_(std=0.02)
        model.thinker.prior_head.down.weight.normal_(std=0.02)
        model.thinker.posterior_head.down.weight.normal_(std=0.02)
    model.train()
    ds = instantiate(cfg.data.train_dataset)
    batch = collate_data_samples([ds[i] for i in range(8)])
    acc = SimpleNamespace(
        device=torch.device("cuda"), backward=lambda l: l.backward(), clip_grad_norm_=torch.nn.utils.clip_grad_norm_
    )
    grads = {}
    hooks = [
        p.register_hook(lambda g, n=n: grads.__setitem__(n, grads.get(n, 0.0) + g.abs().sum().item()))
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    losses, _, _ = model.train_step([batch], acc, model.build_optimizers(1, 100), None, 0)
    for h in hooks:
        h.remove()
    print(free_bits, {k: round(v, 5) for k, v in losses.items()})
    assert "gram_kl_loss" in losses and torch.isfinite(torch.tensor(losses["gram_kl_loss"]))

    def got(prefix):
        return sum(v for k, v in grads.items() if k.startswith(prefix))

    assert got("thinker.inner") > 0 and got("thinker.posterior_head") > 0 and got("posterior_encoder.") > 0
    assert got("condition_encoder.") > 0 and got("thinker_painter_translator.") > 0
    assert got("painter.") == 0
    if free_bits == 0.0:
        # KL 0 at init is under any positive free-bits floor; with none, the prior gets the KL gradient
        assert got("thinker.prior_head") > 0


@needs_gpu
def test_inference_uses_only_the_prior():
    model, cfg = _build(GRAM)
    model.eval()
    with torch.no_grad():  # off the translator's zero init, which ignores the thinker's output
        for proj in model.thinker_painter_translator.projs:
            proj[-1].weight.normal_(std=0.02)
    model.thinker.prior_head.down.bias.data[: model.thinker.inner.config.hidden_size].normal_()  # nonzero prior mean
    model.posterior_encoder.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("posterior used"))
    ds = instantiate(cfg.data.val_dataset)
    raw = collate_data_samples([ds[i] for i in range(4)])
    torch.manual_seed(0)
    [d] = model.prep_mb_data([raw], torch.device("cuda"))
    model.thinker.prior_mode = "mean"
    with torch.no_grad():
        p1, p2 = model(d["sample"]).pred, model(d["sample"]).pred
    assert torch.equal(p1, p2)
    model.thinker.prior_mode = "sample"
    with torch.no_grad():
        p3 = model(d["sample"]).pred
    assert not torch.equal(p1, p3)
