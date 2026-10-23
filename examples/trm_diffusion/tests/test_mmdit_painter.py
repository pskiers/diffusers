"""Generic MM-DiT painter (models/mmdit_painter.py) and its thinker steering, on
tiny models (CPU is enough; models/minit2i.py's released-weights path is checked
by tests/test_minit2i_thinker.py):

- the fused blocks with cond_rope "1d" against the vendored MMJiT.forward;
- stage-1 training reaches every painter parameter;
- condition dropout in training == null_condition_sample (what CFG sampling uses);
- steering, image and condition stream: a no-op at initialization, and
  gradients reaching the thinker's output at both kinds of position; with
  bridge_mode "normalized", independent of the thinker output's scale;
- every new experiment config builds, with consistent token counts.

Run from examples/trm_diffusion:  pytest tests/test_mmdit_painter.py -s
"""

import copy
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.data_sample import DataSample  # noqa: E402
from models.interfaces import TRMOutput  # noqa: E402
from models.mmdit_painter import (  # noqa: E402
    EmbeddingCondTokenizer,
    NullCondTokenizer,
    FlowMatchingScheduler,
    MMDiTPainter,
    PatchCondTokenizer,
    fuse_attention,
)
from models.translators import PainterResidualTranslator  # noqa: E402
from third_party.minit2i import mmdit  # noqa: E402

DEV = torch.device("cpu")
NET = dict(image_size=48, patch_size=12, in_channels=1, hidden_size=64, depth=4, preamble_depth=1, num_heads=2, head_dim=32)
OPTIM = SimpleNamespace(lr=1e-4, weight_decay=0.0, warmup_steps=0, lr_min_ratio=0.0)


def _painter(kind: str = "patch", **kw) -> MMDiTPainter:
    tokenizer = PatchCondTokenizer(1, 12) if kind == "patch" else EmbeddingCondTokenizer(7)
    return MMDiTPainter(
        scheduler=FlowMatchingScheduler(noise_scale=1.0), train_cfg=SimpleNamespace(grad_clip=1.0), eval_cfg=None,
        optim_cfg=OPTIM, cond_tokenizer=tokenizer, net=NET, steer_from_block=2, pixel_range="[0,1]", **kw,
    )


def _batch(kind: str = "patch", b: int = 3) -> DataSample:
    torch.manual_seed(0)
    images = torch.rand(b, 1, 48, 48)
    if kind == "patch":
        return DataSample(images=images, spatial_conditions=torch.rand(b, 1, 48, 48))
    mask = torch.tensor([[1.0] * 5 + [0.0] * 3] * b)
    return DataSample(images=images, embedding_conditions=torch.randn(b, 8, 7), embedding_mask=mask)


@torch.no_grad()
def test_fused_1d_matches_vendored_mmjit():
    torch.manual_seed(0)
    cfg = mmdit.MMJiTConfig(image_size=32, patch_size=8, in_channels=3, txt_input_size=24, hidden_size=64,
                            txt_hidden_size=64, cond_vec_size=64, depth_double=3, txt_preamble_depth=1,
                            num_heads=2, head_dim=32, pca_channels=16)
    ref = mmdit.MMJiT(cfg).eval()
    for p in ref.parameters():  # MMJiT's mask_token starts at 0; move every parameter off its init
        p.add_(torch.randn_like(p) * 0.05)
    painter = MMDiTPainter(scheduler=FlowMatchingScheduler(), train_cfg=None, eval_cfg=None, optim_cfg=None,
                           cond_tokenizer=EmbeddingCondTokenizer(24), net=fuse_attention(copy.deepcopy(ref)),
                           cond_rope="1d", steer_from_block=1, freeze=True)
    x_t, context = torch.randn(2, 3, 32, 32), torch.randn(2, 5, 24)
    mask = torch.tensor([[1.0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
    expected = ref(x_t, torch.full((2,), 0.3), context, mask)
    ours = painter.decode_tail(painter.encode_head(x_t, context, mask))
    assert (ours - expected).abs().max() < 1e-5


@pytest.mark.parametrize("kind", ["patch", "embedding"])
def test_stage1_gradients_reach_every_parameter(kind):
    painter = _painter(kind, label_drop_rate=0.0)  # dropout: see test_condition_dropout_equals_cfg_null
    torch.manual_seed(1)
    sample = painter._prepare_training_sample(_batch(kind), DEV)
    assert sample.painter_hidden is None  # trainable: computed in forward, with gradients
    # The zero-initialized output layer stops gradients below it on the very first
    # step; once it has moved, every parameter must be reached — except mask_token
    # under patch conditioning, where every token is real, and the last block's
    # condition-stream output path, which nothing reads (as in MiniT2I).
    with torch.no_grad():
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    painter._loss(sample).backward()
    last = f"net.double_blocks.{NET['depth'] - 1}."
    dead = {last + n for n in ["txt_norm2.weight", "txt_attn_proj.weight", "txt_attn_proj.bias",
                               "txt_mlp.w1.weight", "txt_mlp.w2.weight", "txt_mlp.w3.weight"]}
    dead |= {"net.mask_token"} if kind == "patch" else set()
    no_grad = {n for n, p in painter.named_parameters() if p.grad is None or p.grad.abs().sum() == 0}
    assert no_grad == dead, no_grad ^ dead


@pytest.mark.parametrize("kind", ["patch", "embedding"])
def test_condition_dropout_equals_cfg_null(kind):
    painter = _painter(kind, label_drop_rate=1.0, freeze=True)
    torch.manual_seed(2)
    dropped = painter._prepare_training_sample(_batch(kind), DEV)
    real = dataclasses.replace(_batch(kind), x_noisy=dropped.x_noisy, timesteps=dropped.timesteps)
    null = painter.fill_runtime_fields(painter.null_condition_sample(real))
    assert torch.equal(dropped.painter_hidden, null.painter_hidden)
    assert not torch.equal(painter.fill_runtime_fields(real).painter_hidden, null.painter_hidden)


@pytest.mark.parametrize("bridge_mode", ["logits", "normalized"])
@pytest.mark.parametrize("kind", ["patch", "embedding"])
def test_steering_noop_at_init_and_gradients(kind, bridge_mode):
    painter = _painter(kind, freeze=True)
    torch.manual_seed(3)
    with torch.no_grad():  # as after stage 1: the zero-initialized output layer has moved
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    sample = painter._prepare_training_sample(_batch(kind), DEV)
    n_cond = sample.painter_hidden.shape[1] - 16
    translator = PainterResidualTranslator(in_dim=8, painter_dim=64, grid=2, painter_grid=4, num_blocks=2,
                                           steer_cond=True, bridge_mode=bridge_mode)
    logits = torch.randn(3, 4 + n_cond, 8, requires_grad=True)  # 2x2 image grid (pooled) + condition tokens
    plain = painter(sample).pred
    assert torch.equal(painter(sample, translator(TRMOutput(logits=logits))).pred, plain)

    with torch.no_grad():
        for proj in [*translator.projs, *translator.cond_projs]:
            proj.weight.normal_(std=0.1)
    torch.nn.functional.mse_loss(painter(sample, translator(TRMOutput(logits=logits))).pred, sample.target).backward()
    img_grad, cond_grad = logits.grad[:, :4].abs().sum(), logits.grad[:, 4:].abs().sum()
    assert img_grad > 0 and cond_grad > 0, (img_grad, cond_grad)
    assert all(p.grad is None for p in painter.parameters())  # painter stays frozen

    # condition steering alone changes the output
    cond_only = logits.detach().clone()
    cond_only[:, :4] = 0
    with torch.no_grad():
        assert not torch.equal(painter(sample, translator(TRMOutput(logits=cond_only))).pred, plain)

    # normalized: shrinking the thinker's output (e.g. by weight decay) leaves the steering as it was
    # (up to BatchNorm's eps, 1e-5 — negligible next to the output's variance here, 1e-2)
    if bridge_mode == "normalized":
        with torch.no_grad():
            full = translator(TRMOutput(logits=logits))
            small = translator(TRMOutput(logits=logits * 0.1))
        for a, b in zip(full.block_residuals + full.cond_block_residuals, small.block_residuals + small.cond_block_residuals):
            assert torch.allclose(a, b, atol=1e-3 * a.abs().max())


@pytest.mark.parametrize("hidden_dim, temb", [(None, True), (32, False), (32, True)])
def test_translator_chain_and_timestep_emb(hidden_dim, temb):
    """Translator variants: still a no-op at initialization; once the zero-initialized
    output layers move, gradients reach every new part, and with the timestep
    embedding the steering depends on t."""
    painter = _painter("patch", freeze=True)
    torch.manual_seed(4)
    with torch.no_grad():
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    sample = painter._prepare_training_sample(_batch("patch"), DEV)
    translator = PainterResidualTranslator(in_dim=8, painter_dim=64, grid=2, painter_grid=4, num_blocks=2,
                                           bridge_mode="normalized", hidden_dim=hidden_dim, with_timestep_emb=temb)
    assert translator.condition_keys == (["timesteps"] if temb else [])
    logits = torch.randn(3, 4, 8, requires_grad=True)
    steer = lambda t: translator(TRMOutput(logits=logits), timesteps=t)
    assert torch.equal(painter(sample, steer(sample.timesteps)).pred, painter(sample).pred)

    with torch.no_grad():
        for proj in translator.projs:
            (proj[-1] if hidden_dim else proj).weight.normal_(std=0.1)
    torch.nn.functional.mse_loss(painter(sample, steer(sample.timesteps)).pred, sample.target).backward()
    assert logits.grad.abs().sum() > 0
    parts = list(translator.projs.parameters())
    parts += list(translator.in_proj.parameters()) if hidden_dim else []
    parts += list(translator.timestep_mlp.parameters()) if temb else []
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in parts)
    with torch.no_grad():
        r1 = steer(torch.full((3,), 0.1)).block_residuals[0]
        r2 = steer(torch.full((3,), 0.7)).block_residuals[0]
    assert torch.equal(r1, r2) != temb


def test_x0_loss_weighting():
    """loss_weighting="x0": the thinker-stage MSE equals the x0 MSE (velocity MSE x (1 - t)^2)."""
    from models.losses import MSELoss

    painter = _painter("patch", freeze=True, loss_weighting="x0")
    torch.manual_seed(6)
    with torch.no_grad():
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    s = painter._prepare_training_sample(_batch("patch"), DEV)
    t4 = s.timesteps.view(-1, 1, 1, 1)
    assert (1 - s.timesteps).min() > painter.velocity_clamp  # unclamped: v * (1 - t) = x0 - x_t exactly
    pred = painter(s).pred
    loss, _ = MSELoss(1.0, painter.scheduler)(pred, None, s)
    x0_hat, x0 = s.x_noisy + pred * (1 - t4), s.images
    assert torch.allclose(loss, (x0_hat - x0).pow(2).mean(), rtol=1e-4)
    assert torch.allclose(painter._loss(s), loss, rtol=1e-5)
    unweighted, _ = MSELoss(1.0, painter.scheduler)(pred, None, dataclasses.replace(s, loss_weight=None))
    assert torch.allclose(unweighted, torch.nn.functional.mse_loss(pred, s.target))


def test_time_cond_painter():
    """net.time_cond: every block starts as the identity (adaLN-Zero), stage-1 training
    reaches the modulation layers, and afterwards the output depends on t."""
    painter = MMDiTPainter(
        scheduler=FlowMatchingScheduler(noise_scale=1.0), train_cfg=SimpleNamespace(grad_clip=1.0), eval_cfg=None,
        optim_cfg=OPTIM, cond_tokenizer=PatchCondTokenizer(1, 12), net={**NET, "time_cond": True}, steer_from_block=2,
        pixel_range="[0,1]", label_drop_rate=0.0)
    torch.manual_seed(7)
    with torch.no_grad():
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    s = painter._prepare_training_sample(_batch("patch"), DEV)
    x = torch.randn(3, 16, 64)
    for block in painter.net.double_blocks:  # identity at init
        out, _ = block(x, torch.randn(3, 16, 64), painter._vec(s.timesteps))
        assert torch.equal(out, x)
    painter._loss(s).backward()
    adaln = [p for b in painter.net.double_blocks for p in b.adaln.parameters()] + list(painter.net.final_adaln.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in adaln)
    with torch.no_grad():
        for p in adaln:
            p.normal_(std=0.02)
        t1, t2 = torch.full((3,), 0.2), torch.full((3,), 0.7)
        x0a = painter.decode_tail(painter.encode_head(s.x_noisy, s.spatial_conditions, t=t1), t=t1)
        x0b = painter.decode_tail(painter.encode_head(s.x_noisy, s.spatial_conditions, t=t2), t=t2)
    assert not torch.allclose(x0a, x0b)
    with pytest.raises(ValueError):
        painter.encode_head(s.x_noisy, s.spatial_conditions)  # a time-conditioned net needs t


def test_ddpm_objective():
    """objective="ddpm": integer t, x_t from the DDPM scheduler, x0 prediction as output and target."""
    from diffusers import DDPMScheduler

    sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="sample")
    painter = MMDiTPainter(
        scheduler=sched, train_cfg=SimpleNamespace(grad_clip=1.0), eval_cfg=None, optim_cfg=OPTIM,
        cond_tokenizer=PatchCondTokenizer(1, 12), net={**NET, "time_cond": True}, steer_from_block=2,
        pixel_range="[0,1]", objective="ddpm", time_cond_scale=1.0, label_drop_rate=0.0)
    batch = _batch("patch")
    torch.manual_seed(8)
    s = painter._prepare_training_sample(batch, DEV)
    assert s.timesteps.dtype == torch.long and int(s.timesteps.max()) < 100
    assert torch.equal(s.target, painter.to_model_range(batch.images))
    torch.manual_seed(8)
    t = torch.randint(0, 100, (3,), dtype=torch.long)
    assert torch.allclose(s.x_noisy, sched.add_noise(s.target, torch.randn_like(s.target), t))
    out = painter(s)
    assert out.pred_type == "sample" and out.pred.shape == s.target.shape
    with pytest.raises(ValueError):
        MMDiTPainter(scheduler=FlowMatchingScheduler(), train_cfg=None, eval_cfg=None, optim_cfg=OPTIM,
                     cond_tokenizer=PatchCondTokenizer(1, 12), net=NET, objective="ddpm")


def test_null_cond_painter():
    """NullCondTokenizer: no condition tokens — image-only attention, nothing to drop or null."""
    painter = MMDiTPainter(
        scheduler=FlowMatchingScheduler(noise_scale=1.0), train_cfg=SimpleNamespace(grad_clip=1.0), eval_cfg=None,
        optim_cfg=OPTIM, cond_tokenizer=NullCondTokenizer(in_channels=1, patch_size=12), net={**NET, "preamble_depth": 0},
        steer_from_block=0, pixel_range="[0,1]", label_drop_rate=1.0)
    assert painter.condition_keys == []
    torch.manual_seed(9)
    with torch.no_grad():
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    s = painter._prepare_training_sample(_batch("patch"), DEV)
    hidden = painter.encode_head(s.x_noisy, t=s.timesteps)
    assert hidden.shape[1] == painter.img_grid**2  # image tokens only
    painter._loss(s).backward()
    img = [p for n, p in painter.named_parameters() if ".img_" in n or "img_embedder" in n]
    assert img and all(p.grad is not None for p in img)
    assert torch.equal(painter(painter.null_condition_sample(s)).pred, painter(s).pred)


@pytest.mark.parametrize("opts", [dict(norm="layer", qk_norm=False),
                                  dict(patch_embed="plain", zero_init_output=False),
                                  dict(mlp="gelu", rope=False)])
def test_dit_style_painter_options(opts):
    """build_mmdit's DiT-style options: no RMSNorm left / plain patch embedding, still trains."""
    from third_party.minit2i import mmdit as vendored

    painter = MMDiTPainter(
        scheduler=FlowMatchingScheduler(noise_scale=1.0), train_cfg=SimpleNamespace(grad_clip=1.0), eval_cfg=None,
        optim_cfg=OPTIM, cond_tokenizer=PatchCondTokenizer(1, 12), net={**NET, "time_cond": True, **opts},
        steer_from_block=2, pixel_range="[0,1]", label_drop_rate=0.0)
    mods = list(painter.net.modules())
    if opts.get("norm") == "layer":
        assert not any(isinstance(m, vendored.RMSNorm) for m in mods)
        assert all(isinstance(b.q_norm, torch.nn.Identity) for b in painter.net.double_blocks)
    if opts.get("mlp") == "gelu":
        assert not any(isinstance(m, vendored.SwiGLUMlp) for m in mods)
    if opts.get("rope") is False:
        blk = painter.net.double_blocks[0]
        q_t, q_i = torch.randn(2, 16, 2, 32), torch.randn(2, 16, 2, 32)
        assert not blk.use_rope and torch.equal(blk._rope(q_t, q_i), torch.cat([q_t, q_i], dim=1))
    if opts.get("patch_embed") == "plain":
        assert not any(isinstance(m, vendored.BottleneckPatchEmbed) for m in mods)
        assert painter.net.final_layer.linear.weight.abs().sum() > 0  # not zero-initialized
    torch.manual_seed(10)
    s = painter._prepare_training_sample(_batch("patch"), DEV)
    loss = painter._loss(s)
    loss.backward()
    assert torch.isfinite(loss) and painter.net.double_blocks[0].img_qkv.weight.grad is not None


def test_model_range_01():
    """model_range [0,1] with [0,1] data: no remapping, and eval decoding stays in [0,1]."""
    painter = _painter("patch", model_range="[0,1]")
    x = torch.rand(2, 1, 48, 48)
    assert torch.equal(painter.to_model_range(x), x)
    assert torch.equal(painter.decode_for_eval(x), x) and torch.equal(painter.images_to_log(x), x)
    assert torch.equal(_painter("patch").images_to_log(x), x)  # default [-1,1] model: round trip


TINY_THINKER = ["thinker.hidden_size=64", "thinker.n_heads=2", "thinker.n_sup=1", "thinker.H_cycles=1",
                "thinker.L_cycles=1", "thinker.L_layers=1", "thinker.forward_dtype=float32"]
TINY_PAINTER = ["painter.net.hidden_size=64", "painter.net.num_heads=2", "painter.net.head_dim=32", "painter.painter_dtype=null"]


def _build(overrides):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from factory import build_model

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.eval_callbacks = None  # no classifier / detector loading
    return build_model(cfg, instantiate(cfg.diffusion))


CNN12 = ["painter.steer_from_block=1", "translator.num_blocks=7", "condition_encoder.factor=12", "translator.grid=12",
         "thinker.seq_len=144", "+train.force_unconditional_painter=true"]
P16 = ["painter.net.patch_size=16", "painter.steer_from_block=1", "translator.num_blocks=7", "translator.painter_grid=9",
       "translator.hidden_dim=64", "translator.with_timestep_emb=true", "+train.force_unconditional_painter=true"]
SUDOKU = DataSample(images=torch.rand(2, 1, 144, 144), spatial_conditions=torch.rand(2, 1, 144, 144),
                    solution=torch.randint(0, 9, (2, 81)))
CLEVR = DataSample(images=torch.rand(2, 3, 256, 256) * 2 - 1, embedding_conditions=torch.randn(2, 10, 55),
                   embedding_mask=torch.tensor([[1.0] * 4 + [0.0] * 6] * 2))


@pytest.mark.parametrize(
    "painter_exp, thinker_exp, batch, extra",
    [
        ("mnist_mmdit_painter", "mnist_thinker_mmdit", SUDOKU, ["translator.steer_cond=true"]),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit", SUDOKU, ["painter.steer_from_block=1", "translator.num_blocks=7"]),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, []),
        *[("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, CNN12 + [v]) for v in
          ("translator.with_timestep_emb=true", "translator.hidden_dim=64", "painter.loss_weighting=x0")],
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU,
         CNN12 + ["translator.hidden_dim=64", "translator.with_timestep_emb=true"]),
        *[("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, P16 + v) for v in (
            [], ["painter.label_drop_rate=1.0"], ["painter.net.time_cond=true"],
            ["diffusion=ddpm", "sampling=cfg", "painter.objective=ddpm", "painter.time_cond_scale=1", "translator.timestep_scale=1"],
            ["painter.label_drop_rate=1.0", "painter.net.time_cond=true", "diffusion=ddpm", "sampling=cfg",
             "painter.objective=ddpm", "painter.time_cond_scale=1", "translator.timestep_scale=1"])],
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, P16 + ["painter.net.norm=layer", "painter.net.qk_norm=false"]),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, P16 + [
            "painter.net.patch_embed=plain", 'painter.model_range="[0,1]"', "painter.net.zero_init_output=false"]),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, P16 + ['painter.model_range="[0,1]"']),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, P16 + [  # the DiT-ified MM-DiT
            "painter.net.patch_embed=plain", 'painter.model_range="[0,1]"', "painter.net.zero_init_output=false",
            "painter.net.norm=layer", "painter.net.qk_norm=false", "painter.net.mlp=gelu", "painter.net.rope=false",
            "painter.cond_tokenizer._target_=models.mmdit_painter.NullCondTokenizer", "painter.net.preamble_depth=0"]),
        # the most DiT-like combination: no condition stream, depth 3, every block steered, fp32 painter
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, [
            "painter.net.patch_size=16", "painter.net.depth=3", "painter.net.time_cond=true", "diffusion=ddpm", "sampling=cfg",
            "painter.objective=ddpm", "painter.time_cond_scale=1", "translator.timestep_scale=1",
            "painter.cond_tokenizer._target_=models.mmdit_painter.NullCondTokenizer", "painter.net.preamble_depth=0",
            "painter.steer_from_block=0", "translator.num_blocks=3", "translator.painter_grid=9", "translator.hidden_dim=64",
            "translator.with_timestep_emb=true", "+train.force_unconditional_painter=true", "painter.painter_dtype=null",
            "painter.label_drop_rate=0.0"]),
        ("clevr_mmdit_painter", "clevr_thinker_mmdit", CLEVR, ["translator.steer_cond=true"]),
    ],
)
def test_experiment_configs_build(tmp_path, painter_exp, thinker_exp, batch, extra):
    # stage 1 with the same painter-side overrides (architecture, objective) as stage 2
    painter_side = [o for o in extra if o.startswith(("painter.net.", "painter.objective", "painter.time_cond_scale",
                                                      "painter.cond_tokenizer", "painter.steer_from_block", "painter.model_range", "diffusion=", "sampling="))]
    painter = _build([f"experiment={painter_exp}", *TINY_PAINTER, *painter_side])
    ckpt = tmp_path / "painter.pt"
    torch.save({"model_state": painter.state_dict()}, ckpt)

    model = _build([f"experiment={thinker_exp}", f"painter.checkpoint={ckpt}", *TINY_PAINTER, *TINY_THINKER, *extra])
    assert not any(p.requires_grad for p in model.painter.parameters())
    [d] = model.prep_mb_data([batch], DEV)
    s = d["sample"]
    with torch.no_grad():
        steered = model(s).pred
        # the painter as the model runs it: from its null condition under force_unconditional_painter
        painted = model.painter.null_condition_sample(s) if model.train_cfg.force_unconditional_painter else s
        plain = model.painter(painted).pred
    assert torch.equal(steered, plain)  # zero-initialized translator
    noise_pred, logits, _, _ = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
    assert noise_pred.shape == batch.images.shape


def test_force_unconditional_painter(tmp_path):
    """The painter paints from its null condition; the thinker still reads the real one."""
    painter = _build(["experiment=mnist_mmdit_painter", *TINY_PAINTER])
    with torch.no_grad():  # as after stage 1: the zero-initialized output layer has moved
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    ckpt = tmp_path / "painter.pt"
    torch.save({"model_state": painter.state_dict()}, ckpt)
    model = _build(["experiment=mnist_thinker_mmdit", f"painter.checkpoint={ckpt}", "painter.steer_from_block=1",
                    "translator.num_blocks=7", "+train.force_unconditional_painter=true", *TINY_PAINTER, *TINY_THINKER])
    [d] = model.prep_mb_data([SUDOKU], DEV)
    s = d["sample"]
    with torch.no_grad():
        steered = model(s).pred  # zero-initialized translator: just the painter, as it is run
        real = model.painter(s).pred
        null = model.painter(model.painter.null_condition_sample(s)).pred
        enc = model._encode_condition(s)
        enc_null = model.condition_encoder(model.painter.fill_runtime_fields(model.painter.null_condition_sample(s)).painter_hidden).enc_emb
    assert torch.equal(steered, null) and not torch.allclose(steered, real)
    assert not torch.allclose(enc, enc_null)  # the thinker's input is still the real condition's


@pytest.mark.parametrize("steer_from", [0, 1])
def test_dit_residual_bisection_config(tmp_path, steer_from):
    """Our steering pipeline on the DiT-direct painter (ResidualSteeredDiTPainter): builds from
    the real painter config, starts as a no-op, gradients reach the thinker's output once the
    translator moves, and residuals go exactly to blocks steer_from.. of the DiT."""
    from models.interfaces import BlockResidualSteering

    painter = _build(["experiment=mnist_dit_painter_pixel"])
    ckpt = tmp_path / "dit.pt"
    torch.save({"model_state": painter.state_dict()}, ckpt)
    model = _build(["experiment=mnist_thinker_dit_residual", f"painter.checkpoint={ckpt}",
                    f"painter.steer_from_block={steer_from}", f"translator.num_blocks={3 - steer_from}", *TINY_THINKER])
    assert not any(p.requires_grad for p in model.painter.parameters())
    [d] = model.prep_mb_data([SUDOKU], DEV)
    s = d["sample"]
    with torch.no_grad():
        assert torch.equal(model(s).pred, model.painter(s).pred)  # zero-initialized translator

    # residuals land on blocks steer_from.. only
    outs = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, j=j: outs.__setitem__(j, o))  # block output incl. residual
             for j, blk in enumerate(model.painter.dit.transformer_blocks)]
    torch.manual_seed(0)
    res = [torch.randn(2, 81, 512) for _ in range(3 - steer_from)]
    with torch.no_grad():
        plain = model.painter(s).pred
        plain_outs = dict(outs)
        steered = model.painter(s, BlockResidualSteering(block_residuals=res)).pred
    for h in hooks:
        h.remove()
    assert not torch.allclose(steered, plain)
    if steer_from == 1:  # block 0 is not steered: its output is unchanged
        assert torch.equal(outs[0], plain_outs[0])
    else:
        assert not torch.equal(outs[0], plain_outs[0])

    with torch.no_grad():  # move off the zero init, as after a few optimizer steps
        for proj in model.thinker_painter_translator.projs:
            proj[-1].weight.normal_(std=0.02)
    noise_pred, logits, _, _ = model.reasoning_step(s, d["z_H"], d["z_L"])
    model.loss_fn(noise_pred, logits, s)[0].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.thinker_painter_translator.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.get_thinker_params())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the thinker's adam_atan2 optimizer is CUDA-only")
@pytest.mark.parametrize("painter_exp, thinker_exp, extra", [
    ("mnist_dit_painter_pixel", "mnist_thinker_dit_residual", []),
    ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", P16 + [
        "painter.net.time_cond=true", "diffusion=ddpm", "sampling=cfg", "painter.objective=ddpm",
        "painter.time_cond_scale=1", "translator.timestep_scale=1"]),
])
def test_act_training_with_new_painters(tmp_path, painter_exp, thinker_exp, extra):
    """mode=thinker_act (persistent-carry ACT loop) runs with the residual-steered painters:
    the per-sample merge of fresh samples handles their fields (painter_hidden, integer
    DDPM timesteps, ...) and the losses stay finite."""
    painter_side = [o for o in extra if o.startswith(("painter.net.", "painter.objective", "painter.time_cond_scale",
                                                      "painter.cond_tokenizer", "painter.steer_from_block",
                                                      "painter.model_range", "diffusion=", "sampling="))]
    painter = _build([f"experiment={painter_exp}", *(TINY_PAINTER if "mmdit" in painter_exp else []), *painter_side])
    ckpt = tmp_path / "painter.pt"
    torch.save({"model_state": painter.state_dict()}, ckpt)
    model = _build([f"experiment={thinker_exp}", f"painter.checkpoint={ckpt}", "mode=thinker_act",
                    "thinker.with_halt_head=true", "thinker.n_sup=3", *TINY_THINKER[:3], *TINY_THINKER[4:],
                    *(TINY_PAINTER if "mmdit" in painter_exp else []), *extra])
    assert type(model).__name__ == "ThinkerFrozenPainterACT"
    dev = torch.device("cuda")
    model = model.to(dev)
    opts = model.build_optimizers(1, 100)
    acc = SimpleNamespace(device=dev, backward=lambda l: l.backward(), clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    step = 0
    for _ in range(4):  # enough calls for samples to halt and be replaced (n_sup 3)
        batch = DataSample(**{k: v.clone() for k, v in vars(SUDOKU).items() if v is not None},
                           solution_mask=torch.rand(2, 81) < 0.3)
        losses, _, step = model.train_step([batch], acc, opts, None, step)
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert step == 4
