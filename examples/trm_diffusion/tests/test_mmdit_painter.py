"""Generic MM-DiT painter (models/mmdit_painter.py) and its thinker steering, on
tiny models (CPU is enough; models/minit2i.py's released-weights path is checked
by tests/test_minit2i_thinker.py):

- the fused blocks with cond_rope "1d" against the vendored MMJiT.forward;
- stage-1 training reaches every painter parameter;
- condition dropout in training == null_condition_sample (what CFG sampling uses);
- steering, image and condition stream: a no-op at initialization, and
  gradients reaching the thinker's output at both kinds of position;
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


@pytest.mark.parametrize("kind", ["patch", "embedding"])
def test_steering_noop_at_init_and_gradients(kind):
    painter = _painter(kind, freeze=True)
    torch.manual_seed(3)
    with torch.no_grad():  # as after stage 1: the zero-initialized output layer has moved
        painter.net.final_layer.linear.weight.normal_(std=0.02)
    sample = painter._prepare_training_sample(_batch(kind), DEV)
    n_cond = sample.painter_hidden.shape[1] - 16
    translator = PainterResidualTranslator(in_dim=8, painter_dim=64, grid=2, painter_grid=4, num_blocks=2, steer_cond=True)
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


SUDOKU = DataSample(images=torch.rand(2, 1, 144, 144), spatial_conditions=torch.rand(2, 1, 144, 144),
                    solution=torch.randint(0, 9, (2, 81)))
CLEVR = DataSample(images=torch.rand(2, 3, 256, 256) * 2 - 1, embedding_conditions=torch.randn(2, 10, 55),
                   embedding_mask=torch.tensor([[1.0] * 4 + [0.0] * 6] * 2))


@pytest.mark.parametrize(
    "painter_exp, thinker_exp, batch, extra",
    [
        ("mnist_mmdit_painter", "mnist_thinker_mmdit", SUDOKU, ["translator.steer_cond=true"]),
        ("mnist_mmdit_painter", "mnist_thinker_mmdit_cnn", SUDOKU, []),
        ("clevr_mmdit_painter", "clevr_thinker_mmdit", CLEVR, ["translator.steer_cond=true"]),
    ],
)
def test_experiment_configs_build(tmp_path, painter_exp, thinker_exp, batch, extra):
    painter = _build([f"experiment={painter_exp}", *TINY_PAINTER])
    ckpt = tmp_path / "painter.pt"
    torch.save({"model_state": painter.state_dict()}, ckpt)

    model = _build([f"experiment={thinker_exp}", f"painter.checkpoint={ckpt}", *TINY_PAINTER, *TINY_THINKER, *extra])
    assert not any(p.requires_grad for p in model.painter.parameters())
    [d] = model.prep_mb_data([batch], DEV)
    with torch.no_grad():
        steered = model(d["sample"]).pred
        plain = model.painter(d["sample"]).pred
    assert torch.equal(steered, plain)  # zero-initialized translator
    noise_pred, logits, _, _ = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
    assert noise_pred.shape == batch.images.shape
