# ABOUTME: Tests train.thinker_sees_x0 (the thinker reads the clean target, the painter's input is unchanged): on Sudoku
# ABOUTME: it equals the dataset-level `_seex0` input; on CLEVR x0 replaces x_t; a CLEVR distill teacher matches its run.
import dataclasses
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.data_sample import collate_data_samples  # noqa: E402
from eval.clevr_eval_callbacks import _scene_to_data_sample  # noqa: E402
from factory import build_model  # noqa: E402


R = "runs/local_rope_debug"
PAINTER = f"{R}/ckpt/painter_p16_all.pt"
CLEVR_ROOT = "cache_dir"
DEV = torch.device("cuda")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
needs_sudoku = pytest.mark.skipif(not os.path.exists(PAINTER), reason="needs the p16_all painter (runs/local_rope_debug)")
needs_clevr = pytest.mark.skipif(
    not os.path.exists(f"{CLEVR_ROOT}/CLEVR_v1.0/scenes"), reason="needs CLEVR in cache_dir/CLEVR_v1.0"
)
CLEVR = [
    "experiment=clevr_thinker_mmdit_cnn",
    f"data.clevr_root={CLEVR_ROOT}",
    "mode=thinker_act",
    "thinker.with_halt_head=true",
    "painter.painter_dtype=null",
    "+train.force_unconditional_painter=true",
]


def _build(overrides, painter=PAINTER):
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides + [f"painter.checkpoint={painter}", "run.wandb_project=null"])
    cfg.eval_callbacks = None
    return build_model(cfg, instantiate(cfg.diffusion)).to(DEV), cfg


def _sudoku_overrides(name):
    return [l.strip() for l in open(f"{R}/{name}") if l.strip() and not l.startswith(("painter.checkpoint=", "eval"))]


def _batch(cfg, n, **ds_kwargs):
    ds = instantiate(cfg.data.val_dataset, **ds_kwargs)
    return collate_data_samples([ds[i] for i in range(n)])


@needs_gpu
@needs_sudoku
def test_sudoku_condition_mode_equals_dataset_seex0(monkeypatch):
    # TF32 convolutions (cuDNN's default) would turn the inputs' float32 rounding difference into ~1e-3 relative
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    switch, cfg = _build(_sudoku_overrides("ov_all_act.txt") + ["+train.thinker_sees_x0=condition"])
    seex0, _ = _build(_sudoku_overrides("ov_all_seex0.txt"))
    seex0.load_state_dict(switch.state_dict())
    switch.eval(), seex0.eval()
    puzzles, solved = _batch(cfg, 8), _batch(cfg, 8, condition_is_solution=True)
    assert not torch.equal(puzzles.spatial_conditions, solved.spatial_conditions)

    torch.manual_seed(0)
    [d] = switch.prep_mb_data([puzzles], DEV)
    s = d["sample"]
    assert torch.equal(s.spatial_conditions.cpu(), puzzles.spatial_conditions)  # the painter's input: the puzzle
    assert torch.allclose(s.thinker_x0.cpu(), solved.spatial_conditions, atol=1e-6)  # the thinker's: the solution
    s_seex0 = dataclasses.replace(s, spatial_conditions=solved.spatial_conditions.to(DEV), thinker_x0=None)
    with torch.no_grad():  # thinker_x0 went [0, 1] -> model range -> [0, 1]: equal up to float rounding
        enc, enc_seex0 = switch._encode_condition(s), seex0._encode_condition(s_seex0)
        assert torch.allclose(enc, enc_seex0, atol=1e-5), (enc - enc_seex0).abs().max()

        e = switch._batch_to_sample(puzzles, DEV)  # eval: raw [0, 1] images
        e_seex0 = seex0._batch_to_sample(solved, DEV)
        assert torch.equal(e.spatial_conditions, puzzles.spatial_conditions.to(DEV))
        assert torch.allclose(e.thinker_x0, e_seex0.spatial_conditions, atol=1e-6)
        e = dataclasses.replace(e, x_noisy=s.x_noisy, timesteps=s.timesteps)
        e_seex0 = dataclasses.replace(e_seex0, x_noisy=s.x_noisy, timesteps=s.timesteps)
        enc, enc_seex0 = switch._encode_condition(e), seex0._encode_condition(e_seex0)
        assert torch.allclose(enc, enc_seex0, atol=1e-5), (enc - enc_seex0).abs().max()

        null, null_seex0 = switch.null_condition_sample(e), seex0.null_condition_sample(e_seex0)
        assert null.thinker_x0.abs().max() == 0
        assert torch.equal(switch._encode_condition(null), seex0._encode_condition(null_seex0))


@needs_gpu
@needs_clevr
def test_clevr_x_noisy_mode_reads_x0_in_place_of_x_t():
    switch, cfg = _build(CLEVR + ["+train.thinker_sees_x0=x_noisy"], painter="null")
    switch.train()
    raw = _batch(cfg, 4)
    torch.manual_seed(0)
    [d] = switch.prep_mb_data([raw], DEV)
    s = d["sample"]
    assert torch.equal(s.thinker_x0, s.images) and not torch.equal(s.thinker_x0, s.x_noisy)

    enc = switch._encode_condition(s)
    keys = switch.condition_encoder.condition_keys
    direct = switch.condition_encoder(getattr(s, keys[0]), x_noisy=s.images, timesteps=s.timesteps).enc_emb
    reads_x_t = switch.condition_encoder(getattr(s, keys[0]), x_noisy=s.x_noisy, timesteps=s.timesteps).enc_emb
    assert torch.equal(enc, direct) and not torch.allclose(enc, reads_x_t)
    enc.float().pow(2).mean().backward()
    cnn_grads = [p.grad for n, p in switch.condition_encoder.named_parameters() if "latent_encoder" in n]
    assert cnn_grads and all(g is not None for g in cnn_grads) and sum(g.abs().sum() for g in cnn_grads) > 0

    e = switch._batch_to_sample(raw, DEV)  # eval: from raw pixels to the model range, as in training
    assert torch.allclose(e.thinker_x0, s.images, atol=1e-6)
    assert switch.null_condition_sample(e).thinker_x0.abs().max() == 0

    # the CLEVR eval callbacks build samples from scenes: with_target_image gives the dataset's own image
    ds = instantiate(cfg.data.val_dataset)
    cb = _scene_to_data_sample(
        ds.scenes[0], cfg.data.mode, image_dir=ds.image_dir, transform=ds.transform, with_target_image=True
    )
    item = ds[0]  # (its object tokens differ: "reduced" mode draws a random relation subset on every call)
    assert torch.equal(cb.images, item.images)
    cb = switch.with_eval_thinker_x0(collate_data_samples([cb]).to(DEV))
    assert torch.allclose(cb.thinker_x0, switch.painter.to_model_range(item.images[None].to(DEV)), atol=0)


@needs_gpu
@needs_clevr
def test_clevr_distill_teacher_matches_its_run(tmp_path):
    teacher, cfg = _build(CLEVR + ["+train.thinker_sees_x0=x_noisy"], painter="null")
    for m in teacher.modules():  # the translator bridge's statistics, as a trained teacher's
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.normal_(), m.running_var.uniform_(0.5, 2.0)
    ckpt = tmp_path / "teacher.pt"
    torch.save({"model_state": teacher.state_dict(), "ema_state": None}, ckpt)

    student, _ = _build(
        CLEVR
        + [
            "mode=thinker_act_distill",
            f"+train.distill_teacher_checkpoint={ckpt}",
            "+train.distill_teacher_sees_x0=x_noisy",
        ],
        painter="null",
    )
    student.painter.load_state_dict(teacher.painter.state_dict())
    student.train()
    torch.manual_seed(0)
    [d] = student.prep_mb_data([_batch(cfg, 4)], DEV)
    s = d["sample"]
    assert s.thinker_x0 is None  # the student reads x_t
    ts = student.teacher_sample(s)
    assert torch.equal(ts.thinker_x0, s.images)

    teacher.eval()
    with torch.no_grad():
        logits_out = []
        teacher.forward_with_carry(ts, logits_out=logits_out)
        expected = teacher.thinker_painter_translator._norm(logits_out[-1].float().transpose(1, 2)).transpose(1, 2)
    target = student.teacher_target(s)
    assert torch.allclose(target, expected, atol=1e-4), (target - expected).abs().max()
