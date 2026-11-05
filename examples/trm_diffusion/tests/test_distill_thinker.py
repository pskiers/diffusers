# ABOUTME: Tests mode=thinker_act_distill against a real `_seex0` teacher checkpoint: the teacher sees the solved image,
# ABOUTME: its cached target equals the teacher run's own bridged output, and training moves only the student.
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
from models.utility_models import strip_compiled_prefix  # noqa: E402


R = "runs/local_rope_debug"
TEACHER = f"{R}/ckpt/seex0/A_20000.pt"
PAINTER = f"{R}/ckpt/painter_p16_all.pt"
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(TEACHER) and os.path.exists(PAINTER)),
    reason="needs a GPU, the A_20000 teacher and the p16_all painter (runs/local_rope_debug)",
)


def _overrides(name):
    return [l.strip() for l in open(f"{R}/{name}") if l.strip() and not l.startswith(("painter.checkpoint=", "eval"))]


def _build(overrides):
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(
            config_name="config", overrides=overrides + [f"painter.checkpoint={PAINTER}", "run.wandb_project=null"]
        )
    cfg.eval_callbacks = None
    return build_model(cfg, instantiate(cfg.diffusion)), cfg


def _distill_model():
    model, cfg = _build(
        _overrides("ov_all_act.txt") + ["mode=thinker_act_distill", f"+train.distill_teacher_checkpoint={TEACHER}"]
    )
    return model.cuda(), cfg


def test_teacher_input_and_target_match_the_teacher_run():
    model, cfg = _distill_model()
    model.train()
    puzzle_ds = instantiate(cfg.data.val_dataset)
    raw = collate_data_samples([puzzle_ds[i] for i in range(8)])
    torch.manual_seed(0)
    [d] = model.prep_mb_data([raw], torch.device("cuda"))
    s = d["sample"]
    assert torch.equal(s.spatial_conditions.cpu(), raw.spatial_conditions)  # the student reads the puzzle
    ts = model.teacher_sample(s)
    assert torch.allclose(ts.spatial_conditions.cpu(), raw.images, atol=1e-6)  # the teacher reads the solved image

    # the teacher run's own model, on the same x_t / t and its own (solved-image) condition
    teacher, _ = _build(_overrides("ov_all_seex0.txt"))
    ck = torch.load(TEACHER, map_location="cpu", weights_only=False)
    sd = {**strip_compiled_prefix(ck["model_state"]), **strip_compiled_prefix(ck["ema_state"])}
    teacher.load_state_dict(sd, strict=False)
    teacher.cuda().eval()
    with torch.no_grad():
        logits_out = []
        teacher.forward_with_carry(ts, logits_out=logits_out)
        expected = teacher.thinker_painter_translator._norm(logits_out[-1].float().transpose(1, 2)).transpose(1, 2)
    target = model.teacher_target(s)
    assert target.shape == (8, 81, 11)
    assert torch.allclose(target, expected, atol=1e-4), (target - expected).abs().max()

    tr_t, tr_s = teacher.thinker_painter_translator.state_dict(), model.thinker_painter_translator.state_dict()
    assert all(torch.equal(tr_t[k].cpu(), tr_s[k].cpu()) for k in tr_t)


def test_train_step_moves_only_the_student():
    model, cfg = _distill_model()
    model.train()
    assert not model.thinker_painter_translator.training and not model.teacher_thinker.training
    ds = instantiate(cfg.data.train_dataset)
    batch = collate_data_samples([ds[i] for i in range(16)])
    frozen = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if k.startswith(("teacher_", "thinker_painter_translator.", "painter."))
    }
    student = {
        k: v.clone() for k, v in model.state_dict().items() if k.startswith(("thinker.inner", "condition_encoder."))
    }
    acc = SimpleNamespace(
        device=torch.device("cuda"), backward=lambda l: l.backward(), clip_grad_norm_=torch.nn.utils.clip_grad_norm_
    )
    opts = model.build_optimizers(1, 100)
    losses = [model.train_step([batch], acc, opts, None, step)[0]["distill_loss"] for step in range(3)]
    print("distill losses", losses)
    assert all(torch.isfinite(torch.tensor(losses)))
    after = model.state_dict()
    assert all(torch.equal(frozen[k], after[k]) for k in frozen), [
        k for k in frozen if not torch.equal(frozen[k], after[k])
    ]
    moved = [k for k in student if not torch.equal(student[k], after[k])]
    assert any(k.startswith("thinker.inner") for k in moved) and any(k.startswith("condition_encoder.") for k in moved)
    for p in model.teacher_thinker.parameters():
        assert p.grad is None
