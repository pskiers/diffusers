# ABOUTME: Tests the one-stage teacher/student thinkers on the conditional p16 Sudoku painter: _mix (a per-episode
# ABOUTME: fraction of samples reads the solution), _pair (gradient routing of the matching loss) and GRAM's KL budget.
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.data_sample import collate_data_samples  # noqa: E402
from factory import build_model  # noqa: E402
from models.gram_thinkers import _LossPlusKL  # noqa: E402


PAINTER = "runs/local_rope_debug/ckpt/painter_p16.pt"
DEV = torch.device("cuda")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(PAINTER)), reason="needs a GPU and the p16 painter"
)
# mnist_thinker_p16_cond_act_tn2_tx0 as wcss_mmdit.sh builds it, painter in fp32
BASE = [
    "experiment=mnist_thinker_mmdit_cnn",
    "painter.net.patch_size=16",
    "mode=thinker_act",
    "thinker.with_halt_head=true",
    "translator.input_noise_std=2",
    "+train.thinker_sees_x0=condition",
    "painter.steer_from_block=1",
    "painter.steer_to_block=null",
    "translator.num_blocks=7",
    "translator.painter_grid=9",
    "translator.hidden_dim=384",
    "translator.with_timestep_emb=true",
    "painter.label_drop_rate=0.0",
    "painter.painter_dtype=null",
]


def _build(extra):
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=BASE + extra + [f"painter.checkpoint={PAINTER}", "run.wandb_project=null"])
    cfg.eval_callbacks = None
    model = build_model(cfg, instantiate(cfg.diffusion)).to(DEV)
    raw = collate_data_samples([instantiate(cfg.data.val_dataset)[i] for i in range(16)])
    return model, raw


def test_mix_reads_the_solution_for_a_fraction_of_samples_in_training_only():
    model, raw = _build(["+train.thinker_x0_prob=0.5"])
    model.train()
    torch.manual_seed(0)
    [d] = model.prep_mb_data([raw], DEV)
    s = d["sample"]
    solved = model.painter.decode_for_eval(s.images)
    is_solved = [torch.equal(s.thinker_x0[i], solved[i]) for i in range(16)]
    is_puzzle = [torch.equal(s.thinker_x0[i], s.spatial_conditions[i]) for i in range(16)]
    assert all(a != b for a, b in zip(is_solved, is_puzzle))  # each sample reads exactly one of the two
    assert 0 < sum(is_solved) < 16
    assert torch.equal(s.spatial_conditions.cpu(), raw.spatial_conditions)  # the painter keeps the puzzle

    model.eval()  # eval: every sample reads the puzzle
    [d] = model.prep_mb_data([raw], DEV)
    e = d["sample"]
    assert e.thinker_x0 is None
    with torch.no_grad():
        assert torch.equal(model._encode_condition(e), model._encode_with(model.condition_encoder, e, None))
    assert model.with_eval_thinker_x0(model._batch_to_sample(raw, DEV)).thinker_x0 is None


def test_pair_matching_loss_trains_only_the_student_pass():
    model, raw = _build(["mode=thinker_act_paired", "+train.pair_match_weight=0.03"])
    model.train()
    torch.manual_seed(0)
    [d] = model.prep_mb_data([raw], DEV)
    s = d["sample"]
    z = (d["z_H"], d["z_L"])
    tr = list(model.thinker_painter_translator.parameters())
    th = list(model.thinker.parameters())

    painter_loss, _, match, _, _, _ = model.pair_step(s, z, z)
    match.mean().backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in tr)  # no route into the translator / bridge
    assert sum(p.grad.abs().sum() for p in th if p.grad is not None) > 0

    # the translator's output layers start at zero, which blocks the painter's gradient to the thinker at step 0
    zero_init = [p for p in tr if p.abs().max() == 0]
    assert zero_init
    with torch.no_grad():
        for p in zero_init:
            p.normal_(std=1e-2)
    model.zero_grad()
    painter_loss, _, match, _, _, _ = model.pair_step(s, z, z)
    painter_loss.backward()  # the teacher's painter loss trains the translator and the thinker
    assert sum(p.grad.abs().sum() for p in tr if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in th if p.grad is not None) > 0

    model.eval()  # eval reads the puzzle
    with torch.no_grad():
        assert torch.equal(model._encode_condition(s), model._encode_with(model.condition_encoder, s, None))
        assert not torch.allclose(model._encode_condition(s), model._encode_with(model.condition_encoder, s, "condition"))


def test_gram_kl_budget_moves_beta_toward_the_target():
    class _Thinker:
        def __init__(self, kl):
            self.kl, self._kl_terms = kl, ()

        def kl_loss(self, balance, free_bits, per_channel=True):
            assert not per_channel  # the budget is in nats per token
            return torch.full((4,), self.kl), {"gram_kl_nats_per_token": self.kl}

    inner = lambda noise_pred, logits, sample: (torch.tensor(0.0), {})  # noqa: E731
    loss_fn = _LossPlusKL(inner, beta=0.01, balance=0.8, free_bits=0.5, kl_target=2.2, kl_lr=0.01)
    assert "log_beta" in loss_fn.state_dict()
    for kl, rises in ((40.0, True), (0.5, False), (0.0, False)):  # KL is exactly 0 at init
        object.__setattr__(loss_fn, "thinker", _Thinker(kl))
        before = loss_fn.log_beta.item()
        loss, stats = loss_fn(None, None, None)
        assert math.isclose(loss.item(), math.exp(before) * kl, rel_tol=1e-5)  # this step's loss uses the old beta
        assert math.isclose(loss_fn.log_beta.item() - before, 0.01 * (kl - 2.2) / 2.2, rel_tol=1e-4)
        assert (loss_fn.log_beta.item() > before) == rises
