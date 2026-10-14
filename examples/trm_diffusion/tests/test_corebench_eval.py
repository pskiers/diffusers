"""T2I-CoReBench evaluation plumbing (eval/corebench_eval.py, CoReBenchEvalCallback)
without the VLM judge, which runs in its own environment (validated by a full
eval.py run instead): image layout per protocol, per-image seeding that does
not depend on batching or the GPU split, score aggregation, and the callback's
generate -> judge -> metrics -> cleanup flow with a stand-in judge.

Needs a GPU. Run from examples/trm_diffusion:  pytest tests/test_corebench_eval.py -s
"""

import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval.corebench_eval as cb  # noqa: E402
from models.eval_callbacks import CoReBenchEvalCallback  # noqa: E402
from models.minit2i import MiniT2IFlowScheduler, MiniT2IPainter  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICE = torch.device("cuda")
GEN = dict(steps=2, cfg_scale=6.0, seed=0)


@pytest.fixture(scope="module")
def painter():
    """fp32: in bf16 the result depends slightly on the batch composition
    (different kernels round differently, and CFG x6 amplifies it: mean ~1.5,
    max ~160 8-bit levels after 100 steps), which would hide what the seeding
    test checks. In fp32, batch 4 vs 1 differ by <1 level."""
    return MiniT2IPainter(scheduler=MiniT2IFlowScheduler(), train_cfg=None, eval_cfg=None, optim_cfg=None,
                          painter_dtype=None).to(DEVICE).eval()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/flan-t5-large", model_max_length=256)


@pytest.fixture(scope="module")
def prompts():
    all_prompts = cb.corebench_prompts()
    return [all_prompts[0], all_prompts[500]]  # one composition, one reasoning prompt


def test_prompts_and_layout(prompts):
    all_prompts = cb.corebench_prompts()
    assert len(all_prompts) == 1080 and sum(len(p.metadata["Checklist"]) for p in all_prompts) == 13536
    assert {p.group for p in all_prompts} == set(cb.COMPOSITION + cb.REASONING)
    p = prompts[0]
    assert cb.output_path("looped_dit", Path("I"), p, 3) == Path(f"I/{p.group}/{p.item_id}/00003.png")
    assert cb.output_path("official", Path("I"), p, 3, "m") == Path(f"I/m/{p.group}/{p.item_id}-3.png")
    # the official judge maps a file back to its item with '-'.join(name.split('-')[:3])
    assert "-".join(f"{p.item_id}-3.png".split("-")[:3]) == p.item_id


def test_to_pil_matches_looped_dit():
    x = torch.linspace(-1.2, 1.2, 3 * 4 * 4).view(1, 3, 4, 4)
    expected = (x.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()
    assert np.array_equal(np.asarray(cb.to_pil(x)[0]), expected)


def _read(root, protocol, prompts):
    return {
        (p.item_id, s): np.asarray(Image.open(cb.output_path(protocol, root, p, s)), dtype=np.int16)
        for p in prompts for s in range(2)
    }


def test_seeding_independent_of_batching_and_ranks(painter, tokenizer, prompts, tmp_path):
    """Same images whether made 4 per batch on one GPU or 1 per batch split
    over two "ranks" — per-image seeds, unlike Looped-DiT's rank-dependent ones.
    (Batch size can change float rounding, hence a 1-level tolerance.)"""
    a, b = tmp_path / "a", tmp_path / "b"
    n = cb.generate_corebench_images(painter, a, tokenizer, DEVICE, samples_per_prompt=2, batch_size=4,
                                     prompts=prompts, **GEN)
    assert n == 4
    for rank in (0, 1):
        cb.generate_corebench_images(painter, b, tokenizer, DEVICE, rank=rank, world_size=2, samples_per_prompt=2,
                                     batch_size=1, prompts=prompts, **GEN)
    ia, ib = _read(a, "looped_dit", prompts), _read(b, "looped_dit", prompts)
    diffs = [np.abs(ia[k] - ib[k]).max() for k in ia]
    print(f"\nmax per-image difference (8-bit levels): {diffs}")
    assert max(diffs) <= 1
    assert np.abs(ia[(prompts[0].item_id, 0)] - ia[(prompts[0].item_id, 1)]).mean() > 10  # samples differ
    # existing images are skipped on a rerun
    assert cb.generate_corebench_images(painter, a, tokenizer, DEVICE, samples_per_prompt=2, prompts=prompts, **GEN) == 0


def test_prompt_subset():
    sub = cb.corebench_prompts(per_dimension=2)
    assert len(sub) == 24 and all(sum(p.group == g for p in sub) == 2 for g in cb.COMPOSITION + cb.REASONING)


def test_summarize_scores():
    dims = {t: float(i) for i, t in enumerate(cb.COMPOSITION + cb.REASONING)}
    m = cb.summarize_scores({"dimensions": dims, "overall": 42.0})
    assert m["overall"] == 42.0
    assert m["composition"] == np.mean([0, 1, 2, 3]) and m["reasoning"] == np.mean(range(4, 12))
    assert m["mean_of_dimensions"] == np.mean(range(12)) and m["R-RR"] == 11.0


def test_callback_flow_with_stand_in_judge(painter, tmp_path, monkeypatch):
    """The whole callback on one rank: generation, the judge subprocess (a
    stand-in that checks the images and writes scores.json), metrics, cleanup."""
    fake_judge = tmp_path / "fake_judge_python"
    fake_judge.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys
        from pathlib import Path
        args = sys.argv[2:]  # sys.argv[1] is eval/corebench_judge.py
        get = lambda flag: args[args.index(flag) + 1]
        n = len(list(Path(get("--image-dir")).rglob("*.png")))
        assert n == 24, n  # 1 prompt per dimension x 2 samples
        assert len(list(Path(get("--data")).glob("*.json"))) == 12  # the subset's prompt files
        assert get("--protocol") == "looped_dit" and get("--tensor-parallel-size") == "1"
        dims = {{t: 50.0 for t in {list(cb.COMPOSITION + cb.REASONING)}}}
        out = Path(get("--out-dir")); out.mkdir(parents=True, exist_ok=True)
        (out / "scores.json").write_text(json.dumps({{"dimensions": dims, "overall": 50.0}}))
        """))
    fake_judge.chmod(0o755)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    callback = CoReBenchEvalCallback(every_n_steps=100, out_dir=str(tmp_path / "corebench"), judge_python=str(fake_judge),
                                     samples_per_prompt=2, steps=2, batch_size=24, prompts_per_dimension=1,
                                     poll_seconds=0.1)
    accelerator = SimpleNamespace(process_index=0, num_processes=1, device=DEVICE, is_main_process=True)
    assert callback(painter, None, accelerator, step=50) == {}  # not due yet
    metrics = callback(painter, None, accelerator, step=100)
    assert metrics["corebench/overall"] == 50.0 and metrics["corebench/reasoning"] == 50.0
    assert len(metrics) == 4 + 12
    step_dir = tmp_path / "corebench" / "step00000100"
    assert not (step_dir / "images").exists()  # keep_images=False
    assert "scores" in json.loads((step_dir / "judged.json").read_text())
