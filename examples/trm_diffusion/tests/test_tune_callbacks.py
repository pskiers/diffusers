"""Tuning-run pieces on the real MiniT2I painter (tiny thinker):

- SteeringDiagnosticsCallback: exactly no effect and no thinker gradient at the
  translator's zero init; nonzero once it moves; the same metrics sharded over
  2 processes as on one (tests/tune_callback_check.py);
- a train_trm.py run with tune_callbacks=steering, lr_schedule=range_test and
  train.stop_step: stops early, the lr follows the ramp, and the JSON lines the
  driver reads are written.

Needs 2 GPUs and the Std-120K fixture (tests/test_std120k_dataset.py builds it).
Run from examples/trm_diffusion:  pytest tests/test_tune_callbacks.py -s
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.tune_callback_check import TINY_THINKER  # noqa: E402

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
ENV = {**os.environ, "WANDB_MODE": "disabled"}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    fixture = ROOT / "data" / "std120k_testfixture"
    if not fixture.exists():
        pytest.skip("run tests/test_std120k_dataset.py first (builds the Std-120K fixture)")
    from datasets.std120k_dataset import LocalRawFiles, Std120KDataset

    root = tmp_path_factory.mktemp("data") / "std120k"
    Std120KDataset(root=root, raw_files=LocalRawFiles(fixture), build_workers=4)
    return root


def test_callback_zero_init_and_sharding(data_root, tmp_path):
    out = tmp_path / "metrics.json"
    cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", "2", "tests/tune_callback_check.py",
           str(data_root), str(out)]
    res = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    m = json.loads(out.read_text())
    print(json.dumps(m, indent=1))

    zero = m["zero_init"]
    assert zero["tune/dloss"] == 0.0 and zero["tune/steer_ratio"] == 0.0
    assert zero["tune/grad_norm_thinker"] == 0.0 and zero["tune/grad_norm_encoder"] > 0

    sharded, single = m["sharded"], m["single"]
    assert sharded["tune/steer_ratio"] > 0 and sharded["tune/grad_norm_thinker"] > 0
    for key in ("tune/dloss", "tune/loss", "tune/loss_painter", "tune/recursion_gain", "tune/steer_ratio",
                "tune/dloss_t0", "tune/dloss_t3", "tune/param_norm_thinker"):
        assert sharded[key] == pytest.approx(single[key], rel=1e-4, abs=1e-7), key


def test_train_script_range_test_and_stop_step(data_root, tmp_path):
    run = tmp_path / "run"
    cmd = [
        sys.executable, "train_trm.py", "experiment=std120k_thinker_minit2i", f"run.output_dir={run}",
        f"data.root={data_root}", "data.num_workers=2", "data.val_dataset.max_samples=8", "~eval_callbacks",
        "tune_callbacks=steering", "tune_callbacks.0.max_batches=2", "lr_schedule=range_test",
        *TINY_THINKER, "train.batch_size=2", "train.gradient_accumulation_steps=1",
        "train.num_steps=40", "++train.stop_step=8", "eval.eval_every=4", "eval.save_every=100", "eval.log_every=2",
        "eval.batch_size=2",
    ]
    res = subprocess.run(cmd, cwd=ROOT, env={**ENV, "CUDA_VISIBLE_DEVICES": "0"}, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    records = [json.loads(line) for line in (run / "tune_metrics.jsonl").read_text().splitlines()]
    print(json.dumps(records[-1], indent=1))
    assert [r["step"] for r in records] == [4, 8]
    assert torch.load(run / "checkpoint_final.pt", map_location="cpu", weights_only=True)["step"] == 8
    for r in records:  # lr of the step's last optimizer step: 1e-4 * 1e-2 * 1e4 ** (step / num_steps)
        expected = 1e-4 * 1e-2 * 1e4 ** ((r["step"] - 1) / 40)
        assert r["lrs"][0] == pytest.approx(expected, rel=1e-6) and r["lrs"][-1] == pytest.approx(expected, rel=1e-6)
    assert "adam_eps_frac_1" in records[-1] and 0.0 <= records[-1]["adam_eps_frac_1"] <= 1.0
