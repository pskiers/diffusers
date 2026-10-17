"""tuning/driver.py decisions and state machine, on a synthetic objective
(no SLURM, no GPU): sbatch is dry-run and the runs' tune_metrics.jsonl files
are written by the test.

Run from examples/trm_diffusion:  pytest tests/test_tuning_driver.py
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tuning import driver  # noqa: E402

OPT = {"lr": 3e-4, "r": 2.0, "tau_frac": 0.3}
SE = 1e-4


def objective(lr, r, tau_frac):
    return (
        -0.05
        + 0.01 * math.log10(lr / OPT["lr"]) ** 2
        + 0.005 * math.log10(r / OPT["r"]) ** 2
        + 0.005 * math.log10(tau_frac / OPT["tau_frac"]) ** 2
    )


def fake_records(spec):
    if spec["horizon"] == "range_test":  # improves, then blows up past 3e-3
        lrs = [spec["lr"] * 1e-2 * 1e4 ** (i / 49) for i in range(50)]
        dloss = [-0.05 * min(1.0, lr / 1e-5) if lr < 3e-3 else 1.0 for lr in lrs]
        return [{"step": i, "lrs": [lr], "dloss": d, "dloss_se": SE} for i, (lr, d) in enumerate(zip(lrs, dloss))]
    d = objective(spec["lr"], spec["r"], spec["tau_frac"])
    return [{"step": i, "lrs": [spec["lr"]], "dloss": d, "dloss_se": SE} for i in range(driver.EVALS_PER_RUN)]


@pytest.fixture
def tune_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "TUNE_DIR", tmp_path / "tune")
    monkeypatch.setattr(driver, "BEST_DIR", tmp_path / "best")
    monkeypatch.setattr(driver, "compose_base", lambda base: {
        "full_steps": 80000, "n_sup": 16, "seed": 42, "lr": 1e-4, "painter_lr": 1e-4, "wd": 1.0,
    })
    return tmp_path


def start_session(family, change, from_family=None, tau_frac=0.25):
    driver.start(SimpleNamespace(
        family=family, change=change, from_family=from_family, lr=None, r=None, tau_frac=tau_frac,
        nodes=2, job_script="job.sh", advance_script="advance.sh", sbatch_args="", advance_sbatch_args="",
        dry_run=True, force=False, overrides=["experiment=x", "thinker.optim_cfg.lr=5e-4"],
    ))


def finish_round(family):
    """Write the current round's results; returns the round."""
    state = driver.load_state(family)
    for key in state["stages"][state["stage"]]["keys"]:
        run = state["runs"][key]
        path = Path(run["dir"]) / "tune_metrics.jsonl"
        if not path.exists():
            path.write_text("".join(json.dumps(r) + "\n" for r in fake_records(run["spec"])))
    return state["round"]


def run_session(family, change, from_family=None, tau_frac=0.25, skip=()):
    start_session(family, change, from_family, tau_frac)
    for index in skip:
        driver.skip(family, index)
    for _ in range(20):
        if driver.load_state(family).get("done"):
            break
        driver.advance(family, finish_round(family), if_done=True)
    state = driver.load_state(family)
    assert state.get("done")
    return state, json.loads((driver.BEST_DIR / f"{family}.json").read_text())


def test_cold_then_small_then_large(tune_dirs, capsys):
    state, best = run_session("fam", "cold")
    p = best["params"]
    assert best["lr_div"] == pytest.approx(3e-3, rel=0.25)
    assert p["lr"] == pytest.approx(3e-4, rel=0.15)  # lr_div / 10, confirmed by the refine stage
    assert p["r"] == pytest.approx(3.0)  # edge -> extended to 9, which is worse
    assert p["tau_frac"] == pytest.approx(0.25)
    assert "thinker.optim_cfg.lr=5e-4" not in state["base"]  # managed keys are dropped from the base
    # reuse: r's middle value and the refine's middle value are the previous stage's best run
    assert len(state["runs"]) == 1 + 3 + 3 + 2 + 3

    state, small = run_session("fam_small", "small", from_family="fam")
    assert small["params"] == pytest.approx(p) and len(state["runs"]) == 3

    state, large = run_session("fam_large", "large", from_family="fam")
    assert large["params"]["lr"] == pytest.approx(p["lr"], rel=0.15)
    assert state["stages"][1]["values"][1] == pytest.approx(best["lr_div_ratio"] * large["lr_div"])


def test_rounds_make_advance_idempotent(tune_dirs):
    start_session("fam", "cold")
    first = finish_round("fam")
    driver.advance("fam", first)  # range test -> lr sweep (a new round)
    state = driver.load_state("fam")
    assert state["stage"] == 1 and state["round"] == first + 1
    driver.advance("fam", first)  # the fallback job of the old round: no-op
    driver.advance("fam")  # a job from before rounds existed: no-op once rounds exist
    assert driver.load_state("fam")["stage"] == 1
    (driver.TUNE_DIR / "fam" / ".lock").mkdir()  # another advance in progress
    with pytest.raises(SystemExit):
        driver.advance("fam", first + 1)


def test_skip(tune_dirs):
    # a future stage: the lr refine never runs, and the result is the same here
    state, best = run_session("fam", "cold", skip=[3])
    assert state["stages"][3].get("skip") and "keys" not in state["stages"][3]
    assert len(state["runs"]) == 1 + 3 + 3 + 3
    assert best["params"]["lr"] == pytest.approx(3e-4, rel=0.15)
    # the running stage: moves straight on to the next one
    start_session("fam2", "cold")
    driver.advance("fam2", finish_round("fam2"))
    driver.skip("fam2", 1)
    state = driver.load_state("fam2")
    assert state["stage"] == 2 and state["stages"][2]["param"] == "r"
    with pytest.raises(SystemExit):
        driver.skip("fam2", 0)  # already finished


def test_run_overrides(tune_dirs):
    state = {"base": ["experiment=x"], "n_sup": 16, "full_steps": 80000}
    spec = {"lr": 2e-4, "r": 3.0, "tau_frac": 0.25, "seed": 1, "horizon": "short"}
    ov = dict(o.split("=", 1) for o in driver.run_overrides(state, spec, Path("out")))
    assert float(ov["painter.optim_cfg.lr"]) == pytest.approx(6e-4)
    assert 1 / (2e-4 * float(ov["thinker.optim_cfg.weight_decay"])) == pytest.approx(0.25 * 80000, rel=1e-5)
    assert int(ov["++train.stop_step"]) == 8000 and int(ov["eval.eval_every"]) % 16 == 0


def test_find_lr_div_without_divergence():
    records = [{"lrs": [1e-6 * 10**i], "dloss": -0.01 * i, "dloss_se": 1e-4} for i in range(5)]
    assert driver.find_lr_div(records) == (1e-2, False)


def test_decide_sweep_ties():
    # all tied: lr takes the lowest value, other params the one closest to the current value
    assert driver.decide_sweep([1, 2, 4], [0.0, 0.0, 0.0], [1, 1, 1], "lr", 2) == (1, None)
    assert driver.decide_sweep([1, 2, 4], [0.0, 0.0, 0.0], [1, 1, 1], "r", 2) == (2, None)
    # clear best at an edge -> extend outward geometrically
    assert driver.decide_sweep([1, 2, 4], [3.0, 2.0, 0.0], [1e-3] * 3, "lr", 2) == (4, 8)
    # a failed (inf) run never wins
    assert driver.decide_sweep([1, 2, 4], [0.0, -1.0, math.inf], [1e-3] * 3, "lr", 2) == (2, None)
