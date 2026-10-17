"""
tuning/driver.py — staged nuisance-hyperparameter tuning of a thinker
training command on SLURM.

The tuned parameters, in coordinates that are nearly independent:
    lr        thinker lr                    (thinker.optim_cfg.lr)
    r         encoder/translator lr / lr    (painter.optim_cfg.lr = r * lr)
    tau_frac  weight-decay timescale tau = 1 / (lr * wd) as a fraction of the
              full run's train.num_steps    (thinker.optim_cfg.weight_decay)
Warmup and schedule are fixed (the base command's). Every tuning run trains
the first `horizon` of the full run (train.stop_step, same lr schedule) and is
scored by tune/dloss (models/tune_callbacks.py; lower is better), averaged
over its last WINDOW evals. All runs share the base seed, so they see the same
data and noise.

Plans (--change):
    cold   range test -> lr x {1/30, 1/10, 1/3} of the divergence lr
           -> r x {1/3, 1, 3} -> lr x {1/2, 1, 2} -> tau_frac x {1/2, 1, 2} (medium horizon)
    small  lr x {1/2, 1, 2} around the --from family's best
    large  range test -> lr x {1/3, 1, 3} around the new divergence lr scaled
           by the --from family's best lr / divergence lr -> r x {1/3, 1, 3}
A sweep whose best value is at an edge, and clearly better than its
neighbour, is extended one step outward (at most MAX_EXTENSIONS times).
Within a tie (2 standard errors) the lowest lr, or the value closest to the
current one, wins.

Horizons were set from the Aug 2026 MNIST-sudoku search: one-factor lr
comparisons ranked correctly by 5% of the run and tau comparisons by 13%.

Usage (from examples/trm_diffusion; see slurm_scripts/wcss_tune.sh):
    python -m tuning.driver start --family NAME --change cold [--nodes 1] -- <train_trm.py overrides>
    python -m tuning.driver status NAME
    python -m tuning.driver skip NAME STAGE  # e.g. a sweep that can no longer change anything
    python -m tuning.driver advance NAME     # what each stage's advance job runs
status, skip and advance need only the standard library (e.g. the login node's python3).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNE_DIR = ROOT / "runs" / "tune"
BEST_DIR = ROOT / "tuning" / "best"

HORIZONS = {"short": 0.10, "medium": 0.30}
RANGE_TEST_BATCHES = 100  # ramp length, in batches (n_sup steps each)
EVALS_PER_RUN = 16
WINDOW = 3
TIE_SE = 2.0
MAX_EXTENSIONS = 2

PLANS = {
    "cold": [
        {"kind": "range_test"},
        {"kind": "sweep", "param": "lr", "center": "lr_div", "factors": [1 / 30, 1 / 10, 1 / 3]},
        {"kind": "sweep", "param": "r", "factors": [1 / 3, 1, 3]},
        {"kind": "sweep", "param": "lr", "factors": [1 / 2, 1, 2]},
        {"kind": "sweep", "param": "tau_frac", "factors": [1 / 2, 1, 2], "horizon": "medium"},
    ],
    "small": [
        {"kind": "sweep", "param": "lr", "factors": [1 / 2, 1, 2]},
    ],
    "large": [
        {"kind": "range_test"},
        {"kind": "sweep", "param": "lr", "center": "lr_div_scaled", "factors": [1 / 3, 1, 3]},
        {"kind": "sweep", "param": "r", "factors": [1 / 3, 1, 3]},
    ],
}
MANAGED_KEYS = (
    "run.output_dir", "train.seed", "train.num_steps", "train.stop_step", "thinker.optim_cfg.lr",
    "thinker.optim_cfg.weight_decay", "painter.optim_cfg.lr", "tune_callbacks", "lr_schedule",
    "eval.eval_every", "eval.save_every",
)


# ── Decisions (pure) ──────────────────────────────────────────────────────────


def score_run(records: list[dict]) -> tuple[float, float]:
    """(score, se): mean dloss and dloss_se over the last WINDOW evals; (inf, 0)
    for a run without finite results."""
    tail = records[-WINDOW:]
    if len(tail) < min(WINDOW, EVALS_PER_RUN) or not all(math.isfinite(r["dloss"]) for r in tail):
        return math.inf, 0.0
    return sum(r["dloss"] for r in tail) / len(tail), sum(r["dloss_se"] for r in tail) / len(tail)


def find_lr_div(records: list[dict], rise_se: float = 3.0, rise_frac: float = 0.5) -> tuple[float, bool]:
    """Range test: the first lr past the (3-point running median) minimum of
    dloss at which dloss rose back by max(rise_se SE, rise_frac |min|), or
    stopped being finite. (lr, True), or (last lr, False) if it never did."""
    lrs = [r["lrs"][0] for r in records]
    d = [r["dloss"] if math.isfinite(r["dloss"]) else math.inf for r in records]
    smooth = [sorted(d[max(0, i - 1) : i + 2])[len(d[max(0, i - 1) : i + 2]) // 2] for i in range(len(d))]
    i_min = min(range(len(d)), key=lambda i: smooth[i])
    threshold = smooth[i_min] + max(rise_se * records[i_min]["dloss_se"], rise_frac * abs(smooth[i_min]))
    for j in range(i_min + 1, len(d)):
        if not math.isfinite(d[j]) or smooth[j] > threshold:
            return lrs[j], True
    return lrs[-1], False


def decide_sweep(values: list[float], scores: list[float], ses: list[float], param: str, current: float):
    """(chosen value, value to add or None). values ascending."""
    best = min(range(len(values)), key=lambda i: scores[i])
    if not math.isfinite(scores[best]):
        raise RuntimeError("every run of the sweep failed")

    def tied(i, j):
        return scores[i] - scores[j] <= TIE_SE * math.hypot(ses[i], ses[j])

    for edge, inner in ((0, 1), (len(values) - 1, len(values) - 2)):
        if best == edge and len(values) > 1 and not tied(inner, edge):
            return values[best], values[edge] * values[edge] / values[inner]
    candidates = [i for i in range(len(values)) if tied(i, best)]
    if param == "lr":
        chosen = min(candidates, key=lambda i: values[i])
    else:
        chosen = min(candidates, key=lambda i: abs(math.log(values[i] / current)))
    return values[chosen], None


# ── State ─────────────────────────────────────────────────────────────────────


def _state_path(family: str) -> Path:
    return TUNE_DIR / family / "state.json"


def load_state(family: str) -> dict:
    return json.loads(_state_path(family).read_text())


def save_state(state: dict) -> None:
    path = _state_path(state["family"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)
    (path.parent / "status.txt").write_text(status_text(state) + "\n")


def compose_base(base: list[str]) -> dict:
    """The values the driver needs from the base command's composed config."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=base)
    return {
        "full_steps": int(cfg.train.num_steps),
        "n_sup": int(cfg.thinker.n_sup),
        "seed": int(cfg.train.seed),
        "lr": float(cfg.thinker.optim_cfg.lr),
        "painter_lr": float(cfg.painter.optim_cfg.lr),
        "wd": float(cfg.thinker.optim_cfg.weight_decay),
    }


def _key(spec: dict) -> str:
    return f"{spec['horizon']}_lr{spec['lr']:.3g}_r{spec['r']:.3g}_tau{spec['tau_frac']:.3g}_s{spec['seed']}"


def _align(steps: float, n_sup: int) -> int:
    return max(n_sup, round(steps / n_sup) * n_sup)


def run_overrides(state: dict, spec: dict, out_dir: Path) -> list[str]:
    lr, r, n_sup, full = spec["lr"], spec["r"], state["n_sup"], state["full_steps"]
    ov = [
        *state["base"],
        f"run.output_dir={out_dir}",
        f"train.seed={spec['seed']}",
        f"thinker.optim_cfg.lr={lr:.6g}",
        f"painter.optim_cfg.lr={r * lr:.6g}",
        "tune_callbacks=steering",
        "eval.save_every=1000000000",
    ]
    if spec["horizon"] == "range_test":
        steps = RANGE_TEST_BATCHES * n_sup
        return ov + [
            "lr_schedule=range_test",
            f"train.num_steps={steps}",
            "thinker.optim_cfg.weight_decay=0",
            f"eval.eval_every={2 * n_sup}",
            "tune_callbacks.0.max_batches=8",
            "tune_callbacks.0.recursion_gain=false",
            "tune_callbacks.0.grad_norms=false",
        ]
    stop = _align(HORIZONS[spec["horizon"]] * full, n_sup)
    return ov + [
        f"thinker.optim_cfg.weight_decay={1 / (lr * spec['tau_frac'] * full):.6g}",
        f"++train.stop_step={stop}",
        f"eval.eval_every={_align(stop / EVALS_PER_RUN, n_sup)}",
    ]


def read_records(run_dir: Path) -> list[dict]:
    path = run_dir / "tune_metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── SLURM ─────────────────────────────────────────────────────────────────────


def sbatch(state: dict, args: list[str]) -> str:
    if state["dry_run"]:
        print("[dry-run] sbatch", " ".join(args))
        return "0"
    out = subprocess.run(["sbatch", "--parsable", *args], check=True, capture_output=True, text=True)
    return out.stdout.strip().split(";")[0]


def scancel(state: dict, jobs: list[str]) -> None:
    jobs = [j for j in jobs if j and j != "0" and j != os.environ.get("SLURM_JOB_ID")]
    if jobs and not state["dry_run"]:
        subprocess.run(["scancel", *jobs], check=False)


def queued_jobs(jobs: list[str]) -> list[str]:
    """The given jobs that SLURM still lists (pending, running or completing)."""
    jobs = [j for j in jobs if j and j != "0"]
    if not jobs:
        return []
    out = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(jobs)], capture_output=True, text=True)
    return out.stdout.split()


def submit_round(state: dict, stage: dict) -> None:
    """Submit the stage's runs that don't exist yet, as a new round, plus a
    fallback advance job for when no run's own end-of-job advance gets to it
    (wcss_tune_job.sh runs `advance --round R --if-done` after training)."""
    state["round"] = state.get("round", 0) + 1
    stage["keys"], stage["jobs"] = [], []
    for spec in stage_specs(state, stage):
        key = _key(spec)
        stage["keys"].append(key)
        if key in state["runs"]:
            continue
        run_dir = TUNE_DIR / state["family"] / f"{state['family']}-{key}"  # dir name = wandb run name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "overrides.txt").write_text("\n".join(run_overrides(state, spec, run_dir)) + "\n")
        job = sbatch(state, [
            f"--nodes={state['nodes']}", f"--job-name=tune-{state['family']}",
            f"--output={run_dir}/slurm-%j.out", f"--error={run_dir}/slurm-%j.err",
            *state["sbatch_args"], state["job_script"], str(run_dir / "overrides.txt"),
            state["family"], str(state["round"]),
        ])
        state["runs"][key] = {"spec": spec, "dir": str(run_dir), "job": job}
        stage["jobs"].append(job)

    scancel(state, [state.get("advance_job", "")])  # the previous round's fallback, if still queued
    real = [j for j in stage["jobs"] if j != "0"]
    log_dir = TUNE_DIR / state["family"]
    state["advance_job"] = sbatch(state, [
        *([f"--dependency=afterany:{':'.join(real)}"] if real else []),
        f"--job-name=tune-{state['family']}-advance", f"--output={log_dir}/advance-%j.out",
        *state["advance_sbatch_args"], state["advance_script"], state["family"], str(state["round"]),
    ])


# ── Stages ────────────────────────────────────────────────────────────────────


def stage_specs(state: dict, stage: dict) -> list[dict]:
    p = state["params"]
    base = {"lr": p["lr"], "r": p["r"], "tau_frac": p["tau_frac"], "seed": state["seed"]}
    if stage["kind"] == "range_test":
        return [{**base, "horizon": "range_test"}]
    return [{**base, stage["param"]: v, "horizon": stage.get("horizon", "short")} for v in stage["values"]]


def start_stage(state: dict) -> None:
    stage = state["stages"][state["stage"]]
    if stage["kind"] == "sweep" and "values" not in stage:
        center = {
            "lr_div": state.get("lr_div"),
            "lr_div_scaled": state.get("lr_div", 0) * state.get("prior_lr_div_ratio", 0.1),
        }.get(stage.get("center"), state["params"][stage["param"]])
        stage["values"] = sorted(center * f for f in stage["factors"])
    submit_round(state, stage)


def next_stage(state: dict) -> None:
    """Move past the current stage and any skipped ones; start the next or finish."""
    state["stage"] += 1
    while state["stage"] < len(state["stages"]) and state["stages"][state["stage"]].get("skip"):
        print(f"stage {state['stage']} skipped")
        state["stage"] += 1
    if state["stage"] < len(state["stages"]):
        start_stage(state)
    else:
        scancel(state, [state.get("advance_job", "")])
        finish(state)


def finish(state: dict) -> None:
    state["done"] = True
    best = {
        "family": state["family"],
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": state["params"],
        "lr_div": state.get("lr_div"),
        "lr_div_ratio": state["params"]["lr"] / state["lr_div"] if state.get("lr_div") else None,
        "full_steps": state["full_steps"],
        "base": state["base"],
        "change": state["change"],
        "from": state["from"],
    }
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    (BEST_DIR / f"{state['family']}.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"done: {state['params']} -> {BEST_DIR / (state['family'] + '.json')}")


class _Lock:
    """Only one advance / skip at a time per family (mkdir is atomic on Lustre too).
    A crash inside can leave runs/tune/<family>/.lock behind: remove it by hand."""

    def __init__(self, family: str):
        self.path = TUNE_DIR / family / ".lock"

    def __enter__(self):
        try:
            self.path.mkdir()
        except FileExistsError:
            raise SystemExit(f"{self.path} exists: another advance is running (or crashed, then remove it)")
        return self

    def __exit__(self, *_):
        self.path.rmdir()


def advance(family: str, round_id: int | None = None, if_done: bool = False, force: bool = False) -> None:
    """Score the current round and move on. round_id: only act if it is still the
    current round (else a newer advance already did); without one (jobs from
    before rounds existed), only act if the session has no rounds yet. if_done:
    only act if no other job of the round is still queued (end-of-job calls).
    force: act regardless (manual resume after a failed stage)."""
    with _Lock(family):
        state = load_state(family)
        if state.get("done"):
            return
        stale = state.get("round") != round_id if round_id is not None else "round" in state
        if stale and not force:
            print(f"round {round_id} is not the current round ({state.get('round')}): nothing to do")
            return
        stage = state["stages"][state["stage"]]
        if if_done and not state["dry_run"]:
            others = [j for j in queued_jobs(stage.get("jobs", [])) if j != os.environ.get("SLURM_JOB_ID")]
            if others:
                print(f"round {round_id}: jobs {others} still queued, leaving the advance to the last one")
                return
        _advance(state, stage)
        save_state(state)


def _advance(state: dict, stage: dict) -> None:
    results = {k: score_run(read_records(Path(state["runs"][k]["dir"]))) for k in stage["keys"]}
    stage["results"] = {k: list(v) for k, v in results.items()}

    if stage["kind"] == "range_test":
        [key] = stage["keys"]
        records = read_records(Path(state["runs"][key]["dir"]))
        if not records:
            raise RuntimeError(f"range test {key} produced no records")
        state["lr_div"], stage["diverged"] = find_lr_div(records)
        print(f"range test: lr_div = {state['lr_div']:.3g}" + ("" if stage["diverged"] else " (no divergence in range)"))
    else:
        param = stage["param"]
        scores = [results[_key(s)] for s in stage_specs(state, stage)]
        chosen, extra = decide_sweep(stage["values"], [s for s, _ in scores], [e for _, e in scores],
                                     param, state["params"][param])
        if extra is not None and stage.get("extensions", 0) < MAX_EXTENSIONS:
            stage["extensions"] = stage.get("extensions", 0) + 1
            stage["values"] = sorted(stage["values"] + [extra])
            print(f"{param}: best at the edge ({chosen:.3g}), extending to {extra:.3g}")
            submit_round(state, stage)
            return
        state["params"][param] = chosen
        print(f"{param}: {chosen:.3g}")
    next_stage(state)


def skip(family: str, index: int) -> None:
    """Skip stage `index` (keeping the current params for it). If it is the
    running stage, its jobs are cancelled and the next stage starts now."""
    with _Lock(family):
        state = load_state(family)
        if state.get("done") or index < state["stage"]:
            raise SystemExit(f"stage {index} is already finished")
        stage = state["stages"][index]
        stage["skip"] = True
        if index == state["stage"]:
            scancel(state, stage.get("jobs") or [state["runs"][k]["job"] for k in stage.get("keys", [])])
            state["round"] = state.get("round", 0) + 1  # stale advance calls become no-ops
            next_stage(state)
        print(f"stage {index} ({stage['kind']} {stage.get('param', '')}) skipped")
        save_state(state)


def start(args) -> None:
    if _state_path(args.family).exists() and not args.force:
        raise SystemExit(f"{_state_path(args.family)} exists (use --force to restart the family)")
    base = [o for o in args.overrides if o.lstrip("+~").split("=")[0] not in MANAGED_KEYS]
    dropped = [o for o in args.overrides if o not in base]
    if dropped:
        print(f"ignoring overrides the driver sets itself (use --lr / --r / --tau-frac): {dropped}")
    composed = compose_base(base)
    full = composed["full_steps"]
    state = {
        "family": args.family, "change": args.change, "from": args.from_family, "base": base,
        "full_steps": full, "n_sup": composed["n_sup"], "seed": composed["seed"],
        "nodes": args.nodes, "dry_run": args.dry_run, "runs": {}, "stage": 0,
        "stages": [dict(s) for s in PLANS[args.change]],
        "job_script": args.job_script, "advance_script": args.advance_script,
        "sbatch_args": args.sbatch_args.split(), "advance_sbatch_args": args.advance_sbatch_args.split(),
    }
    if args.change == "cold":
        lr = args.lr or composed["lr"]
        state["params"] = {
            "lr": lr,
            "r": args.r or composed["painter_lr"] / lr,
            "tau_frac": args.tau_frac or 1 / (composed["lr"] * composed["wd"] * full),
        }
    else:
        prior = json.loads((BEST_DIR / f"{args.from_family or args.family}.json").read_text())
        state["params"] = dict(prior["params"])
        state["prior_lr_div_ratio"] = prior.get("lr_div_ratio") or 0.1
    print(f"{args.family}: {args.change} from {state['params']}, full run {full} steps")
    start_stage(state)
    save_state(state)


def status_text(state: dict) -> str:
    lines = [f"{state['family']} ({state['change']}): stage {state['stage']}/{len(state['stages'])}"
             + (" done" if state.get("done") else "") + f", round {state.get('round')}, params {state['params']}"]
    for i, stage in enumerate(state["stages"]):
        if stage.get("skip"):
            lines.append(f"  stage {i}: {stage['kind']} {stage.get('param', '')} (skipped)")
        if "keys" not in stage:
            continue
        lines.append(f"  stage {i}: {stage['kind']} {stage.get('param', '')}")
        for k in stage["keys"]:
            run = state["runs"][k]
            records = read_records(Path(run["dir"]))
            score, se = score_run(records)
            last = records[-1] if records else {}
            lines.append(f"    {k:<48} job {run['job']:>8}  evals {len(records):>3}  "
                         f"dloss {score:+.5f} ± {se:.5f}  step {last.get('step')}")
    if state.get("lr_div"):
        lines.append(f"  lr_div {state['lr_div']:.3g}")
    return "\n".join(lines)


def status(family: str) -> None:
    print(status_text(load_state(family)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start")
    p.add_argument("--family", required=True)
    p.add_argument("--change", choices=list(PLANS), required=True)
    p.add_argument("--from", dest="from_family", help="family whose best params small/large start from (default: --family)")
    p.add_argument("--lr", type=float, help="cold: starting lr (default: base config)")
    p.add_argument("--r", type=float, help="cold: starting r (default: base config's painter lr / lr)")
    p.add_argument("--tau-frac", type=float, help="cold: starting tau_frac (default: base config's)")
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--job-script", default=str(ROOT / "slurm_scripts" / "wcss_tune_job.sh"))
    p.add_argument("--advance-script", default=str(ROOT / "slurm_scripts" / "wcss_tune_advance.sh"))
    p.add_argument("--sbatch-args", default="", help="extra sbatch arguments for the training jobs")
    p.add_argument("--advance-sbatch-args", default="", help="extra sbatch arguments for the advance jobs")
    p.add_argument("--dry-run", action="store_true", help="print sbatch commands instead of submitting")
    p.add_argument("--force", action="store_true")
    p.add_argument("overrides", nargs="*")
    p = sub.add_parser("advance")
    p.add_argument("family")
    p.add_argument("--round", type=int, help="only act if this is still the current round")
    p.add_argument("--if-done", action="store_true", help="only act if no other job of the round is queued")
    p.add_argument("--force", action="store_true", help="act even if the round is not current (manual resume)")
    p = sub.add_parser("skip")
    p.add_argument("family")
    p.add_argument("stage", type=int, help="stage index, as in status.txt")
    sub.add_parser("status").add_argument("family")
    args = parser.parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "advance":
        advance(args.family, args.round, args.if_done, args.force)
    elif args.command == "skip":
        skip(args.family, args.stage)
    else:
        status(args.family)


if __name__ == "__main__":
    main()
