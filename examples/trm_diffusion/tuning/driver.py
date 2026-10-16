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
    python -m tuning.driver advance NAME     # what each stage's advance job runs
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


def submit_runs(state: dict, specs: list[dict]) -> list[str]:
    """Submit the specs not already in state["runs"]; returns all their keys."""
    keys = []
    for spec in specs:
        key = _key(spec)
        keys.append(key)
        if key in state["runs"]:
            continue
        run_dir = TUNE_DIR / state["family"] / f"{state['family']}-{key}"  # dir name = wandb run name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "overrides.txt").write_text("\n".join(run_overrides(state, spec, run_dir)) + "\n")
        job = sbatch(state, [
            f"--nodes={state['nodes']}", f"--job-name=tune-{state['family']}",
            f"--output={run_dir}/slurm-%j.out", f"--error={run_dir}/slurm-%j.err",
            *state["sbatch_args"], state["job_script"], str(run_dir / "overrides.txt"),
        ])
        state["runs"][key] = {"spec": spec, "dir": str(run_dir), "job": job}
    return keys


def submit_advance(state: dict, keys: list[str]) -> None:
    jobs = [state["runs"][k]["job"] for k in keys if state["runs"][k]["job"] != "0"]
    dependency = [f"--dependency=afterany:{':'.join(jobs)}"] if jobs else []
    log_dir = TUNE_DIR / state["family"]
    sbatch(state, [
        *dependency, f"--job-name=tune-{state['family']}-advance",
        f"--output={log_dir}/advance-%j.out", *state["advance_sbatch_args"],
        state["advance_script"], state["family"],
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
    stage["keys"] = submit_runs(state, stage_specs(state, stage))
    submit_advance(state, stage["keys"])


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


def advance(family: str) -> None:
    state = load_state(family)
    if state.get("done"):
        return
    stage = state["stages"][state["stage"]]
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
            stage["keys"] = submit_runs(state, stage_specs(state, stage))
            submit_advance(state, stage["keys"])
            save_state(state)
            return
        state["params"][param] = chosen
        print(f"{param}: {chosen:.3g}")

    state["stage"] += 1
    if state["stage"] < len(state["stages"]):
        start_stage(state)
    else:
        finish(state)
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
             + (" done" if state.get("done") else "") + f", params {state['params']}"]
    for i, stage in enumerate(state["stages"]):
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
    for name in ("advance", "status"):
        sub.add_parser(name).add_argument("family")
    args = parser.parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "advance":
        advance(args.family)
    else:
        status(args.family)


if __name__ == "__main__":
    main()
