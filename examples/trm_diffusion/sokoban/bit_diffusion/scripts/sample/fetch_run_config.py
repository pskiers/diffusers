"""
Fetch a finished training run's config from W&B and write it back as a Hydra
config that sample.py can load directly.

Why: sampling must rebuild the exact model architecture (num_layers, self_cond,
conditioning, trm.*, ...) and point at the right checkpoint (output_dir + run_name).
All of that was logged to W&B as the run config, so we pull it verbatim instead of
re-specifying it by hand.

It also injects wandb_run_id / wandb_entity so sample.py resumes the SAME W&B
experiment and appends the test metrics onto the existing training charts -- this
works even for runs trained before wandb_run_id.txt was saved next to checkpoints.

Writes:  <bit_diffusion>/config/_resumed/<run_name>.yaml
Prints:  the Hydra config-name to load it (e.g. "_resumed/std_6L_baseline") on
         stdout. All diagnostics go to stderr so the driver can capture the name.

Usage:
    python fetch_run_config.py --run-name std_6L_baseline \
        --project Sokoban-Ablation-Uncond [--entity my-entity]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb
import yaml


def eprint(*args):
    print(*args, file=sys.stderr)


def find_run(api: "wandb.Api", entity: str | None, project: str, run_name: str):
    """Locate a run by its display name. Falls back to a client-side scan if the
    server-side display_name filter is unsupported, and picks the newest on ties."""
    path = f"{entity}/{project}" if entity else project

    runs = list(api.runs(path, filters={"display_name": run_name}))
    if not runs:
        runs = [r for r in api.runs(path) if r.name == run_name]
    if not runs:
        raise SystemExit(f"No W&B run named '{run_name}' found in '{path}'.")
    if len(runs) > 1:
        runs.sort(key=lambda r: r.created_at, reverse=True)
        eprint(f"WARNING: {len(runs)} runs named '{run_name}'; using newest (id={runs[0].id}).")
    return runs[0]


def detect_type(cfg: dict) -> str:
    trm = cfg.get("trm") or {}
    if trm.get("n_inner") is not None:
        return "embedded"
    if cfg.get("trm"):
        return "trm"
    return "std"


def normalize_wandb_config(raw: dict) -> dict:
    """Return a plain {key: value} config from a W&B run config.

    W&B stores config in a wrapped form ({"key": {"value": ..., "desc": ...}})
    and injects an internal "_wandb" key. wandb.Api().run.config normally unwraps
    this, but the behaviour is version-dependent, so we unwrap defensively here so
    downstream (detect_type, sample.py) always sees real values -- not {"value": ...}
    nodes, which is what silently misclassified embedded runs and hid output_dir.
    """
    cfg = {}
    for key, val in raw.items():
        if key.startswith("_"):  # drop _wandb and other internal bookkeeping keys
            continue
        if isinstance(val, dict) and set(val.keys()) <= {"value", "desc"} and "value" in val:
            cfg[key] = val["value"]
        else:
            cfg[key] = val
    return cfg


BASE_CONFIG_BY_TYPE = {
    "std": "standard_diffusion.yaml",
    "trm": "trm_diffusion.yaml",
    "embedded": "embedded_trm_diffusion.yaml",
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base; override wins on leaf conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def backfill_from_base(cfg: dict, config_dir: Path) -> dict:
    """Fill keys missing from an old run's W&B config (e.g. output_dir) from the
    matching base config, so sample.py can rebuild the run. W&B values always win."""
    base_name = BASE_CONFIG_BY_TYPE[detect_type(cfg)]
    base_path = config_dir / base_name
    if not base_path.exists():
        eprint(f"WARNING: base config {base_path} not found; not backfilling defaults.")
        return cfg
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    return deep_merge(base, cfg)


def main():
    ap = argparse.ArgumentParser(description="Materialize a W&B run config for sample.py.")
    ap.add_argument("--run-name", required=True, help="W&B run display name (also the checkpoint dir name).")
    ap.add_argument("--project", required=True, help="W&B project the run lives in.")
    ap.add_argument("--entity", default=None, help="W&B entity/team (defaults to your default entity).")
    ap.add_argument("--out-dir", default=None, help="Where to write the YAML (default: <bit_diffusion>/config/_resumed).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "config" / "_resumed"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = find_run(api, args.entity, args.project, args.run_name)

    cfg = dict(run.config)
    if not cfg:
        raise SystemExit(f"Run '{run.name}' (id={run.id}) has an empty config; cannot reconstruct it.")

    # W&B may hand back the wrapped {"key": {"value": ...}} form (version-dependent);
    # unwrap to plain values and strip the internal "_wandb" key before anything reads it.
    cfg = normalize_wandb_config(cfg)

    # Runs trained before some keys (e.g. output_dir) were logged to W&B would be
    # missing them and crash sample.py's struct access. Backfill from the base config.
    had_output_dir = "output_dir" in cfg
    config_dir = Path(__file__).resolve().parents[2] / "config"
    cfg = backfill_from_base(cfg, config_dir)
    if not had_output_dir:
        eprint(
            f"WARNING: run '{run.name}' had no 'output_dir' in its W&B config; "
            f"backfilled default '{cfg.get('output_dir')}'. If checkpoints live "
            "elsewhere, pass an explicit checkpoint_path to sample.py."
        )

    # Make checkpoint lookup and W&B resume-logging deterministic for sample.py.
    cfg["run_name"] = run.name
    cfg["wandb_project"] = run.project
    cfg["wandb_entity"] = run.entity
    cfg["wandb_run_id"] = run.id
    cfg["resume_from_checkpoint"] = None
    # Lightning logged these into the run config; they are not real Hydra knobs.
    for key in ("num_parameters", "num_trainable_parameters"):
        cfg.pop(key, None)

    out_path = out_dir / f"{run.name}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    eprint(f"Wrote {out_path}")
    eprint(f"  run id   : {run.id}")
    eprint(f"  entity   : {run.entity}")
    eprint(f"  project  : {run.project}")
    eprint(f"  type     : {detect_type(cfg)}")
    eprint(f"  output_dir: {cfg.get('output_dir')}")

    # Only the Hydra config-name on stdout, for the bash driver to capture.
    print(f"_resumed/{run.name}")


if __name__ == "__main__":
    main()
