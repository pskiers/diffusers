"""
ABOUTME: Report for rebuttal exp 02 (predictor-corrector DM sampling, slurm_scripts/wcss_pc_sweep.sh): main metric vs
ABOUTME: K per dataset with inference FLOPs per sample, the best Langevin r from the tune runs, and phase-2 manifest rows.

FLOPs per sample = predictor steps x (1 + K) x (2 under CFG) x F_dm, F_dm = one denoiser forward (flops_utils.
measure_dm_flops, B=1, measured on the architecture built from the same config as the eval; no checkpoint needed).
Both correctors cost one (guided) model evaluation per corrector step.

Usage (from examples/trm_diffusion):
    python experiments/pc_sweep_report.py                      # reads results/pc_sweep/<dataset>/*.json
    python experiments/pc_sweep_report.py --root <dir> --flops-cache results/pc_sweep/flops.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

# eval.py / amaze script overrides of each paper-table DM (wcss_pc_sweep.sh), minus checkpoint / output paths
DATASETS = {
    "extreme": dict(overrides=["experiment=mnist_unet_concat_painter_srm_1000t", "data=mnist_sudoku",
                               "diffusion.num_train_timesteps=4800"],
                    metric=lambda d: d["metrics"]["puzzle_acc"], metric_name="puzzle_acc", steps=20, cfg=True),
    "hard": dict(overrides=["experiment=mnist_unet_concat_painter_srm_1000t", "diffusion.num_train_timesteps=4800"],
                 metric=lambda d: d["metrics"]["given_consistent_puzzle_acc"],
                 metric_name="given_consistent_puzzle_acc", steps=20, cfg=True),
    "maze": dict(overrides=["experiment=amaze_dit_maze"], metric=lambda d: d["overall"]["exact1"],
                 metric_name="exact1", steps=99, cfg=False,
                 extra=lambda d: {k: d["overall"][k] for k in ("violation", "coverage", "pass1", "exact5")}),
    "queens": dict(overrides=["experiment=amaze_dit_queens"], metric=lambda d: d["overall"]["exact1"],
                   metric_name="exact1", steps=99, cfg=False,
                   extra=lambda d: {k: d["overall"][k] for k in ("violation", "coverage", "pass1", "exact5")}),
    "clevr": dict(overrides=["experiment=clevr_unet_concat_painter", "backbone.block_out_channels=[256,512,512]",
                             "backbone.layers_per_block=3", "backbone.cross_attention_dim=1024",
                             "data.clevr_root=cache_dir"],
                  metric=lambda d: d["metrics"]["clevr_spatial_acc"], metric_name="clevr_spatial_acc", steps=20,
                  cfg=True, extra=lambda d: {k: d["metrics"][k] for k in ("clevr_precision", "clevr_recall",
                                                                          "clevr_perfect_gen")}),
}
NAME_RE = re.compile(r"(?P<variant>none|langevin|restart)_K(?P<K>\d+)(?:_r(?P<snr>[\d.]+))?_(?P<subset>full|tune)$")


def measure_f_dm(dataset: str) -> int:
    """FLOPs of one denoiser forward of the dataset's DM, on one real eval sample (B=1)."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from datasets.data_sample import collate_data_samples
    from factory import build_datasets, build_model
    from flops_utils import measure_dm_flops

    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=DATASETS[dataset]["overrides"])
    _, eval_ds = build_datasets(cfg)
    device = torch.device("cuda")  # on CPU the FLOP counter misses the attention kernel (torch 2.4)
    model = build_model(cfg, instantiate(cfg.diffusion)).eval().to(device)
    item = eval_ds[0]
    batch = collate_data_samples([item]) if not isinstance(item, dict) else item
    keep = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)
            if f.name not in ("images", "x_noisy", "timesteps", "target")}
    sample = dataclasses.replace(type(batch)(**keep), x_noisy=torch.zeros(1, *model.noise_shape),
                                 timesteps=torch.zeros(1, dtype=torch.long)).to(device)
    with torch.no_grad():
        return int(measure_dm_flops(model, sample))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(HERE / "results" / "pc_sweep"))
    p.add_argument("--flops-cache", default=None, help="JSON {dataset: F_dm}; measured and written if missing")
    args = p.parse_args()
    root = Path(args.root)
    cache_path = Path(args.flops_cache) if args.flops_cache else root / "flops.json"
    f_dm = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    lines = ["# Exp 02: predictor-corrector sampling of the DM baselines", "",
             "FLOPs/sample = steps x (1+K) x (2 if CFG) x F_dm. `tune` rows: Langevin r selection slice (seed 1).", ""]
    phase2 = []
    for ds, spec in DATASETS.items():
        files = sorted((root / ds).glob("*.json")) if (root / ds).exists() else []
        if not files:
            continue
        if ds not in f_dm:
            f_dm[ds] = measure_f_dm(ds)
            cache_path.write_text(json.dumps(f_dm, indent=2))
        rows = []
        for f in files:
            m = NAME_RE.match(f.stem)
            if m is None:
                continue
            d = json.loads(f.read_text())
            K = int(m["K"])
            flops = spec["steps"] * (1 + K) * (2 if spec["cfg"] else 1) * f_dm[ds]
            rows.append(dict(variant=m["variant"], K=K, snr=m["snr"] or "-", subset=m["subset"],
                             metric=spec["metric"](d), flops=flops, **(spec.get("extra", lambda _: {})(d))))
        rows.sort(key=lambda r: (r["subset"] != "full", r["variant"], r["K"], r["snr"]))
        extra_keys = [k for k in rows[0] if k not in ("variant", "K", "snr", "subset", "metric", "flops")]
        lines += [f"## {ds} ({spec['metric_name']}; {spec['steps']} predictor steps, CFG {'2' if spec['cfg'] else 'no'},"
                  f" F_dm {f_dm[ds] / 1e9:.2f} GFLOPs)", "",
                  "| " + " | ".join(["subset", "variant", "K", "r", spec["metric_name"], "GFLOPs/sample", *extra_keys])
                  + " |", "|---" * (6 + len(extra_keys)) + "|"]
        lines += ["| " + " | ".join([r["subset"], r["variant"], str(r["K"]), r["snr"], f"{r['metric']:.4f}",
                                     f"{r['flops'] / 1e9:.1f}", *(f"{r[k]:.4f}" for k in extra_keys)]) + " |"
                  for r in rows]
        tune = [r for r in rows if r["subset"] == "tune" and r["variant"] == "langevin"]
        if tune:
            best = max(tune, key=lambda r: r["metric"])
            lines += ["", f"Best Langevin r (tune slice, K=1): **{best['snr']}** ({best['metric']:.4f})"]
            phase2 += [f"{ds} langevin {k} {best['snr']} full" for k in (1, 2, 4, 8)]
        lines.append("")
    if phase2:
        lines += ["## Phase-2 manifest rows (Langevin K sweep at the best r)", "", "```", *phase2, "```"]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
