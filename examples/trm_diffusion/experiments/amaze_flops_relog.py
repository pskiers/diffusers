"""Re-log the AMAZE FLOPs sweep JSONs to wandb as readable runs (no re-sampling).

amaze_flops_sweep.py logs every setting as a consecutive wandb "step", so the default
panels plot accuracy against the setting's INDEX, and its table has no usable labels.
This script reads the JSONs it wrote and logs, per mode, one run with:
  * table  "results"      one row per setting: labelled columns (setting, series, the
                           knobs of that mode), split, sampling, TFLOPs per maze, exact,
                           pass, violation, coverage, exact per board size and per shape
  * table  "by_size"      long form (setting, series, size, TFLOPs, exact)
  * chart  exact / pass vs TFLOPs per maze (one line per series)
  * chart  exact vs TFLOPs, one line per board size (dm, trm_flat, cut_at_k)
  * profile mode: the per-(denoising step, reasoning iteration) z_H change table
  * --report_dir: the report PNGs as images in a separate run
Run summaries hold only the run's description, never "last point" metrics.

python experiments/amaze_flops_relog.py --task maze --root runs/flops_analysis/maze \
    --project PaTh-rebuttal --group flops [--suffix relog] [--report_dir runs/flops_analysis/maze/report]
"""
import argparse
import json
from pathlib import Path

SPLITS = {"tune_split": "tuning (boards 0-15 per cell)", "report_split": "report (boards 16-47 per cell)"}


def split_of(d) -> str:
    """tune_split / report_split from a sweep result's board offset."""
    return "tune_split" if int(d.get("offset", 0)) == 0 else "report_split"

DESCRIPTIONS = {
    "dm": "DiT baseline: accuracy vs number of DDPM sampling steps (FLOPs = steps x passes x F_model).",
    "trm_flat": "PaTh default inference: the same n_sup at every denoising step, fresh thinker state each step; "
                "first points also lower the H/L reasoning cycles at n_sup=1.",
    "profile": "Trained PaTh policy, instrumented: how much the thinker state z_H changes per reasoning iteration, "
               "and how similar z_H is between denoising steps (regime boundary).",
    "cut_at_k": "Run the thinker only on the first k of the denoising steps, then reuse its last steering.",
    "converge": "Stop reasoning per maze once the relative z_H change < eps (fresh state, or carried within a regime).",
    "carry": "n_sup x thinker state policy: fresh each step / carried across all steps / carried within a regime.",
    "trm_wise": "PaTh 'wise' inference: a budget-matched schedule of n_sup over the denoising steps, carried state, "
                "steering frozen after freeze_frac of the trajectory.",
    "alloc": "Where reasoning is needed: the same total reasoning budget moved over the denoising trajectory "
             "(flat vs bump/dip at several peaks/widths, front, back).",
}


def _series_and_setting(mode, p):
    """(series, setting, knob columns) for one point of a mode."""
    if mode == "dm":
        return "DiT", f"{p['num_steps']} step{'s' if p['num_steps'] != 1 else ''}", {"num_steps": p["num_steps"]}
    if mode == "trm_flat":
        hl = f" H={p['H_cycles']} L={p['L_cycles']}"
        return "PaTh flat", f"n_sup={p['n_sup']}{hl}", {"n_sup": p["n_sup"], "H_cycles": p["H_cycles"],
                                                         "L_cycles": p["L_cycles"]}
    if mode == "profile":
        return "PaTh trained policy", "n_sup=16 (trained)", {}
    if mode == "cut_at_k":
        k = p["freeze_step"]
        return "cut_at_k", f"reason on first {k} of the steps ({p['reason_frac']:.0%})", {
            "reason_frac": p["reason_frac"], "reason_steps": k}
    if mode == "converge":
        it = p.get("iters_per_step") or []
        return p["variant"], f"eps={p['eps']} {p['variant']}", {
            "eps": p["eps"], "variant": p["variant"], "mean_iters_per_step": sum(it) / len(it) if it else None}
    if mode == "carry":
        return p["variant"], f"n_sup={p['n_sup']} {p['variant']}", {"n_sup": p["n_sup"], "variant": p["variant"]}
    if mode == "trm_wise":
        hl = "" if "H_cycles" not in p else f" H={p['H_cycles']} L={p['L_cycles']}"
        return "PaTh wise", f"budget = flat n_sup {p['flat_n']}{hl}", {
            "flat_n": p["flat_n"], "H_cycles": p.get("H_cycles"), "L_cycles": p.get("L_cycles"),
            "freeze_step": p.get("freeze_step")}
    if mode == "alloc":
        shape = p["direction"] + ("" if p.get("peak") is None else f" peak={p['peak']} width={p.get('width')}")
        return f"budget n_sup {p['flat_n']}", f"n_sup {p['flat_n']}: {shape}", {
            "flat_n": p["flat_n"], "direction": p["direction"], "peak": p.get("peak"), "width": p.get("width")}
    # Unknown mode: still log it, labelled by its scalar knobs.
    knobs = {k: v for k, v in p.items() if isinstance(v, (int, float, str)) and k not in (
        "flops", "exact", "pass", "violation", "coverage", "n", "wall_time_sec")}
    return mode, " ".join(f"{k}={v}" for k, v in knobs.items()) or mode, knobs


def log_mode(wandb, path, split, args):
    """Re-log one sweep JSON (run name flops_<task>_<mode>[_<suffix>])."""
    d = json.loads(path.read_text())
    name = f"flops_{args.task}_{d['mode']}" + (f"_{args.suffix}" if args.suffix else "")
    log_points(wandb, d, split, project=args.project, group=args.group, name=name,
               tags=[args.task, d["mode"], "flops", "relog"], extra_config={"source_json": str(path)})


def log_points(wandb, d, split, project, group, name, tags, job_type=None, extra_config=None):
    """One wandb run for a sweep result dict ``d`` (the JSON amaze_flops_sweep.py writes)."""
    mode, pts = d["mode"], d["points"]
    cfg_scale = float(d.get("cfg_scale", 1.0))
    sampling = f"CFG {cfg_scale:g} (2 passes/step)" if cfg_scale != 1.0 else "no CFG (1 pass/step)"
    sizes = sorted({int(s) for p in pts for s in p["per_scale"]})
    shapes = sorted({g for p in pts for g in p.get("per_shape", {})})
    rows, knob_cols = [], []
    for p in pts:
        series, setting, knobs = _series_and_setting(mode, p)
        knob_cols += [k for k in knobs if k not in knob_cols]
        rows.append((series, setting, knobs, p))
    cols = (["setting", "series", *knob_cols, "split", "n_mazes", "sampling", "tflops_per_maze",
             "exact", "pass", "violation", "coverage"]
            + [f"exact_size_{s}" for s in sizes] + [f"exact_{g}" for g in shapes])
    table = wandb.Table(columns=cols)
    by_size = wandb.Table(columns=["setting", "series", "size", "tflops_per_maze", "exact", "n"])
    for series, setting, knobs, p in rows:
        tf = p["flops"] / 1e12
        table.add_data(setting, series, *[knobs.get(k) for k in knob_cols], SPLITS[split], p["n"], sampling, tf,
                       p["exact"], p["pass"], p["violation"], p["coverage"],
                       *[p["per_scale"].get(str(s), {}).get("exact") for s in sizes],
                       *[p.get("per_shape", {}).get(g, {}).get("exact") for g in shapes])
        for s in sizes:
            v = p["per_scale"].get(str(s))
            if v is not None:
                by_size.add_data(setting, series, f"size {s}", tf, v["exact"], v.get("n"))

    job_type = job_type or ("sweep" if mode in ("dm", "trm_flat", "trm_wise", "alloc") else "analysis")
    run = wandb.init(project=project, group=group, name=name, job_type=job_type, tags=list(tags), reinit=True,
                     config={k: v for k, v in d.items() if k not in ("points", "profile", "timesteps")} | (extra_config or {}))
    run.summary.update({"description": DESCRIPTIONS.get(mode, mode), "split": SPLITS[split], "sampling": sampling,
                        "n_mazes": d.get("n_puzzles"), "n_settings": len(pts)})
    log = {"results": table, "by_size": by_size,
           "exact_vs_tflops": wandb.plot.line(table, "tflops_per_maze", "exact", stroke="series",
                                              title=f"{mode}: exact vs TFLOPs per maze ({SPLITS[split]}, {sampling})"),
           "pass_vs_tflops": wandb.plot.line(table, "tflops_per_maze", "pass", stroke="series",
                                             title=f"{mode}: pass vs TFLOPs per maze ({SPLITS[split]}, {sampling})")}
    if mode in ("dm", "trm_flat", "cut_at_k"):
        log["exact_by_size_vs_tflops"] = wandb.plot.line(by_size, "tflops_per_maze", "exact", stroke="size",
                                                         title=f"{mode}: exact per board size vs TFLOPs per maze")
    if mode == "profile" and d.get("profile"):
        prof = d["profile"]
        ts = d.get("timesteps") or list(range(len(prof["rel_change_mean"])))
        zt = wandb.Table(columns=["denoise_step", "timestep", "iteration", "rel_change_mean", "rel_change_median"])
        for s, (mean_row, med_row) in enumerate(zip(prof["rel_change_mean"], prof["rel_change_median"])):
            for i, (a, b) in enumerate(zip(mean_row, med_row)):
                zt.add_data(s, ts[s] if s < len(ts) else None, i + 1, a, b)
        log["z_H_change_per_iteration"] = zt
    run.log(log)
    run.finish()
    print(f"{name}: {len(pts)} settings -> {run.url}")
    return run.url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="maze")
    ap.add_argument("--root", required=True, help="runs/flops_analysis/<task> (contains tune_split/, report_split/)")
    ap.add_argument("--project", default="PaTh-rebuttal")
    ap.add_argument("--group", default="flops")
    ap.add_argument("--suffix", default="relog")
    ap.add_argument("--report_dir", default=None)
    args = ap.parse_args()
    import wandb
    root = Path(args.root)
    for split in ("report_split", "tune_split"):
        for path in sorted((root / split).glob(f"{args.task}_*.json")):
            log_mode(wandb, path, split, args)
    if args.report_dir:
        pngs = sorted(Path(args.report_dir).glob("*.png"))
        if pngs:
            run = wandb.init(project=args.project, group=args.group, job_type="analysis", reinit=True,
                             name=f"flops_report_{args.task}" + (f"_{args.suffix}" if args.suffix else ""),
                             tags=[args.task, "flops", "report", "relog"])
            run.log({p.stem: wandb.Image(str(p), caption=p.stem) for p in pngs})
            summ = Path(args.report_dir) / f"{args.task}_p1_profile_summary.json"
            if summ.exists():
                run.summary.update(json.loads(summ.read_text()))
            run.finish()
            print(f"report images: {len(pngs)} -> {run.url}")


if __name__ == "__main__":
    main()
