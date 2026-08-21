#!/usr/bin/env python3
"""Score pre-generated Bagel/Janus solution images with the SAME AmazeMetrics used
for PT/DiT, and log the same general/OOD/per-shape/per-size tables to wandb.

Generation must follow this layout contract (index-aligned with the per-size test
parquet, one PNG per attempt):

    <gen_dir>/<combo>/<puzzle_index>_<attempt>.png

where ``combo`` is ``{geometry}_n{scale}`` (maze) or ``n{scale}`` (queens),
``puzzle_index`` is the 0-based row index in that test parquet, and ``attempt``
is 0..K-1. Missing files are scored as blank (black) so one failed generation
never crashes the run.

Usage:
    python experiments/score_amaze_images.py maze --gen-dir runs/ft_bagel_maze/generated \
        --run-name ft_bagel_maze --wandb-project amaze_final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

TRM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRM_ROOT))

from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import AmazeMetrics

# Must match experiments/sample_amaze_metrics.py (in-dist vs OOD split).
MAZE_SCALES = [5, 7, 8, 9, 11, 13, 16]
MAZE_OOD_SCALES = [3]
MAZE_GEOMETRIES = ["square", "hexagon", "triangle", "circle"]
QUEEN_SCALES = [4, 5, 6, 7, 8, 9, 10]
QUEEN_OOD_SCALES = [12]

IMAGE_SIZE = 144
_TO_TENSOR = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])
_METRIC_KEYS = ("violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5")


def _aggregate(rows):
    if not rows:
        return {k: 0.0 for k in _METRIC_KEYS}
    import pandas as pd
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in _METRIC_KEYS}


def _load_img(path: Path) -> torch.Tensor:
    if path.exists():
        return _TO_TENSOR(Image.open(path).convert("RGB"))
    return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


def _maze_combo(geometry: str, scale: int) -> str:
    return f"{geometry}_n{scale}"


def _maze_parquet(data_root: Path, geometry: str, scale: int) -> Path:
    return data_root / "test_maze" / geometry / f"n{scale}_{geometry}_test.parquet"


def _queen_parquet(data_root: Path, scale: int) -> Path:
    return data_root / "test_queens" / f"n{scale}_test.parquet"


def score_combo(gen_dir: Path, combo: str, parquet: Path, task: str, device, k: int):
    if not parquet.exists():
        raise FileNotFoundError(f"test parquet not found: {parquet} (run gen_amaze.py test {task})")
    ds = AmazeDataset(str(parquet), split="test", image_size=IMAGE_SIZE,
                      condition_field="m_original_img", target_field="sol_img",
                      num_channels=3, include_metadata=True)
    scorer = AmazeMetrics(device=device, task=task)
    rows, sample_pair = [], None
    chunk = 64
    for start in range(0, len(ds), chunk):
        puzzles = [ds[i] for i in range(start, min(start + chunk, len(ds)))]
        n = len(puzzles)
        inputs = torch.zeros(n, k, 3, IMAGE_SIZE, IMAGE_SIZE)
        for pi, gi in enumerate(range(start, start + n)):
            for a in range(k):
                inputs[pi, a] = _load_img(gen_dir / combo / f"{gi}_{a}.png")
        if sample_pair is None and n > 0:
            cond = puzzles[0].spatial_conditions
            sample_pair = {"generated": inputs[0, 0].clone(),
                           "condition": cond.detach().cpu() if cond is not None else None}
        metadata = [p.metadata if p.metadata is not None else {} for p in puzzles]
        for rec in scorer.compute_and_accumulate_metrics(inputs, metadata):
            pass_at_k = rec[f"pass_at_{k}"] if k > 1 else rec["pass"]
            rows.append({
                "violation": rec["background_violation"], "coverage": rec["gt_cell_coverage"],
                "mse_inside": rec["mse_inside"], "mse_outside": rec["mse_outside"],
                "pass1": rec["pass"], "pass5": pass_at_k,
            })
    return rows, sample_pair


def _make_wandb_image(tensor):
    if tensor is None:
        return None
    import numpy as np
    import wandb
    arr = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return wandb.Image((arr * 255).astype("uint8"))


def _print_row(label, agg):
    print(f"\n== {label} ==")
    print(f"  viol {agg['violation']*100:.2f}%  cov {agg['coverage']*100:.2f}%  "
          f"mseIn {agg['mse_inside']:.4f}  mseOut {agg['mse_outside']:.4f}  "
          f"P@1 {agg['pass1']*100:.2f}%  P@5 {agg['pass5']*100:.2f}%")


def _log_wandb(run, task, result, samples):
    import wandb
    prefix = f"amaze_eval/{task}"
    for key, val in result["overall"].items():
        run.summary[f"{prefix}/overall/{key}"] = val
    for key, val in result.get("overall_ood", {}).items():
        run.summary[f"{prefix}/overall_ood/{key}"] = val

    img_cols = ["group", "generated", "condition", "violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5"]
    metric_cols = ["group", "violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5"]

    def _img_table(named_pairs):
        t = wandb.Table(columns=img_cols)
        for name, agg, pair in named_pairs:
            pair = pair or {}
            t.add_data(name, _make_wandb_image(pair.get("generated")), _make_wandb_image(pair.get("condition")),
                       agg["violation"], agg["coverage"], agg["mse_inside"], agg["mse_outside"], agg["pass1"], agg["pass5"])
        return t

    def _metric_table(named_aggs):
        t = wandb.Table(columns=metric_cols)
        for name, agg in named_aggs:
            t.add_data(name, agg["violation"], agg["coverage"], agg["mse_inside"], agg["mse_outside"], agg["pass1"], agg["pass5"])
        return t

    if task == "maze":
        for g, by_scale in result["per_shape"].items():
            rows = [(f"{s}x{s}", by_scale[str(s)], samples.get(_maze_combo(g, s))) for s in MAZE_SCALES]
            run.log({f"{prefix}/{g}_table": _img_table(rows)})
        for g, by_scale in result["per_shape_ood"].items():
            rows = [(f"{s}x{s}", by_scale[str(s)], samples.get(_maze_combo(g, s))) for s in MAZE_OOD_SCALES]
            run.log({f"{prefix}/{g}_ood_table": _img_table(rows)})
        run.log({f"{prefix}/per_geometry_table": _metric_table([(g, result["per_geometry"][g]) for g in MAZE_GEOMETRIES])})
        run.log({f"{prefix}/per_geometry_ood_table": _metric_table([(g, result["per_geometry_ood"][g]) for g in MAZE_GEOMETRIES])})
    else:
        combined = {**result["per_scale"], **result["per_scale_ood"]}
        rows = [(f"{s}x{s}", combined[s], samples.get(f"n{s}")) for s in combined]
        run.log({f"{prefix}/per_scale_table": _img_table(rows)})


def _score_maze(gen_dir, data_root, device, k):
    per_combo, ood_combo, samples = {}, {}, {}
    for g in MAZE_GEOMETRIES:
        for s in MAZE_SCALES:
            combo = _maze_combo(g, s)
            rows, pair = score_combo(gen_dir, combo, _maze_parquet(data_root, g, s), "maze", device, k)
            per_combo[f"{g}_{s}"] = rows
            samples[combo] = pair
        for s in MAZE_OOD_SCALES:
            combo = _maze_combo(g, s)
            rows, pair = score_combo(gen_dir, combo, _maze_parquet(data_root, g, s), "maze", device, k)
            ood_combo[f"{g}_{s}"] = rows
            samples[combo] = pair

    all_rows = [r for rows in per_combo.values() for r in rows]
    per_geometry = {g: _aggregate([r for s in MAZE_SCALES for r in per_combo[f"{g}_{s}"]]) for g in MAZE_GEOMETRIES}
    per_geometry_ood = {g: _aggregate([r for s in MAZE_OOD_SCALES for r in ood_combo[f"{g}_{s}"]]) for g in MAZE_GEOMETRIES}
    result = {
        "task": "maze",
        "overall": _aggregate(all_rows),
        "overall_ood": _aggregate([r for rows in ood_combo.values() for r in rows]),
        "per_shape": {g: {str(s): _aggregate(per_combo[f"{g}_{s}"]) for s in MAZE_SCALES} for g in MAZE_GEOMETRIES},
        "per_shape_ood": {g: {str(s): _aggregate(ood_combo[f"{g}_{s}"]) for s in MAZE_OOD_SCALES} for g in MAZE_GEOMETRIES},
        "per_geometry": per_geometry, "per_geometry_ood": per_geometry_ood,
        "n_puzzles": len(all_rows),
    }
    return result, samples


def _score_queens(gen_dir, data_root, device, k):
    per_scale_agg, ood_scale_agg, samples, raw = {}, {}, {}, {}
    for s in QUEEN_SCALES:
        rows, pair = score_combo(gen_dir, f"n{s}", _queen_parquet(data_root, s), "queens", device, k)
        per_scale_agg[str(s)] = _aggregate(rows)
        raw[str(s)] = rows
        samples[f"n{s}"] = pair
    for s in QUEEN_OOD_SCALES:
        rows, pair = score_combo(gen_dir, f"n{s}", _queen_parquet(data_root, s), "queens", device, k)
        ood_scale_agg[str(s)] = _aggregate(rows)
        samples[f"n{s}"] = pair
    all_rows = [r for s in QUEEN_SCALES for r in raw[str(s)]]
    result = {
        "task": "queens",
        "overall": _aggregate(all_rows),
        "overall_ood": ood_scale_agg[str(QUEEN_OOD_SCALES[0])] if ood_scale_agg else _aggregate([]),
        "per_scale": per_scale_agg, "per_scale_ood": ood_scale_agg,
        "n_puzzles": len(all_rows),
    }
    return result, samples


def main():
    ap = argparse.ArgumentParser(description="Score pre-generated FT solution images with AmazeMetrics -> amaze_final.")
    ap.add_argument("task", choices=["maze", "queens"])
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=TRM_ROOT / "data" / "amaze")
    ap.add_argument("--samples-per-puzzle", type=int, default=5)
    ap.add_argument("--wandb-project", default="amaze_final")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = args.samples_per_puzzle

    if args.task == "maze":
        result, samples = _score_maze(args.gen_dir, args.data_root, device, k)
    else:
        result, samples = _score_queens(args.gen_dir, args.data_root, device, k)

    _print_row(f"{args.task} overall (general)", result["overall"])
    _print_row(f"{args.task} overall (OOD)", result["overall_ood"])

    out_json = args.out_json or (args.gen_dir / f"amaze_metrics_{args.task}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved -> {out_json}")

    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, resume="allow")
        try:
            _log_wandb(run, args.task, result, samples)
            print(f"wandb: logged {args.task} metrics into run {args.run_name} (project {args.wandb_project}).")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
