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
from eval.amaze_eval import (
    AmazeMetrics,
    MAZE_GEOMETRIES,
    MAZE_OOD_SCALES,
    MAZE_SCALES,
    QUEEN_OOD_SCALES,
    QUEEN_SCALES,
    build_maze_result,
    build_queens_result,
    log_tables,
)

IMAGE_SIZE = 144
_TO_TENSOR = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])


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


def _print_row(label, agg):
    print(f"\n== {label} ==")
    print(f"  viol {agg['violation']*100:.2f}%  cov {agg['coverage']*100:.2f}%  "
          f"mseIn {agg['mse_inside']:.4f}  mseOut {agg['mse_outside']:.4f}  "
          f"P@1 {agg['pass1']*100:.2f}%  P@5 {agg['pass5']*100:.2f}%")


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
    return build_maze_result(per_combo, ood_combo), samples


def _score_queens(gen_dir, data_root, device, k):
    per_scale_rows, ood_scale_rows, samples = {}, {}, {}
    for s in QUEEN_SCALES:
        rows, pair = score_combo(gen_dir, f"n{s}", _queen_parquet(data_root, s), "queens", device, k)
        per_scale_rows[str(s)] = rows
        samples[f"n{s}"] = pair
    for s in QUEEN_OOD_SCALES:
        rows, pair = score_combo(gen_dir, f"n{s}", _queen_parquet(data_root, s), "queens", device, k)
        ood_scale_rows[str(s)] = rows
        samples[f"n{s}"] = pair
    return build_queens_result(per_scale_rows, ood_scale_rows), samples


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
            log_tables(run, args.task, result, samples)
            print(f"wandb: logged {args.task} metrics into run {args.run_name} (project {args.wandb_project}).")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
