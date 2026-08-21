#!/usr/bin/env python3
"""Export the amaze parquet into the directory layout the vendored AMAZE
fine-tuning loaders expect (Bagel / Janus):

    <out>/{maze,queens}/maze_dataset_train.parquet
    <out>/{maze,queens}/maze_dataset_test.parquet

Train comes from ``train.orig.parquet`` (native resolution, full columns, base64
images) — the same 30k puzzles as PT/DDPM, but NOT the 144px ``train.parquet``.
Images stay base64 PNG and the ``instruction`` column is ensured (paper prompt).
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import pandas as pd
from PIL import Image

TRM_ROOT = Path(__file__).resolve().parent.parent
AMAZE_ROOT = Path(os.environ.get("AMAZE_OUT_ROOT", str(TRM_ROOT / "data" / "amaze")))

IMAGE_COLS = ("original_img", "m_original_img", "sol_img", "mask_img", "cell_map")

MAZE_PROMPT = (
    "Add the blue solution path for the maze, connect start point (solid red circle) "
    "to end point (red 'X' mark). Ensure all original maze elements (walls, points, "
    "etc.) remain unchanged\u2014only add the path."
)
QUEEN_PROMPT = (
    "Given the puzzle image, generate the solved board by placing one queen "
    "(represented by a solid black circle in the center of a grid cell) in each row, "
    "column, and colored region while ensuring queens do not touch in 8-neighborhood."
)


def _train_src(task: str) -> Path:
    leaf = AMAZE_ROOT / f"train_{task}" / "all_train_size144"
    orig = leaf / "train.orig.parquet"
    return orig if orig.exists() else leaf / "train.parquet"


def _test_src(task: str) -> Path:
    return AMAZE_ROOT / f"test_{task}" / "all_test.parquet"


def _to_b64(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return cell
    if isinstance(cell, str):
        return cell
    if isinstance(cell, (bytes, bytearray)):
        return base64.b64encode(bytes(cell)).decode("utf-8")
    if isinstance(cell, Image.Image):
        buf = io.BytesIO()
        cell.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return cell


def _decode_ok(cell) -> bool:
    try:
        s = cell.split(",", 1)[1] if isinstance(cell, str) and cell.startswith("data:") else cell
        Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        return True
    except Exception:
        return False


def _prepare(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    df = df.copy()
    if "instruction" not in df.columns or df["instruction"].isna().all():
        df["instruction"] = prompt
    else:
        df["instruction"] = df["instruction"].fillna(prompt)
    for col in IMAGE_COLS:
        if col in df.columns:
            df[col] = df[col].map(_to_b64)
    return df


def export_task(task: str, out_root: Path) -> None:
    prompt = MAZE_PROMPT if task == "maze" else QUEEN_PROMPT
    train_src, test_src = _train_src(task), _test_src(task)
    if not train_src.exists():
        raise FileNotFoundError(f"train parquet not found: {train_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"test parquet not found: {test_src} (run gen_amaze.py test {task})")

    if train_src.name == "train.parquet":
        print(f"WARN [{task}]: train.orig.parquet missing \u2014 falling back to 144px train.parquet. "
              "Fine-tuning would run at low resolution; regenerate to get the native-res backup.")

    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _prepare(pd.read_parquet(train_src), prompt)
    test = _prepare(pd.read_parquet(test_src), prompt)
    train.to_parquet(out_dir / "maze_dataset_train.parquet", index=False)
    test.to_parquet(out_dir / "maze_dataset_test.parquet", index=False)

    ok = _decode_ok(train.iloc[0].get("m_original_img")) if len(train) else False
    print(f">> {task}: train {len(train)} rows, test {len(test)} rows \u2192 {out_dir} "
          f"(sample decode: {'ok' if ok else 'FAILED'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export amaze parquet for Bagel/Janus fine-tuning.")
    ap.add_argument("task", choices=["maze", "queens", "both"])
    ap.add_argument("--out-root", type=Path, default=AMAZE_ROOT / "ft")
    args = ap.parse_args()
    tasks = ["maze", "queens"] if args.task == "both" else [args.task]
    for t in tasks:
        export_task(t, args.out_root)
    print(f"Done \u2192 {args.out_root}")


if __name__ == "__main__":
    main()
