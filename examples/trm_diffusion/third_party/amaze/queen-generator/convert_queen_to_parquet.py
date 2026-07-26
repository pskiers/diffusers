#!/usr/bin/env python3
"""
Convert generated Queens Puzzle levels to maze-style parquet files.

Output schema (compatible with code/amaze/data/maze_dataset.py):
- id
- instruction
- original_img
- m_original_img
- sol_img
- cell_map
- sample_json  (full JSON from generate_queens_puzzle.py: n, width, height, cell_size, queen_radius, queens, regions, etc.)

All image columns are base64-encoded PNG bytes.

python convert_queen_to_parquet.py --queen-outdir /media/raid/workspace/zhaoyanpeng/code/queen/train_7_6400 --dataset-outdir /media/raid/workspace/zhaoyanpeng/code/queen/queen_train_7_3200 --test-ratio 0.5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Queens output to parquet dataset.")
    parser.add_argument(
        "--queen-outdir",
        type=Path,
        required=True,
        help="Directory from generate_queens_puzzle.py output (contains puzzle/ gt/ json/ cell_map/).",
    )
    parser.add_argument(
        "--dataset-outdir",
        type=Path,
        required=True,
        help="Output directory for maze_dataset_train.parquet and maze_dataset_test.parquet.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction for test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--instruction-template",
        type=str,
        default=(
            "Given the puzzle image, generate the solved board by placing one queen (represented by a solid black circle in the center of a grid cell) in each row, "
            "column, and colored region while ensuring queens do not touch in 8-neighborhood."
        ),
        help="Instruction text for each sample.",
    )
    return parser.parse_args()


def to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_pgm_raw_ids(path: Path) -> np.ndarray:
    """Load PGM (P5) without value normalization; keep original integer IDs."""
    data = path.read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"Unsupported PGM format (expect P5): {path}")

    # Parse header tokens while skipping comments.
    i = 2
    n = len(data)

    def skip_ws_and_comments(pos: int) -> int:
        while pos < n:
            c = data[pos]
            if c in b" \t\r\n":
                pos += 1
                continue
            if c == ord("#"):
                while pos < n and data[pos] != ord("\n"):
                    pos += 1
                continue
            break
        return pos

    def read_token(pos: int) -> Tuple[bytes, int]:
        pos = skip_ws_and_comments(pos)
        start = pos
        while pos < n and data[pos] not in b" \t\r\n":
            pos += 1
        if start == pos:
            raise ValueError(f"Malformed PGM header: {path}")
        return data[start:pos], pos

    w_tok, i = read_token(i)
    h_tok, i = read_token(i)
    maxv_tok, i = read_token(i)

    width = int(w_tok)
    height = int(h_tok)
    maxval = int(maxv_tok)
    if maxval <= 0 or maxval > 65535:
        raise ValueError(f"Invalid maxval in {path}: {maxval}")

    i = skip_ws_and_comments(i)
    pixel = data[i:]
    count = width * height

    if maxval < 256:
        if len(pixel) < count:
            raise ValueError(f"PGM data too short: {path}")
        arr = np.frombuffer(pixel[:count], dtype=np.uint8).astype(np.uint32)
    else:
        need = count * 2
        if len(pixel) < need:
            raise ValueError(f"PGM data too short: {path}")
        # PGM stores 16-bit samples in big-endian.
        arr = np.frombuffer(pixel[:need], dtype=">u2").astype(np.uint32)

    return arr.reshape(height, width)


def load_cell_map_as_rgb(path: Path) -> Image.Image:
    """
    Read .pgm cell map and encode cell id to RGB:
    R = id & 255, G = (id >> 8) & 255, B = (id >> 16) & 255
    """
    arr = load_pgm_raw_ids(path)
    r = (arr & 255).astype(np.uint8)
    g = ((arr >> 8) & 255).astype(np.uint8)
    b = ((arr >> 16) & 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def find_image(path_no_suffix: Path) -> Path:
    png = path_no_suffix.with_suffix(".png")
    svg = path_no_suffix.with_suffix(".svg")
    if png.exists():
        return png
    if svg.exists():
        raise ValueError(
            f"Found SVG only at {svg}. Please regenerate with --image-format png "
            "because training loader expects raster images."
        )
    raise FileNotFoundError(f"Missing image for {path_no_suffix.name} (.png/.svg)")


def build_rows(queen_outdir: Path, instruction: str) -> List[Dict]:
    puzzle_dir = queen_outdir / "puzzle"
    gt_dir = queen_outdir / "gt"
    json_dir = queen_outdir / "json"
    cell_map_dir = queen_outdir / "cell_map"

    if not puzzle_dir.exists() or not gt_dir.exists() or not json_dir.exists() or not cell_map_dir.exists():
        raise FileNotFoundError(
            "Input directory must contain puzzle/, gt/, json/, and cell_map/ subdirectories."
        )

    level_jsons = sorted(json_dir.glob("level_*.json"))
    if not level_jsons:
        raise ValueError(f"No level json found in {json_dir}")

    rows: List[Dict] = []
    for jf in level_jsons:
        stem = jf.stem
        puzzle_path = find_image(puzzle_dir / stem)
        gt_path = find_image(gt_dir / stem)
        cell_map_path = cell_map_dir / f"{stem}.pgm"
        if not cell_map_path.exists():
            raise FileNotFoundError(f"Missing cell map: {cell_map_path}")

        puzzle_img = load_image(puzzle_path)
        gt_img = load_image(gt_path)
        if puzzle_img.size != gt_img.size:
            raise ValueError(f"Size mismatch between puzzle and gt for {stem}")

        cell_map_img = load_cell_map_as_rgb(cell_map_path).resize(puzzle_img.size, Image.NEAREST)

        # 读取完整 json（n/width/height 供 inference 按尺寸过滤；完整内容写入 sample_json）
        meta: Dict = {}
        try:
            with open(jf, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        n = meta.get("n")
        w = meta.get("width")
        h = meta.get("height")

        row = {
            "id": stem,
            "instruction": instruction,
            "original_img": to_png_base64(puzzle_img),
            "m_original_img": to_png_base64(puzzle_img),
            "sol_img": to_png_base64(gt_img),
            "cell_map": to_png_base64(cell_map_img),
            "sample_json": json.dumps(meta, ensure_ascii=False),
        }
        if n is not None:
            row["n"] = int(n)
        if w is not None:
            row["width"] = int(w)
        if h is not None:
            row["height"] = int(h)
        rows.append(row)
    return rows


def split_rows(rows: List[Dict], test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test-ratio must be in [0.0, 1.0).")
    rng = random.Random(seed)
    data = list(rows)
    rng.shuffle(data)

    n = len(data)
    n_test = int(round(n * test_ratio))
    if n > 1 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n_test >= n:
        n_test = n - 1

    test_rows = data[:n_test]
    train_rows = data[n_test:]
    return train_rows, test_rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args.queen_outdir, args.instruction_template)
    train_rows, test_rows = split_rows(rows, args.test_ratio, args.seed)

    outdir: Path = args.dataset_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=train_df.columns)

    train_path = outdir / "maze_dataset_train.parquet"
    test_path = outdir / "maze_dataset_test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"train: {len(train_df)} -> {train_path}")
    print(f"test:  {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    main()
