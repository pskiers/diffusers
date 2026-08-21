#!/usr/bin/env python3
"""Generate K solution images per test puzzle with a fine-tuned model, in the
layout score_amaze_images.py consumes:

    <gen_dir>/<combo>/<puzzle_index>_<attempt>.png

``combo`` = ``{geometry}_n{scale}`` (maze) or ``n{scale}`` (queens); ``puzzle_index``
is the 0-based row index in that test parquet; ``attempt`` is 0..K-1.

Backends:
  dummy  - copies the input puzzle image as each attempt (pipeline smoke-test,
           no model needed — verifies generate -> score -> wandb end to end).
  bagel  - fine-tuned BAGEL via the vendored InterleaveInferencer   [wire on cluster]
  janus  - fine-tuned Janus-Pro two-stage generation                [wire on cluster]

The input image is the native-resolution ``m_original_img`` (the model's own
transforms resize it); the scorer downsizes generated images to 144 for metrics.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import pandas as pd
from PIL import Image

TRM_ROOT = Path(__file__).resolve().parent.parent

MAZE_SCALES = [5, 7, 8, 9, 11, 13, 16]
MAZE_OOD_SCALES = [3]
MAZE_GEOMETRIES = ["square", "hexagon", "triangle", "circle"]
QUEEN_SCALES = [4, 5, 6, 7, 8, 9, 10]
QUEEN_OOD_SCALES = [12]

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


def _decode(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    s = cell.split(",", 1)[1] if isinstance(cell, str) and cell.startswith("data:") else cell
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def _iter_combos(task: str, data_root: Path):
    if task == "maze":
        for g in MAZE_GEOMETRIES:
            for s in MAZE_SCALES + MAZE_OOD_SCALES:
                yield f"{g}_n{s}", data_root / "test_maze" / g / f"n{s}_{g}_test.parquet"
    else:
        for s in QUEEN_SCALES + QUEEN_OOD_SCALES:
            yield f"n{s}", data_root / "test_queens" / f"n{s}_test.parquet"


class DummyBackend:
    def generate(self, image: Image.Image, prompt: str, k: int):
        return [image.convert("RGB") for _ in range(k)]


def build_backend(name: str, checkpoint: str | None):
    if name == "dummy":
        return DummyBackend()
    if name == "bagel":
        raise NotImplementedError(
            "bagel backend: load the FT checkpoint and build the vendored InterleaveInferencer "
            "(see third_party/amaze/infer/infer_bagel.py main() for the exact model+VAE+ViT+tokenizer "
            "loading), then per puzzle call:\n"
            "    out = inferencer(image=[image], text=[prompt], num_timesteps=16, cfg_text_scale=1.0,\n"
            "                     cfg_img_scale=1.0, cfg_interval=[0.0,1.0], cfg_renorm_min=0.0)\n"
            "    return [out['images'][0]] * k   # or K independent seeds\n"
            "Wire this during the cluster smoke-test (needs the base Bagel repo + weights)."
        )
    if name == "janus":
        raise NotImplementedError(
            "janus backend: load Janus-Pro FT weights and reproduce the two-stage image generation "
            "from third_party/amaze/infer/infer_janus.py. Wire during the cluster smoke-test."
        )
    raise ValueError(f"unknown backend '{name}'")


def main():
    ap = argparse.ArgumentParser(description="Generate FT solution images in the scorer's layout.")
    ap.add_argument("task", choices=["maze", "queens"])
    ap.add_argument("--backend", choices=["dummy", "bagel", "janus"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data-root", type=Path, default=TRM_ROOT / "data" / "amaze")
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--samples-per-puzzle", type=int, default=5)
    args = ap.parse_args()

    backend = build_backend(args.backend, args.checkpoint)
    fallback_prompt = MAZE_PROMPT if args.task == "maze" else QUEEN_PROMPT
    k = args.samples_per_puzzle
    total = 0

    for combo, parquet in _iter_combos(args.task, args.data_root):
        if not parquet.exists():
            print(f"WARN: missing {parquet} — skipping {combo}")
            continue
        df = pd.read_parquet(parquet)
        out = args.gen_dir / combo
        out.mkdir(parents=True, exist_ok=True)
        for idx, row in df.iterrows():
            image = _decode(row["m_original_img"])
            prompt = row.get("instruction") or row.get("text") or fallback_prompt
            for a, im in enumerate(backend.generate(image, prompt, k)):
                im.convert("RGB").save(out / f"{idx}_{a}.png")
        total += len(df)
        print(f">> {combo}: {len(df)} puzzles x{k} -> {out}")

    print(f"Done ({args.backend}) -> {args.gen_dir} ({total} puzzles)")


if __name__ == "__main__":
    main()
