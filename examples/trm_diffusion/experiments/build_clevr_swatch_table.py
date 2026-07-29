"""
experiments/build_clevr_swatch_table.py — Precompute the real-image swatch
lookup table for ObjectFeatureEncoderV1Swatch (models/condition_encoders.py).

Scans the CLEVR training scenes once and, for each unique
(color, shape, material, size) combination, crops a tight, unoccluded patch
around one real object instance's true position from its actual rendered
image — see datasets.clevr_dataset.extract_clevr_swatch_table for the full
rationale (a hand-drawn icon has no real visual correspondence to a Blender
render, so the anchor has to be real pixels from the same renderer/lighting
engine). Saves the result to disk once; ObjectFeatureEncoderV1Swatch just
loads it.

Usage:
    python experiments/build_clevr_swatch_table.py \\
      --clevr_root data/clevr --split train --swatch_size 32 \\
      --out runs/clevr_swatch_table.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets.clevr_dataset import calibrate_mask_projection, extract_clevr_swatch_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevr_root", default="data/clevr")
    parser.add_argument("--split", default="train", choices=["train", "validation", "val"])
    parser.add_argument("--swatch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=1.6)
    parser.add_argument("--num_calibration_scenes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/clevr_swatch_table.pt")
    args = parser.parse_args()

    dataset_path = os.path.join(args.clevr_root, "CLEVR_v1.0")
    filename_split = "val" if args.split == "validation" else args.split
    scene_path = os.path.join(dataset_path, "scenes", f"CLEVR_{filename_split}_scenes.json")
    image_dir = os.path.join(dataset_path, "images", filename_split)

    print(f"Loading scenes from {scene_path}...")
    with open(scene_path, "r") as f:
        scenes = json.load(f)["scenes"]
    print(f"{len(scenes)} scenes loaded.")

    print("Calibrating perspective projection...")
    H_inv = calibrate_mask_projection(scenes, num_scenes=args.num_calibration_scenes)

    print("Extracting swatch table...")
    table = extract_clevr_swatch_table(
        scenes, image_dir, H_inv, swatch_size=args.swatch_size, margin=args.margin, seed=args.seed
    )
    print(f"Table shape: {tuple(table.shape)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(table, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
