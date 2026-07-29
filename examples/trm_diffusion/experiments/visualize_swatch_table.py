"""
experiments/visualize_swatch_table.py — Plot every crop in a swatch table
(built by experiments/build_clevr_swatch_table.py) in a labeled grid, so you
can eyeball whether the crops are tight/clean or clipping the object.

The table is ordered color -> shape -> material -> size (96 entries total),
matching datasets.clevr_dataset.extract_clevr_swatch_table and
models.condition_encoders._clevr_swatch_indices. An all-black tile means
that combination was missing from the scanned scenes (see the "filled with
zeros" warning printed at build time).

Usage:
    python experiments/visualize_swatch_table.py \\
      --table runs/clevr_swatch_table.pt --out runs/clevr_swatch_table.png

    # only look at one shape, larger tiles:
    python experiments/visualize_swatch_table.py \\
      --table runs/clevr_swatch_table.pt --shape cube --tile_inches 2.0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets.clevr_dataset import COLORS, MATERIALS, SHAPES, SIZES

# Must match the nested-loop order in extract_clevr_swatch_table exactly.
_COMBOS = [
    (color, shape, material, size)
    for color in COLORS
    for shape in SHAPES
    for material in MATERIALS
    for size in SIZES
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="runs/clevr_swatch_table.pt")
    parser.add_argument("--out", default=None, help="Defaults to <table>.png next to the table.")
    parser.add_argument("--tile_inches", type=float, default=1.2, help="Size of each swatch tile in the figure.")
    parser.add_argument("--shape", default=None, choices=SHAPES, help="Only show this shape.")
    parser.add_argument("--color", default=None, choices=COLORS, help="Only show this color.")
    parser.add_argument("--material", default=None, choices=MATERIALS, help="Only show this material.")
    parser.add_argument("--size", default=None, choices=SIZES, help="Only show this size.")
    parser.add_argument("--cols", type=int, default=12, help="Grid columns (ignored if --shape is set).")
    args = parser.parse_args()

    table = torch.load(args.table, map_location="cpu")
    assert table.shape[0] == len(_COMBOS), (
        f"Table has {table.shape[0]} entries, expected {len(_COMBOS)} "
        f"({len(COLORS)}x{len(SHAPES)}x{len(MATERIALS)}x{len(SIZES)}) - "
        "wrong file, or COLORS/SHAPES/MATERIALS/SIZES changed since it was built."
    )

    keep = []
    for idx, (color, shape, material, size) in enumerate(_COMBOS):
        if args.shape and shape != args.shape:
            continue
        if args.color and color != args.color:
            continue
        if args.material and material != args.material:
            continue
        if args.size and size != args.size:
            continue
        keep.append(idx)

    if not keep:
        print("No swatches match the given filters.")
        return

    cols = min(args.cols, len(keep)) if not args.shape else min(len(MATERIALS) * len(SIZES), len(keep))
    cols = max(cols, 1)
    rows = (len(keep) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * args.tile_inches, rows * (args.tile_inches + 0.35)), squeeze=False
    )

    is_missing = table.abs().amax(dim=(1, 2, 3)) == 0

    for ax_idx, idx in enumerate(keep):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        img = table[idx].clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        color, shape, material, size = _COMBOS[idx]
        label = f"{size[0]}.{material[0]}.{color}\n{shape}"
        if is_missing[idx]:
            label += "\n[MISSING]"
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    for ax_idx in range(len(keep), rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{args.table}  ({len(keep)}/{len(_COMBOS)} shown, {int(is_missing.sum())} missing overall)")
    fig.tight_layout()

    out_path = args.out or (os.path.splitext(args.table)[0] + ".png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {len(keep)} swatches to {out_path}")


if __name__ == "__main__":
    main()
