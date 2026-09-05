"""experiments/preview_maze_corruption.py — render the corruption catalogue as PNGs.

No model, no GPU: just the operators from maze_corruption_lib on real AMAZE
boards, so you can eyeball what the probe actually feeds the model before
spending cluster time on it.

Writes, under --out-dir:
    catalogue_board<N>.png   one sheet per board: reference row + one row per
                             mode, one column per level, captioned with sizes.
    add/ wall/ gap/          the individual full-resolution PNGs.
    reference/               the unsolved input and the GT solution per board.
    index.txt                what every file is, plus the measured sizes.

Usage:
    python experiments/preview_maze_corruption.py \
        --data data/amaze/test_maze/square/n13_square_test.parquet \
        --boards 0 1 2 3 --out-dir runs/maze_corruption/samples
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import maze_corruption_lib as mcl
from maze_corruption_recovery_probe import LEVELS


def _decode(x):
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    if isinstance(x, str):
        if x.startswith("data:"):
            x = x.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
    raise TypeError(f"unsupported image column type: {type(x)}")


def build(row) -> mcl.MazeSample | None:
    return mcl.build_maze_sample({
        "metadata": row["metadata"],
        "sol_img": _decode(row["sol_img"]),
        "m_original_img": _decode(row["m_original_img"]),
        "cell_map": _decode(row["cell_map"]),
    })


def corrupt(ms: mcl.MazeSample, mode: str, level: float, rng: random.Random) -> tuple:
    """(image, caption) for one (mode, level) on one board."""
    if mode == "gap":
        # GAP always cuts into the FULL solution, so its level is the fraction
        # erased, not a context fraction like ADD/WALL.
        base, shown = mcl.render_partial(ms, 1.0)
        img, erased = mcl.apply_gap(base, ms, shown, rng, gap_frac=level)
        return img, f"erased {len(erased)}/{len(shown)} cells"
    base, shown = mcl.render_partial(ms, level)
    if mode == "add":
        img, wrong, gt_walked, diverge = mcl.apply_add(base, ms, shown, rng)
        if not wrong:
            return img, f"prefix {len(shown)} — NO wrong path possible"
        return img, f"prefix {len(shown)} +{diverge} gt-walk -> {len(wrong)} wrong cells"
    img, line, in_wall, off_path = mcl.apply_shortcut(base, ms, shown, rng)
    return img, (f"prefix {len(shown)}, {int(off_path.sum())} off-path px, "
                 f"{int(in_wall.sum())} through walls")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/amaze/test_maze/square/n13_square_test.parquet")
    ap.add_argument("--boards", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out-dir", default="runs/maze_corruption/samples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ("reference", *LEVELS):
        (out / sub).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    rng = random.Random(args.seed)
    index = [f"source: {args.data}", f"seed: {args.seed}", ""]

    for b in args.boards:
        if b >= len(df):
            print(f"board {b} out of range ({len(df)} rows) — skipped")
            continue
        ms = build(df.iloc[b])
        if ms is None:
            print(f"board {b}: not a usable maze sample — skipped")
            continue
        index += [f"board {b}: {ms.width}x{ms.height}, GT path {len(ms.path)} cells, "
                  f"image {ms.sol.shape[1]}x{ms.sol.shape[0]}"]

        Image.fromarray(ms.morig).save(out / "reference" / f"board{b}_input.png")
        Image.fromarray(ms.sol).save(out / "reference" / f"board{b}_gt.png")

        rows = [("reference", [("input (m_original_img)", ms.morig, "what the model is conditioned on"),
                               ("GT solution (sol_img)", ms.sol, f"{len(ms.path)} cells")])]
        for mode in LEVELS:
            tiles = []
            for lv in LEVELS[mode]:
                img, caption = corrupt(ms, mode, lv, rng)
                name = f"board{b}_p{int(lv * 100)}.png"
                Image.fromarray(img).save(out / mode / name)
                tiles.append((f"{mode.upper()} {int(lv * 100)}%", img, caption))
                index.append(f"  {mode}/{name:<22} {caption}")
            rows.append((mode, tiles))
        index.append("")

        ncol = max(len(t) for _, t in rows)
        fig, axes = plt.subplots(len(rows), ncol, figsize=(3.2 * ncol, 3.55 * len(rows)), squeeze=False)
        for ri, (_, tiles) in enumerate(rows):
            for ci in range(ncol):
                ax = axes[ri][ci]
                ax.set_xticks([]); ax.set_yticks([])
                if ci >= len(tiles):
                    ax.axis("off")
                    continue
                title, img, caption = tiles[ci]
                ax.imshow(Image.fromarray(img).resize((420, 420)))
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel(caption, fontsize=8)
        fig.suptitle(f"Wrong-path corruption catalogue — board #{b} "
                     f"({ms.width}x{ms.height}, GT path {len(ms.path)} cells)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"catalogue_board{b}.png", dpi=100)
        plt.close(fig)
        print(f"board {b}: GT path {len(ms.path)} cells -> catalogue_board{b}.png")

    (out / "index.txt").write_text("\n".join(index))
    print(f"\nwrote {sum(1 for _ in out.rglob('*.png'))} PNGs -> {out.resolve()}")


if __name__ == "__main__":
    main()
