"""paper_plots/plot_clevr_centroid_appendix_examples.py — appendix figure
illustrating the CLEVR-reduced + presence-only-centroid-mask conditioning
setup (configs/experiment/clevr_unet_concat_painter.yaml:
data.mode=reduced, include_centroid_mask=true, centroid_mask_presence_only=
true) for a couple of small scenes.

Three panels per example:
  1. Centroid mask   — the actual 1-channel spatial condition
                        (make_presence_mask_from_scene): WHERE objects are,
                        nothing else (no color/shape/size leaks through it).
  2. Object tokens    — the per-object attribute/relation condition the
                        model receives instead, made human-readable as one
                        table: columns are objects (1..N, matching the
                        centroid mask's blob order), rows are Shape (glyph),
                        Color (real CLEVR fill swatch), Material (M/R),
                        Size (B/S), then the 4 relations (Left of/Right of/
                        In front of/Behind), each cell listing the numbers
                        of the related objects — decoded straight out of the
                        real "reduced" object-feature tensor
                        (datasets.clevr_dataset.make_tensor_from_scene), so
                        what's shown is exactly the sparse, direction-coin-
                        flipped relation info the condition vector encodes,
                        not the full/dense relation graph. A relation cell
                        never holds more than one object number: left/right
                        (and front/behind) come from a strict total order
                        over x (resp. y) coordinates, and the transitive
                        reduction of a total order is always exactly its
                        covering chain — each object has at most one
                        immediate neighbor per direction, so there is
                        nothing to stack (verified empirically: 0 counter-
                        examples across 3000 scenes x 3 seeds).
  3. Solution         — the real rendered CLEVR photo for that scene,
                        center-cropped to square (CLEVR renders at 480x320;
                        a plain resize would squash it non-uniformly).

No re-implementation of the real data pipeline: the mask, the attribute
values and the relation tensor are all produced by datasets/clevr_dataset.py
itself, just decoded back into a readable form for the figure.

Usage (run from examples/trm_diffusion/):
    python paper_plots/plot_clevr_centroid_appendix_examples.py \\
      --out paper_plots/clevr_centroid_appendix_examples.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, Rectangle
from PIL import Image

from datasets.clevr_dataset import (
    MAX_OBJECTS,
    RELATIONS,
    make_presence_mask_from_scene,
    make_tensor_from_scene,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Real CLEVR generation palette (image_generation/data/properties.json in the
# original CLEVR codebase) — fills match what the object actually looks like
# in the rendered photo, not an arbitrary categorical palette.
CLEVR_COLOR_HEX = {
    "gray": "#878787",
    "red": "#AD2323",
    "blue": "#2A4BD7",
    "green": "#1D6914",
    "brown": "#814A19",
    "purple": "#8126C0",
    "cyan": "#29D0D0",
    "yellow": "#FFEE33",
}

RELATION_TITLES = {"left": "Left of", "right": "Right of", "front": "In front of", "behind": "Behind"}

TOKEN_TABLE_ROWS = ["Shape", "Color", "Material", "Size"] + [RELATION_TITLES[r] for r in RELATIONS]


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _load_scenes(clevr_root: str, split: str) -> tuple[list, str]:
    scene_path = os.path.join(clevr_root, "CLEVR_v1.0", "scenes", f"CLEVR_{split}_scenes.json")
    with open(scene_path) as f:
        scenes = json.load(f)["scenes"]
    image_dir = os.path.join(clevr_root, "CLEVR_v1.0", "images", split)
    return scenes, image_dir


def _diversity_score(scene: dict) -> tuple:
    """Favor scenes that show off distinct shapes/materials/colors — a
    figure meant to explain the attribute encoding is more useful if the
    example objects don't all share the same shape."""
    objs = scene["objects"]
    return (
        len({o["shape"] for o in objs}),
        len({o["material"] for o in objs}),
        len({o["color"] for o in objs}),
    )


def _pick_scene(scenes: list, n_objects: int, seed: int, top_k: int = 20):
    candidates = [s for s in scenes if len(s["objects"]) == n_objects]
    if not candidates:
        raise SystemExit(f"No scene with exactly {n_objects} objects found.")
    candidates.sort(key=_diversity_score, reverse=True)
    top = candidates[:top_k]
    return random.Random(seed).choice(top)


def _decode_reduced_relations(scene: dict, seed: int) -> list[dict[str, list[int]]]:
    """Real 'reduced' condition tensor -> per-object {relation: [target idx,...]},
    decoded from the exact 40-dim relation slice each object vector carries
    (see make_tensor_from_scene's mode="reduced" branch): dims 0:10 = left
    targets, 10:20 = right, 20:30 = front, 30:40 = behind, each a one-hot-ish
    row over the (up to MAX_OBJECTS) other objects."""
    scene = dict(scene, mode="reduced")
    random.seed(seed)
    np.random.seed(seed)
    cond, mask = make_tensor_from_scene(scene)
    n = len(scene["objects"])
    out = []
    for i in range(n):
        vec = cond[0, i]
        rel = {}
        for r_idx, name in enumerate(RELATIONS):
            row = vec[15 + r_idx * MAX_OBJECTS: 15 + (r_idx + 1) * MAX_OBJECTS]
            rel[name] = [j for j in range(n) if row[j] > 0.5]
        out.append(rel)
    return out


def _draw_mask_panel(ax, mask: np.ndarray) -> None:
    ax.imshow(mask[0], cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_solution_panel(ax, img: Image.Image) -> None:
    ax.imshow(np.asarray(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


TITLE_FONTSIZE = 16


def _add_panel_title(fig, outer, row: int, col: int, text: str) -> None:
    """Figure-level title anchored to the GRIDSPEC SLOT's top edge (not the
    child axes' own bbox), so all 3 panel titles land at the same height
    even though the token-table axes (aspect="equal") shrinks inside its
    slot while the mask/solution axes (square images) fill theirs — using
    ax.set_title on each independently left them at different heights."""
    pos = outer[row, col].get_position(fig)
    cx = (pos.x0 + pos.x1) / 2
    fig.text(cx, pos.y1 + 0.012, text, ha="center", va="bottom", fontsize=TITLE_FONTSIZE)


def _draw_shape_glyph(ax, cx: float, cy: float, shape: str, r: float = 0.32) -> None:
    """Neutral-color glyph (color is its own table row) for the Shape row.
    Cylinder is the classic technical-drawing "can" glyph: a filled body
    rectangle, a full ellipse cap on top (drawn last, so it reads as a
    rounded lid rather than a flat-topped box), and a bottom arc tracing
    just the front curve of the base."""
    face, edge, lw = "#e3e3e3", "black", 0.9
    if shape == "sphere":
        ax.add_patch(Circle((cx, cy), r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    elif shape == "cube":
        s = r * 1.7
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    else:  # cylinder
        w = r * 1.6
        cap_h = r * 0.7
        body_h = r * 1.5
        top_y = cy - body_h / 2
        bottom_y = cy + body_h / 2
        ax.add_patch(Rectangle((cx - w / 2, top_y), w, body_h, facecolor=face, edgecolor="none", zorder=2))
        ax.plot([cx - w / 2, cx - w / 2], [top_y, bottom_y], color=edge, linewidth=lw, zorder=3)
        ax.plot([cx + w / 2, cx + w / 2], [top_y, bottom_y], color=edge, linewidth=lw, zorder=3)
        ax.add_patch(Arc((cx, bottom_y), w, cap_h, theta1=0, theta2=180, color=edge, linewidth=lw, zorder=3))
        ax.add_patch(Ellipse((cx, top_y), w, cap_h, facecolor=face, edgecolor=edge, linewidth=lw, zorder=4))


TABLE_LABEL_W = 0.85
SWATCH_SIZE = 0.62

# (column title, width in data units, kind). Object attributes first, then
# the 4 relations — wider columns for the longer relation labels so the
# header text isn't cramped.
TOKEN_TABLE_COLS = [
    ("Shape", 1.0, "shape"),
    ("Color", 1.0, "color"),
    ("Material", 1.15, "material"),
    ("Size", 1.0, "size"),
    ("Left of", 1.3, "left"),
    ("Right of", 1.3, "right"),
    ("In front of", 1.65, "front"),
    ("Behind", 1.3, "behind"),
]


def _token_table_layout(n: int, rel: list[dict[str, list[int]]]) -> tuple[list[float], list[float]]:
    """(col_lefts, row_tops) in data units — shared by the actual drawing
    and by main()'s figure sizing, so a 3-object and a 4-object example
    render their tables at the same physical scale (font size is fixed in
    points, so if one table's data-unit box were simply squeezed into the
    same figure height as a taller table, its text would visually crowd)."""
    col_lefts = [TABLE_LABEL_W]
    for _, w, _ in TOKEN_TABLE_COLS:
        col_lefts.append(col_lefts[-1] + w)

    row_tops = [0.0]  # header row
    row_tops.append(1.0)
    for i in range(n):
        max_targets = max((len(rel[i][kind]) for _, _, kind in TOKEN_TABLE_COLS if kind in RELATIONS), default=0)
        row_tops.append(row_tops[-1] + max(1, max_targets))
    return col_lefts, row_tops


def _draw_token_table(ax, objects: list[dict], rel: list[dict[str, list[int]]]) -> None:
    """One table, ROWS = objects (1..N), COLUMNS = Shape (glyph) / Color
    (swatch) / Material (M/R) / Size (B/S) / the 4 relations — the transpose
    of a per-attribute layout, so with only 3-4 objects the table is short
    and wide instead of tall and narrow (matching the square mask/solution
    panels beside it, with far less empty space). A relation cell lists the
    numbers of the related objects one PER LINE (multi-row within the cell)
    — general handling for however many targets a cell has, though in
    practice under mode="reduced" it's never more than one: left/right (and
    front/behind) come from comparing a strict total order (x, resp. y
    coordinates), whose transitive reduction is always exactly its covering
    chain, so there is nothing to stack. Single aspect="equal" axes so the
    Shape/Color patches never get stretched regardless of row/column size."""
    n = len(objects)
    label_w = TABLE_LABEL_W
    col_lefts, row_tops = _token_table_layout(n, rel)
    total_w = col_lefts[-1]
    total_h = row_tops[-1]
    attr_right = col_lefts[4]  # right edge of Size column == left edge of the relation block

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- shading: header row + label column tinted, data rows zebra-striped ----
    header_bg, label_bg, zebra_bg = "#e7e8ec", "#eef0f3", "#f4f5f7"
    ax.add_patch(Rectangle((0, 0), total_w, row_tops[1], facecolor=header_bg, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), label_w, total_h, facecolor=label_bg, edgecolor="none", zorder=0))
    for i in range(n):
        if i % 2 == 1:
            y0, y1 = row_tops[i + 1], row_tops[i + 2]
            ax.add_patch(Rectangle((label_w, y0), total_w - label_w, y1 - y0, facecolor=zebra_bg, edgecolor="none", zorder=0))

    header_cy = row_tops[1] / 2
    for ci, (label, w, kind) in enumerate(TOKEN_TABLE_COLS):
        cx = (col_lefts[ci] + col_lefts[ci + 1]) / 2
        ax.text(cx, header_cy, label, ha="center", va="center", fontsize=12.5, fontweight="bold")

    for i, obj in enumerate(objects):
        y0, y1 = row_tops[i + 1], row_tops[i + 2]
        cy = (y0 + y1) / 2
        rh = y1 - y0
        ax.text(label_w - 0.15, cy, str(i + 1), ha="right", va="center", fontsize=16, fontweight="bold")
        for ci, (label, w, kind) in enumerate(TOKEN_TABLE_COLS):
            cx = (col_lefts[ci] + col_lefts[ci + 1]) / 2
            if kind == "shape":
                _draw_shape_glyph(ax, cx, cy, obj["shape"])
            elif kind == "color":
                s = SWATCH_SIZE
                ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor=CLEVR_COLOR_HEX[obj["color"]],
                                        edgecolor="black", linewidth=0.7, zorder=2))
            elif kind == "material":
                ax.text(cx, cy, "M" if obj["material"] == "metal" else "R", ha="center", va="center", fontsize=13, fontweight="bold")
            elif kind == "size":
                ax.text(cx, cy, "B" if obj["size"] == "large" else "S", ha="center", va="center", fontsize=13, fontweight="bold")
            else:  # a relation column
                targets = rel[i][kind]
                k = len(targets)
                if k == 0:
                    continue
                line_offset = (rh - k) / 2  # center the stack within a taller row
                for line, t in enumerate(targets):
                    ty = y0 + line_offset + line + 0.5
                    ax.text(cx, ty, str(t + 1), ha="center", va="center", fontsize=14, fontweight="bold")

    # ---- gridlines ----
    for y in row_tops:
        emph = y == row_tops[1]
        if y in (0.0, total_h):
            continue  # outer border drawn separately, crisper at the corners
        ax.plot([0, total_w], [y, y], color="#8a8d94" if emph else "#dcdde1",
                 linewidth=1.3 if emph else 0.6, zorder=1)
    for x in col_lefts:
        emph = x == attr_right
        if x in (0.0, total_w):
            continue
        ax.plot([x, x], [0, total_h], color="#8a8d94" if emph else "#dcdde1",
                 linewidth=1.3 if emph else 0.6, zorder=1)
    ax.plot([label_w, label_w], [0, total_h], color="#b3b5bb", linewidth=0.9, zorder=1)
    ax.add_patch(Rectangle((0, 0), total_w, total_h, facecolor="none", edgecolor="#555555", linewidth=1.6, zorder=6))


def _make_example_figure(fig, outer, row: int, scene: dict, rel: list[dict[str, list[int]]], image_dir: str, mask_size: int, show_col_titles: bool) -> None:
    objects = scene["objects"]

    mask = make_presence_mask_from_scene(scene, mask_size=mask_size).numpy()
    img = _center_crop_square(Image.open(os.path.join(image_dir, scene["image_filename"])).convert("RGB"))

    ax_mask = fig.add_subplot(outer[row, 0])
    _draw_mask_panel(ax_mask, mask)

    ax_tokens = fig.add_subplot(outer[row, 1])
    _draw_token_table(ax_tokens, objects, rel)

    ax_sol = fig.add_subplot(outer[row, 2])
    _draw_solution_panel(ax_sol, img)

    if show_col_titles:
        _add_panel_title(fig, outer, row, 0, "Centroid mask")
        _add_panel_title(fig, outer, row, 1, "Object tokens")
        _add_panel_title(fig, outer, row, 2, "Solution")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevr-root", default="cache_dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--object-counts", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--mask-size", type=int, default=32, help="matches image_size=256 // 8 used in training configs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper_plots/clevr_centroid_appendix_examples.pdf")
    args = ap.parse_args()

    scenes, image_dir = _load_scenes(args.clevr_root, args.split)
    chosen = [_pick_scene(scenes, n, args.seed) for n in args.object_counts]
    rels = [_decode_reduced_relations(scene, seed=args.seed) for scene in chosen]

    # Row heights proportional to each table's actual content (a 4-object
    # table needs more vertical room than a 3-object one) so every table
    # renders at the same physical scale instead of the taller one being
    # squeezed into the same row height and crowding its header text.
    table_heights = [_token_table_layout(len(s["objects"]), r)[1][-1] for s, r in zip(chosen, rels)]

    n_rows = len(chosen)
    fig = plt.figure(figsize=(13.5, 0.85 * sum(table_heights)))
    outer = fig.add_gridspec(
        n_rows, 3, width_ratios=[1.0, 2.8, 1.0], height_ratios=table_heights, wspace=0.2, hspace=0.3,
    )

    for row, (scene, rel) in enumerate(zip(chosen, rels)):
        _make_example_figure(fig, outer, row, scene, rel, image_dir, args.mask_size, show_col_titles=(row == 0))

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
