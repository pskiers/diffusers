"""paper_plots/plot_figure1_teaser.py — Figure 1: the paper's opening
teaser. Two parts:

  (a) A schematic two-row trajectory comparison, both rows built from REAL
      renders of 2 representative denoising steps (shared raw noise, then
      the actual noisy running latent x partway through — NOT an x0 pred,
      which looks confidently sharp from early on regardless of
      correctness) plus "..." (no arrow) leading to a real final step. Top
      row: a properly-trained STANDALONE ~80M-param diffusion model (DM,
      experiments/export_figure1_dm80m_curve.py) — a uniform thick
      "committed" arrow, ending in a plausible-LOOKING but constraint-
      violating sample. Bottom row: our full TRM+painter system (PT,
      experiments/export_figure1_teaser_frames.py) — no arrow at all at
      that first transition, just a small dashed "reasoning" cloud icon
      (schematic — free exploration isn't a one-way commit, unlike DM's
      arrow), ending in a valid sample. Both rows' FIRST thumbnail is the
      same shared min-max-normalized raw noise (they start from identical
      noise).

  (b) A short real filmstrip, spanning the full width below panel (a),
      showing ONLY latent reasoning states (no committed y's — this box is
      about what happens INSIDE the reasoning, which by definition never
      includes a committed y): z1 -> [arrow] -> a middle z -> [...] -> z6,
      plain arrows (no nested cloud icon — panel (a)'s "LR" marker already
      means "reasoning happens here"; this is just what's inside it), 3 of
      the real 6 states from the same genuine dip-then-repair example
      plot_qualitative_dip_example.py uses. Connected to panel (a)'s own
      "LR" marker by a short dashed arrow.

Usage:
    python paper_plots/plot_figure1_teaser.py \\
      --frames runs/figure1_teaser_frames.json \\
      --dm-frames runs/figure1_dm80m_curve.json \\
      --qualitative runs/qualitative_dip_example_v8.json --qualitative-index 1 \\
      --out paper_plots/figure1_teaser.pdf
"""

from __future__ import annotations

import argparse
import base64
import io
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.spines import Spine
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_DM = "#D55E00"
COLOR_PT = "#0072B2"
COLOR_VALID = "#2E7D32"
COLOR_INVALID = "#D55E00"
COLOR_ARROW = "#333333"
COLOR_CLOUD_FILL = "#EDEEF2"
COLOR_CLOUD_EDGE = "#9096A3"


def _decode_png_b64(s: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(s))).convert("L"))


def _draw_thumb(ax, img: np.ndarray, border_color=None, border_lw: float = 2.0) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if border_color is not None:
        h, w = img.shape
        ax.add_patch(Rectangle(
            (0, 0), w - 1, h - 1, fill=False, edgecolor=border_color, linewidth=border_lw,
            clip_on=False, zorder=5,
        ))


def _draw_grid_panel(ax, preds: list, fontsize: float = 6.0) -> None:
    """Plain probe-read digit grid, no violation markup — z is never
    actually painted, and this callout is meant to be a quick pointer, not
    a duplicate of plot_qualitative_dip_example.py's fully-annotated
    version."""
    grid = np.array(preds, dtype=int).reshape(9, 9)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(10):
        lw = 0.7 if i % 3 == 0 else 0.25
        color = "#777777" if i % 3 == 0 else "#cccccc"
        ax.plot([0, 9], [i, i], color=color, linewidth=lw, zorder=1)
        ax.plot([i, i], [0, 9], color=color, linewidth=lw, zorder=1)
    for r in range(9):
        for c in range(9):
            ax.text(c + 0.5, r + 0.5, str(grid[r, c] + 1), ha="center", va="center", fontsize=fontsize, color="#333333", zorder=2)


def _commit_arrow(overlay, x0: float, x1: float, y: float, thick: bool = True) -> None:
    overlay.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13 if thick else 10,
        linewidth=2.2 if thick else 1.3, color=COLOR_ARROW, zorder=4,
    ))


def _lr_marker(overlay, cx: float, cy: float, w: float, h: float) -> None:
    """Small dashed "cloud" box with "LR" inside — free exploration isn't a
    one-way commit, unlike DM's plain arrow at the same spot."""
    overlay.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=" + str(min(w, h) * 0.3),
        facecolor=COLOR_CLOUD_FILL, edgecolor=COLOR_CLOUD_EDGE, linewidth=1.0,
        linestyle=(0, (3, 2)), zorder=3,
    ))
    overlay.text(cx, cy, "LR", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLOR_PT, zorder=4)


def _badge_chip(overlay, cx: float, cy: float, valid: bool, rx: float, ry: float) -> None:
    """A validity chip on the shared overlay, positioned OUTSIDE any
    thumbnail's own bounding box (never overlapping the puzzle image).
    Separate x/y radii: overlay's data coordinates span (0, 1) in both
    directions but are NOT equal-aspect (fig_w != fig_h)."""
    color = COLOR_VALID if valid else COLOR_INVALID
    overlay.add_patch(Ellipse((cx, cy), width=2 * rx, height=2 * ry, facecolor=color, edgecolor="white", linewidth=1.0, zorder=6))
    if valid:
        overlay.plot(
            [cx - rx * 0.5, cx - rx * 0.08, cx + rx * 0.55], [cy + ry * 0.02, cy - ry * 0.4, cy + ry * 0.35],
            color="white", linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=7,
        )
    else:
        overlay.plot([cx - rx * 0.42, cx + rx * 0.42], [cy - ry * 0.42, cy + ry * 0.42], color="white", linewidth=1.5, solid_capstyle="round", zorder=7)
        overlay.plot([cx - rx * 0.42, cx + rx * 0.42], [cy + ry * 0.42, cy - ry * 0.42], color="white", linewidth=1.5, solid_capstyle="round", zorder=7)


def _savefig_tight_excluding_overlay_extent(fig, overlay, out_path: str, pad_inches: float = 0.08, dpi=None) -> None:
    """See plot_qualitative_dip_example.py's identically-named function for
    why this is needed instead of bbox_inches="tight" whenever a full-figure
    invisible overlay axes is used for cross-panel annotation."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        if ax is overlay:
            for child in overlay.get_children():
                if isinstance(child, (Spine, Axis)) or not child.get_visible():
                    continue
                bb = child.get_window_extent(renderer)
                if bb.width > 0 and bb.height > 0:
                    boxes.append(bb)
        else:
            boxes.append(ax.get_tightbbox(renderer))
    tight_px = Bbox.union(boxes)
    tight_in = TransformedBbox(tight_px, fig.dpi_scale_trans.inverted()).padded(pad_inches)
    fig.savefig(out_path, bbox_inches=tight_in, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="export_figure1_teaser_frames.py output JSON (PT row)")
    ap.add_argument("--dm-frames", required=True, help="export_figure1_dm80m_curve.py output JSON (DM row)")
    ap.add_argument("--qualitative", required=True, help="qualitative_dip_example.py output JSON (list of candidates)")
    ap.add_argument("--qualitative-index", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = json.load(open(args.frames))
    dm_frames = json.load(open(args.dm_frames))
    qual = json.load(open(args.qualitative))[args.qualitative_index]

    noise_img = _decode_png_b64(frames["noise_frame"])
    pt_noisy_mid = _decode_png_b64(frames["pt_noisy_frames"]["10"])
    pt_last = _decode_png_b64(frames["pt_frames"]["19"])
    dm_noisy_mid = _decode_png_b64(dm_frames["mid_frames"]["12"])
    dm_last = _decode_png_b64(dm_frames["last_frame"])
    pt_final_valid = frames["pt_final_violations"] == 0
    dm_final_valid = dm_frames["points"][-1]["violations"] == 0

    z_states = [qual["z_list"][0], qual["z_list"][2], qual["z_list"][5]]  # z1, z3, z6

    # ================= layout, in inches =================
    thumb = 1.15
    # gap2 > gap1: the "..." (mid -> final) stands in for far more compressed steps than
    # the one real arrow (noise -> mid) does, so it gets visibly more room, not less.
    gap1_in = 0.45   # noise -> noisy-mid: arrow for DM; just the "LR" marker (no arrow) for PT
    gap2_in = 0.85   # noisy-mid -> final: no arrow, just a much bigger "..."
    chip_zone = 0.5
    label_w = 0.55
    row_h = thumb
    row_gap = 0.22
    top_margin = 0.28
    traj_w = label_w + 3 * thumb + gap1_in + gap2_in + chip_zone

    # bottom band: real z-filmstrip, sized on its own terms (NOT stretched to fill the
    # trajectory rows' full width — that was forcing the z panels far bigger than the
    # digit grids need) and centered under the trajectory thumbnails instead. Same
    # gap1 < gap2 asymmetry as the trajectory rows, for the same reason.
    z_c = 0.62
    callout_gap1_in, callout_gap2_in = 0.32, 0.60
    callout_w = 3 * z_c + callout_gap1_in + callout_gap2_in
    box_pad_x, box_pad_top, box_pad_bottom = 0.14, 0.30, 0.12
    box_w_in = callout_w + 2 * box_pad_x
    box_h_in = z_c + box_pad_top + box_pad_bottom
    bottom_gap = 0.34
    bottom_h = box_h_in

    left_margin, right_margin = 0.08, 0.14
    fig_w = left_margin + traj_w + right_margin
    top_zone_h = row_h * 2 + row_gap
    fig_h = top_margin + top_zone_h + bottom_gap + bottom_h + 0.08

    def fx(v):
        return v / fig_w

    def fy(v):
        return v / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    overlay = fig.add_axes((0, 0, 1, 1))
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.patch.set_visible(False)
    for spine in overlay.spines.values():
        spine.set_visible(False)

    dm_row_y_in = fig_h - top_margin - row_h
    pt_row_y_in = dm_row_y_in - row_gap - row_h
    thumb_x_in = [left_margin + label_w]
    thumb_x_in.append(thumb_x_in[0] + thumb + gap1_in)
    thumb_x_in.append(thumb_x_in[1] + thumb + gap2_in)

    # overlay.text(
    #     fx((thumb_x_in[0] + thumb_x_in[-1] + thumb) / 2), fy(fig_h - 0.06),
    #     "denoising trajectory (20 steps) →", ha="center", va="top", fontsize=8.5, style="italic", color="#555555",
    # )

    def _row(row_y_in, img_first, img_mid, img_last, is_dm, final_valid):
        y = fy(row_y_in + row_h / 2)
        overlay.text(
            fx(left_margin), y, "DM" if is_dm else "PaTh",
            ha="left", va="center", fontsize=9.5, fontweight="bold" if not is_dm else "normal",
            color=COLOR_DM if is_dm else COLOR_PT,
        )

        imgs = [img_first, img_mid, img_last]
        for i in range(3):
            ax = fig.add_axes((fx(thumb_x_in[i]), fy(row_y_in), fx(thumb), fy(row_h)))
            is_last = i == 2
            _draw_thumb(ax, imgs[i], border_color=(COLOR_VALID if final_valid else COLOR_INVALID) if is_last else None)

        # transition 1 (noise -> noisy-mid): DM gets a real committed arrow; PT gets NO
        # arrow at all, just a small "LR" (latent reasoning) marker — free exploration
        # isn't a one-way commit.
        g1x0_in, g1x1_in = thumb_x_in[0] + thumb + 0.04, thumb_x_in[1] - 0.04
        marker_cx_in = (g1x0_in + g1x1_in) / 2
        if is_dm:
            _commit_arrow(overlay, fx(g1x0_in), fx(g1x1_in), y, thick=True)
        else:
            _lr_marker(overlay, fx(marker_cx_in), y, fx(0.32), fy(0.30))

        # transition 2 (noisy-mid -> final): NO arrow, just a big "..." — many steps compressed
        g2cx_in = (thumb_x_in[1] + thumb + thumb_x_in[2]) / 2
        overlay.text(fx(g2cx_in), y, "…", ha="center", va="center", fontsize=22, color="#555555")

        chip_cx_in = thumb_x_in[-1] + thumb + chip_zone / 2 - 0.06
        _badge_chip(overlay, fx(chip_cx_in), y, final_valid, fx(0.07), fy(0.07))
        overlay.text(
            fx(chip_cx_in), y - fy(0.16), "valid" if final_valid else "invalid",
            ha="center", va="top", fontsize=7, color=(COLOR_VALID if final_valid else COLOR_INVALID),
        )
        return marker_cx_in

    _row(dm_row_y_in, noise_img, dm_noisy_mid, dm_last, True, dm_final_valid)
    marker_cx_in = _row(pt_row_y_in, noise_img, pt_noisy_mid, pt_last, False, pt_final_valid)

    # ================= bottom band: real z-filmstrip, centered under the trajectory
    # thumbnails (NOT stretched to their full width) =================
    bottom_y_in = pt_row_y_in - bottom_gap - bottom_h
    thumb_block_w = 3 * thumb + gap1_in + gap2_in
    callout_x0_in = thumb_x_in[0] + (thumb_block_w - callout_w) / 2 - 1

    box_x0_in = callout_x0_in - box_pad_x
    box_y0_in = bottom_y_in
    overlay.add_patch(FancyBboxPatch(
        (fx(box_x0_in), fy(box_y0_in)), fx(box_w_in), fy(box_h_in),
        boxstyle="round,pad=0,rounding_size=0.02", facecolor=COLOR_CLOUD_FILL, edgecolor=COLOR_CLOUD_EDGE,
        linewidth=1.0, linestyle=(0, (3, 2)), zorder=0,
    ))
    overlay.text(
        fx(box_x0_in + box_w_in / 2), fy(box_y0_in + box_h_in - 0.04),
        "latent reasoning", ha="center", va="top", fontsize=9, style="italic", color="#444444", zorder=1,
    )

    z0_bottom_in = box_y0_in + box_pad_bottom
    content_cy_in = z0_bottom_in + z_c / 2

    z_x_in = [callout_x0_in, callout_x0_in + z_c + callout_gap1_in]
    z_x_in.append(z_x_in[1] + z_c + callout_gap2_in)
    for i in range(3):
        ax = fig.add_axes((fx(z_x_in[i]), fy(z0_bottom_in), fx(z_c), fy(z_c)))
        _draw_grid_panel(ax, z_states[i])

    # transition 1 (z1 -> z3): a plain arrow, no nested marker — panel (a)'s "LR" already
    # means "reasoning happens here"; this is just what's inside it.
    c1x0_in, c1x1_in = z_x_in[0] + z_c + 0.05, z_x_in[1] - 0.05
    _commit_arrow(overlay, fx(c1x0_in), fx(c1x1_in), fy(content_cy_in), thick=False)

    # transition 2 (z3 -> z6): NO arrow, just a big "..."
    c2cx_in = (z_x_in[1] + z_c + z_x_in[2]) / 2
    overlay.text(fx(c2cx_in), fy(content_cy_in), "…", ha="center", va="center", fontsize=22, color="#555555")

    # dashed connector: panel (a)'s "LR" marker -> this callout box (what it's actually
    # doing) — same x as "LR" itself (not the box's own center), and starting right at
    # "LR"'s own bottom edge, so the arrow visibly touches the marker it's coming from.
    lr_bottom_in = pt_row_y_in + row_h / 2 - 0.15  # bottom edge of the "LR" box (its own height is 0.30in)
    overlay.add_patch(FancyArrowPatch(
        (fx(marker_cx_in), fy(lr_bottom_in)), (fx(marker_cx_in), fy(box_y0_in + box_h_in + 0.02)),
        arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=COLOR_CLOUD_EDGE, linestyle=(0, (3, 2)), zorder=2,
    ))

    _savefig_tight_excluding_overlay_extent(fig, overlay, args.out)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
