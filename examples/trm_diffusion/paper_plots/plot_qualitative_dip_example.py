"""paper_plots/plot_qualitative_dip_example.py — Figure-1-style illustration
of ONE concrete dip example found by experiments/qualitative_dip_example.py:
x_t -> the L_cycles z states -> x_{t-1}, laid out as TWO stacked rows (a
single wide row crammed all 8 panels side by side and became unreadably
squeezed): a BOTTOM row with the two real MNIST images (x_t --arrow--> a
dashed "think" marker --arrow--> x_{t-1}), and a TOP "Latent Reasoning"
box containing the z_1..z_n sequence, connected to the marker by a single
dashed vertical arrow — the same visual grammar as Figure 1's teaser (a
compact inline marker on the real trajectory, with the actual reasoning
detail broken out into its own box elsewhere), so the two figures read
consistently.

x_t/x_{t-1} are real painter-rendered MNIST images (lm_head -> run_painter
-> CFG combine -> decode_for_eval). Every panel is annotated for constraint
violations the same way: a red border (inset, so adjacent violating boxes
don't double up their shared edge) around any violating 3x3 box, plus a red
x just to the left/below the panel for a violating row/column — read off
the ACTUAL rendered pixels by the real MNIST classifier for x_t/x_{t-1}, so
the image and its markup always agree. The 6 z states are never painted (z
is not a real, committed, paintable state during actual inference — only x
is), so they are shown instead as plain digit grids read straight off the
trained linear probe (models/digit_probe.py), with the same violation
criteria computed on the probe's predicted digits.

experiments/qualitative_dip_example.py saves a LIST of up to top_k
candidates, not just one — use --list to see what's available, then
--index to pick whichever one actually looks best.

Usage:
    python paper_plots/plot_qualitative_dip_example.py --input runs/qualitative_dip_example.json --list
    python paper_plots/plot_qualitative_dip_example.py \\
      --input runs/qualitative_dip_example.json --index 0 \\
      --out paper_plots/qualitative_dip_example.pdf
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.spines import Spine
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_VIOLATION = "#D55E00"
COLOR_VALID = "#2E7D32"
COLOR_BOX_FILL = "#f3ecfb"
COLOR_BOX_EDGE = "#8b3fd9"
COLOR_ARROW = "#333333"


def _decode_png_b64(s: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(s))).convert("L"))


def _violation_rows_cols_boxes(preds: list) -> tuple:
    """rows/cols: sorted lists of indices (0-8) containing a duplicate
    digit. boxes: list of (box_row, box_col) indices (0-2, 0-2) whose 3x3
    box does."""
    grid = np.array(preds, dtype=int).reshape(9, 9)
    rows, cols, boxes = [], [], []
    for i in range(9):
        if len(set(grid[i, :].tolist())) != 9:
            rows.append(i)
        if len(set(grid[:, i].tolist())) != 9:
            cols.append(i)
    for br in range(3):
        for bc in range(3):
            block = grid[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3]
            if len(set(block.flatten().tolist())) != 9:
                boxes.append((br, bc))
    return rows, cols, boxes


def _draw_box_violations(ax, preds: list, cell_size: float, border_lw: float = 2.0, inset_frac: float = 0.12) -> None:
    """Red border around any 3x3 box containing a duplicate digit. Row/col
    violations are NOT bordered here (a border spanning a whole row/column
    reads ambiguously once boxes are the only bordered unit) — see
    _mark_row_col_violations, which places a red x outside the panel
    instead, at the violated row's/column's position.

    Each box's border is drawn INSET from the true 3x3 boundary by
    inset_frac * cell_size, rather than exactly on it. Drawing exactly on
    the boundary has two problems: (1) two ADJACENT violating boxes share
    an edge, so that edge gets double the linewidth of a non-shared edge —
    inconsistent apparent thickness makes it hard to tell at a glance which
    boxes are actually violating; (2) a box on the outer edge of the grid
    has its border coincide with the axes boundary, where clip_on silently
    cuts off the outward half of the stroke, making it look thinner than
    interior borders. Insetting gives every violating box its own fully
    visible, equally-thick, non-overlapping outline."""
    _, _, boxes = _violation_rows_cols_boxes(preds)
    m = cell_size * inset_frac
    for (br, bc) in boxes:
        ax.add_patch(Rectangle(
            (bc * 3 * cell_size + m, br * 3 * cell_size + m), 3 * cell_size - 2 * m, 3 * cell_size - 2 * m,
            fill=False, edgecolor=COLOR_VIOLATION, linewidth=border_lw, zorder=5, clip_on=False,
        ))


def _mark_row_col_violations(
    overlay, preds: list, bbox: tuple, gap_x: float, gap_y: float, markersize: float = 5.0, markeredgewidth: float = 1.5,
) -> None:
    """A red x just to the LEFT of the panel (at the violated row's height)
    for each violating row, and just BELOW the panel (at the violated
    column's position) for each violating column. Drawn on the shared
    full-figure `overlay` axes (the marks fall outside the panel's own
    bounds) — `bbox` = (x0, y0, w, h) of the panel in figure-fraction
    coordinates; panels are always laid out with row 0 at the top / column
    0 at the left, matching a 9x9 grid read top-to-bottom, left-to-right.

    gap_x/gap_y are separate figure-fraction offsets (not a single shared
    value) because the figure is much wider than it is tall: the same
    fractional gap is a very different PHYSICAL distance horizontally vs.
    vertically, which used to make the row marks (offset horizontally) look
    much farther from the grid than the column marks (offset vertically)
    even though both used the "same" gap. Callers should derive gap_x/gap_y
    from the SAME physical inch distance divided by fig_w and fig_h
    respectively, so both sets of marks sit equally close to the grid."""
    x0, y0, w, h = bbox
    rows, cols, _ = _violation_rows_cols_boxes(preds)
    for r in rows:
        y = y0 + h * (1 - (r + 0.5) / 9)
        overlay.plot(
            x0 - gap_x, y, marker="x", color=COLOR_VIOLATION, markersize=markersize,
            markeredgewidth=markeredgewidth, zorder=6, clip_on=False,
        )
    for c in cols:
        x = x0 + w * ((c + 0.5) / 9)
        overlay.plot(
            x, y0 - gap_y, marker="x", color=COLOR_VIOLATION, markersize=markersize,
            markeredgewidth=markeredgewidth, zorder=6, clip_on=False,
        )


def _draw_image_panel(ax, img: np.ndarray, preds: list, border_lw: float = 2.0) -> None:
    # aspect="auto": our axes boxes aren't square (manual figure-fraction
    # layout), and imshow's default aspect="equal" would letterbox the image
    # inside the box rather than filling it — leaving the actual pixel
    # content offset from the (x0, y0, w, h) bbox the row/col violation
    # marks are computed against, and misaligning the marks from the grid.
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cell_size = img.shape[0] / 9.0
    _draw_box_violations(ax, preds, cell_size, border_lw)


def _draw_grid_panel(ax, preds: list, border_lw: float = 1.4) -> None:
    """z is never painted (see module docstring) — shown instead as a plain
    digit grid read straight off the probe, with the same violation
    criteria as the y image panels, computed on these same digits."""
    grid = np.array(preds, dtype=int).reshape(9, 9)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    ax.set_aspect("auto")  # same reason as _draw_image_panel: keep content filling the (x0, y0, w, h) bbox exactly
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(10):
        lw = 1.2 if i % 3 == 0 else 0.4
        color = "#777777" if i % 3 == 0 else "#cccccc"
        ax.plot([0, 9], [i, i], color=color, linewidth=lw, zorder=1)
        ax.plot([i, i], [0, 9], color=color, linewidth=lw, zorder=1)
    for r in range(9):
        for c in range(9):
            ax.text(
                c + 0.5, r + 0.5, str(grid[r, c] + 1), ha="center", va="center",
                fontsize=6.5, color="#333333", zorder=2,
            )
    _draw_box_violations(ax, preds, 1.0, border_lw)


def _savefig_tight_excluding_overlay_extent(fig, overlay, out_path: str, pad_inches: float = 0.1, dpi=None) -> None:
    """Same effect as savefig(bbox_inches="tight"), but computed manually to
    work around a matplotlib quirk: Axes.get_tightbbox() unconditionally
    includes the axes' own full window extent (get_window_extent()) in
    addition to its children's — true regardless of patch/spine visibility,
    since it's hardcoded in matplotlib's own implementation, not something
    set_visible(False) can opt out of. For every other (real, content-sized)
    axes in this figure that's harmless, but `overlay` spans the entire
    (0, 0)-(1, 1) figure by design (so its data coordinates can address any
    panel), so its "own extent" IS the whole canvas — the ordinary tight
    bbox calculation would keep the full nominal fig_h even though every
    actually-visible artist sits well within a much smaller vertical span.
    Fix: union overlay's individual CHILD artists' extents (skipping its
    Spine/Axis children, which report the axes' own full extent too, for
    the same reason) instead of overlay.get_tightbbox() itself."""
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
    ap.add_argument("--input", required=True, help="qualitative_dip_example.py output JSON")
    ap.add_argument("--out", required=False, help="required unless --list is given")
    ap.add_argument(
        "--index", type=int, default=0,
        help="which candidate to render, if --input is a list of candidates (0 = best-scoring; default 0). "
             "Use --list to see what's available first.",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="print each candidate's index/score/wrong-cell counts and exit, without rendering anything",
    )
    ap.add_argument(
        "--dpi", type=int, default=None,
        help="raster resolution for non-vector --out formats (e.g. .png); useful when the output will be "
             "re-embedded as a raster inset elsewhere (e.g. Figure 1's teaser) and needs to stay crisp at a "
             "small display size. No effect on vector formats like .pdf. Default: matplotlib's own default.",
    )
    args = ap.parse_args()
    if not args.list and not args.out:
        raise SystemExit("--out is required unless --list is given")

    with open(args.input) as f:
        loaded = json.load(f)
    # qualitative_dip_example.py now returns a LIST of top_k candidates (older runs saved a
    # single dict) — accept both so old JSONs still render.
    candidates = loaded if isinstance(loaded, list) else [loaded]

    if args.list:
        for i, c in enumerate(candidates):
            print(
                f"[{i}] score={c['score']:.2f}  step={c['step_idx']} k={c['k']} h={c['h']}  "
                f"y_prev_wrong={c['y_prev_wrong']} y_next_wrong={c['y_next_wrong']} max_z_wrong={c['max_z_wrong']}"
            )
        return

    if not (0 <= args.index < len(candidates)):
        raise SystemExit(f"--index {args.index} out of range: {args.input} has {len(candidates)} candidate(s).")
    d = candidates[args.index]
    if "images" not in d:
        raise SystemExit(
            "Selected candidate has no 'images' field — re-run experiments/qualitative_dip_example.py "
            "(it re-renders every candidate through the real painter)."
        )

    images = d["images"]
    y_prev_img = _decode_png_b64(images["y_prev"])
    y_next_img = _decode_png_b64(images["y_next"])
    n_z = len(d["z_list"])

    # ---- layout computed in PHYSICAL INCHES, then converted to the
    # figure-fraction coordinates fig.add_axes()/the overlay need. Doing it
    # in inches (rather than guessing fractions of an arbitrary fig_w/fig_h)
    # is what keeps every panel a true square regardless of how wide the
    # whole figure ends up being with n_z z-panels, and what keeps the row
    # marks (offset horizontally) and column marks (offset vertically)
    # equally close to the grid — a shared FRACTIONAL gap is a very
    # different physical distance along a much-wider-than-tall figure.
    #
    # TWO rows, not one: cramming x_t, the whole z-box, and x_{t-1} into a
    # single row (the old layout) forced the figure very wide and very
    # short, squeezing every panel/label/font down to fit the height. Here
    # the real MNIST images sit in their own bottom row (x_t --arrow--> a
    # small "LR" marker --arrow--> x_{t-1}), and the z_1..z_n sequence gets
    # its own "Latent Reasoning" box above, connected to the LR marker by a
    # single dashed vertical arrow — same amount of content, a much less
    # extreme aspect ratio. ----
    y_size = 1.5        # x_t / x_{t-1} panel side, inches — always square
    z_size = 1.05        # z_i panel side, inches — always square
    left_margin, right_margin = 0.5, 0.5
    top_margin = 0.1
    row_gap = 0.75        # x_t/x_{t-1} edge -> marker edge (room for an arrow each side)
    marker_w, marker_h = 1.3, 0.7
    z_gap = 0.22          # gap between adjacent z panels — room for the z_i row marks, see z_mark_gap_in below
    box_pad_x = 0.14
    box_pad_top = 0.40    # room for each z panel's title + the "latent reasoning" label above the box
    box_pad_bottom = 0.28  # room for the z panels' column-violation x marks + "N violations" caption below them
    box_bottom_margin = 0.15  # box's own bottom edge, from the figure bottom
    gap_v = 0.4           # box top -> bottom row's bottom (room for the vertical dashed arrow + the y panels' caption)
    mark_gap_in = 0.09    # physical distance from a y panel's edge to its violation x marks
    z_mark_gap_in = 0.06  # same, but for the much-more-tightly-packed z panels
    z_viol_gap_in = 0.15  # z panel bottom edge -> "N violations" caption
    y_viol_gap_in = 0.24  # y panel bottom edge -> "N violations" caption
    start_size = z_size   # the x_t-preds "start" panel, outside/left of the box — same size as a z panel
    # start panel's right edge -> box's left edge, chosen so the VISIBLE gap start->z1
    # (this, plus box_pad_x to cross the border into the box) equals the z1->z2 gap (z_gap)
    start_gap_in = z_gap - box_pad_x + 0.25

    z_area_width = n_z * z_size + (n_z - 1) * z_gap
    box_width = z_area_width + 2 * box_pad_x
    row_width = 2 * y_size + 2 * row_gap + marker_w
    start_block_w = start_size + start_gap_in  # reserved on the far left, pushing the box (and row, centered over it) right

    content_w = max(box_width, row_width)
    fig_w = left_margin + start_block_w + content_w + right_margin
    base_left_in = left_margin + start_block_w
    start_x_in = left_margin
    box_left_in = base_left_in + (content_w - box_width) / 2
    box_right_in = box_left_in + box_width
    row_left_in = base_left_in + (content_w - row_width) / 2

    y_before_x_in = row_left_in
    marker_left_in = y_before_x_in + y_size + row_gap
    y_after_x_in = marker_left_in + marker_w + row_gap
    marker_cx_in = marker_left_in + marker_w / 2

    # box (z_1..z_n) sits at the BOTTOM, the real MNIST row on TOP — arrow from
    # x_t runs into the "think" marker, x_{t-1} out of it, with a dashed
    # vertical arrow dropping from the marker down into the box below.
    box_y0_in = box_bottom_margin
    z_area_left_in = box_left_in + box_pad_x
    z_row_bottom_in = box_y0_in + box_pad_bottom
    box_y1_in = z_row_bottom_in + z_size + box_pad_top
    z_xs_in = [z_area_left_in + i * (z_size + z_gap) for i in range(n_z)]

    row_bottom_in = box_y1_in + gap_v
    row_center_in = row_bottom_in + y_size / 2
    fig_h = top_margin + row_bottom_in + y_size

    def fx(v_in: float) -> float:
        return v_in / fig_w

    def fy(v_in: float) -> float:
        return v_in / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    overlay = fig.add_axes((0, 0, 1, 1))
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    # NOT set_zorder(10): that would draw the WHOLE overlay (including the
    # box's solid fill) on top of every other axes in the figure regardless
    # of creation order, hiding the z-panel axes created afterward underneath
    # it. Default stacking (later-created axes draw on top) already gives
    # the right order: box+arrows (created here, first) below, y/z panel
    # axes (created below) on top.
    # set_visible(False), not just set_alpha(0): alpha only affects the RENDERED color, not
    # whether savefig(bbox_inches="tight") counts this axes' full (0,0)-(1,1) rectangle toward
    # the figure's tight bbox — with only alpha=0, the patch stayed "visible" as far as bbox
    # accounting was concerned, forcing the saved figure to keep the whole nominal fig_h even
    # though every actual (visible) artist sits well within a much smaller vertical span.
    overlay.patch.set_visible(False)
    # axis("off") hides ticks/labels but NOT the 4 spine artists themselves — they stay
    # get_visible()==True, each spanning the axes' full (0,0)-(1,1) extent, and get_tightbbox()
    # counts them just like the patch above did. This was the actual remaining cause of the
    # large empty margin (mostly above, a little below) even after hiding the patch.
    for spine in overlay.spines.values():
        spine.set_visible(False)

    row_center = fy(row_center_in)

    # ---- "reasoning" box, above the bottom row ----
    box = FancyBboxPatch(
        (fx(box_left_in), fy(box_y0_in)),
        fx(box_right_in) - fx(box_left_in), fy(box_y1_in) - fy(box_y0_in),
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=COLOR_BOX_FILL, edgecolor=COLOR_BOX_EDGE, linewidth=1.2,
        linestyle=(0, (5, 3)), zorder=1,
    )
    overlay.add_patch(box)
    overlay.text(
        fx((box_left_in + box_right_in) / 2), fy(box_y1_in) - 0.03,
        "Latent Reasoning", ha="center", va="top", fontsize=11, style="italic", color="#444444", zorder=6,
    )

    # ---- bottom row: x_t -> LR marker -> x_{t-1} ----
    def _arrow(x0_in, x1_in, y):
        overlay.add_patch(FancyArrowPatch(
            (fx(x0_in), y), (fx(x1_in), y), arrowstyle="-|>", mutation_scale=16,
            linewidth=1.8, color=COLOR_ARROW, zorder=4,
        ))

    _arrow(y_before_x_in + y_size + 0.05, marker_left_in - 0.05, row_center)
    _arrow(marker_left_in + marker_w + 0.05, y_after_x_in - 0.05, row_center)

    marker_box = FancyBboxPatch(
        (fx(marker_left_in), fy(row_center_in - marker_h / 2)), fx(marker_w), fy(marker_h),
        boxstyle="round,pad=0,rounding_size=0.01",
        facecolor=COLOR_BOX_FILL, edgecolor=COLOR_BOX_EDGE, linewidth=1.1,
        linestyle=(0, (3, 2)), zorder=5,
    )
    overlay.add_patch(marker_box)
    overlay.text(
        fx(marker_cx_in), fy(row_center_in), "Think", ha="center", va="center",
        fontsize=13, fontweight="bold", style="italic", color="#0072B2", zorder=6,
    )

    # ---- dashed vertical connector: "think" marker -> reasoning box below ----
    overlay.add_patch(FancyArrowPatch(
        (fx(marker_cx_in), fy(row_center_in - marker_h / 2)), (fx(marker_cx_in), fy(box_y1_in)),
        arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=COLOR_BOX_EDGE,
        linestyle=(0, (3, 2)), zorder=2,
    ))

    # ---- image panels (y: real painted image; z: probe-read digit grid) ----
    gap_x, gap_y = fx(mark_gap_in), fy(mark_gap_in)

    def _viol_caption(cx_in, n_viol, fontsize=10):
        overlay.text(
            fx(cx_in), fy(row_bottom_in - y_viol_gap_in), f"{n_viol} violations", ha="center", va="top",
            fontsize=fontsize, color=COLOR_VALID if n_viol == 0 else COLOR_VIOLATION, zorder=6, fontweight="bold"
        )

    y_before_bbox = (fx(y_before_x_in), fy(row_bottom_in), fx(y_size), fy(y_size))
    ax_y_before = fig.add_axes(y_before_bbox)
    _draw_image_panel(ax_y_before, y_prev_img, d["y_prev"], border_lw=2.6)
    ax_y_before.set_title(r"$x_t$", fontsize=13, pad=4)
    _mark_row_col_violations(overlay, d["y_prev"], y_before_bbox, gap_x, gap_y, markersize=6.0)
    _viol_caption(y_before_x_in + y_size / 2, int(d["y_prev_violations"]))

    y_after_bbox = (fx(y_after_x_in), fy(row_bottom_in), fx(y_size), fy(y_size))
    ax_y_after = fig.add_axes(y_after_bbox)
    _draw_image_panel(ax_y_after, y_next_img, d["y_next"], border_lw=2.6)
    ax_y_after.set_title(r"$x_{t-1}$", fontsize=13, pad=4)
    _mark_row_col_violations(overlay, d["y_next"], y_after_bbox, gap_x, gap_y, markersize=6.0)
    _viol_caption(y_after_x_in + y_size / 2, int(d["y_next_violations"]))

    z_gap_x, z_gap_y = fx(z_mark_gap_in), fy(z_mark_gap_in)

    # ---- "start" panel: x_t's own predicted digits, outside/left of the box — the
    # state the z_1..z_n reasoning sequence begins from, drawn exactly like a z panel. ----
    start_bbox = (fx(start_x_in), fy(z_row_bottom_in), fx(start_size), fy(start_size))
    ax_start = fig.add_axes(start_bbox)
    _draw_grid_panel(ax_start, d["y_prev"], border_lw=1.8)
    ax_start.set_title("start", fontsize=10, pad=3, color="#444444", fontweight="bold")
    _mark_row_col_violations(overlay, d["y_prev"], start_bbox, z_gap_x, z_gap_y, markersize=3.6, markeredgewidth=0.9)
    start_n_viol = int(d["y_prev_violations"])
    overlay.text(
        fx(start_x_in + start_size / 2), fy(z_row_bottom_in - z_viol_gap_in),
        f"{start_n_viol} violations", ha="center", va="top", fontsize=10,
        color=COLOR_VALID if start_n_viol == 0 else COLOR_VIOLATION, zorder=6, fontweight="bold"
    )

    z_violations = d.get("z_violations")
    for i in range(n_z):
        z_bbox = (fx(z_xs_in[i]), fy(z_row_bottom_in), fx(z_size), fy(z_size))
        ax_z = fig.add_axes(z_bbox)
        _draw_grid_panel(ax_z, d["z_list"][i], border_lw=1.8)
        ax_z.set_title(rf"$z_{{{i + 1}}}$", fontsize=10, pad=3, color="#444444")
        _mark_row_col_violations(overlay, d["z_list"][i], z_bbox, z_gap_x, z_gap_y, markersize=3.6, markeredgewidth=0.9)
        if z_violations is not None:
            n_viol = int(z_violations[i])
            overlay.text(
                fx(z_xs_in[i] + z_size / 2), fy(z_row_bottom_in - z_viol_gap_in),
                f"{n_viol} violations", ha="center", va="top", fontsize=10,
                color=COLOR_VALID if n_viol == 0 else COLOR_VIOLATION, zorder=6, fontweight="bold"
            )

    _savefig_tight_excluding_overlay_extent(fig, overlay, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
