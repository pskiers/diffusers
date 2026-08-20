"""
eval/squares_eval.py – Evaluation for the Inscribed Square dataset
(datasets/squares_dataset.py).

2026-08-12: this now calls eval_squares_metrics directly — a verbatim
vendored copy of the paper's own scripts/eval_squares.py, recovered from the
commit where that script was last self-contained and actually ran (see that
module's header for why the current HEAD version can't be used at all: a
later refactor left it with a broken self-import). evaluate_squares() below
is a direct port of that script's __main__ per-sample loop, not a
reimplementation.

This is a substantive behavior change, not just a formula swap: the real
pipeline snaps the generated square to the curve via a rigid-transform
search (rotation ±15°, translation ±5px — snap_square_by_rigid_transform)
*before* scoring squareness/alignment, on by default (`--no-snap` is opt-
out). The previous version of this module scored the raw generated mask
directly, with no such search, and its alignment formula looked up the
distance transform at the box corner's own (rounded) position — the real
pipeline instead snaps each corner to its *nearest actual foreground pixel*
in the mask first (compute_alignment_score_from_box), then looks up the
distance transform there. Both differences make scores from this module
non-comparable to numbers logged before this fix.

squareness/alignment scores need only the generated square + the curve
condition — no ground-truth square tensor — so there's still no "predicted
vs. target" comparison and no best-of-N/valid-discard step (a generated
square is always scored, never thrown out) the way
SteinerEvalCallback/PolygonEvalCallback do for their binary constraint
checks.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from eval.eval_squares_metrics import (
    _to_numpy255,
    compute_alignment_score_from_box,
    get_box,
    snap_square_by_rigid_transform,
    squareness_metric,
)


def to_binary_mask(img: torch.Tensor) -> np.ndarray:
    """(1, H, W) or (H, W) tensor in [-1, 1] -> (H, W) uint8 mask, 255 where
    pixel < 0 (foreground/drawn), 0 elsewhere — same threshold-at-0 rule as
    eval_squares_metrics._to_numpy255, just accepting a torch tensor
    directly (that one only accepts what its own _to_numpy01 already
    handles, which is the same tensor-or-array + squeeze logic; kept as a
    separate thin wrapper here only so callers outside evaluate_squares
    — make_squares_panel_image — don't need to reach into eval_squares_metrics)."""
    return _to_numpy255(img)


@torch.no_grad()
def evaluate_squares(
    generated: torch.Tensor,   # (B, 1, H, W) float, generated squares, in [-1, 1]-ish
    conditions: torch.Tensor,  # (B, 1, H, W) float, curve condition, in [-1, 1]-ish
) -> dict:
    """Per-sample squareness (Eq. 2) + alignment (Eq. 1), directly comparable
    to the paper's Table 1 Square↑/Align↑ columns (mean over a batch).

    Returns dict with keys:
      squareness_mean, alignment_mean — batch means. alignment_mean is in
        raw pixel units and typically negative (see compute_alignment_score_from_box's
        avg_dist, negated to keep "higher = better").
      per_sample_squareness, per_sample_alignment — (B,) numpy arrays.
    """
    B = generated.shape[0]
    squareness = np.zeros(B, dtype=np.float64)
    alignment = np.zeros(B, dtype=np.float64)

    for b in range(B):
        curve_mask = _to_numpy255(conditions[b])
        dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

        # Direct port of scripts/eval_squares.py's __main__ (w_snap branch,
        # the default): rigid-search the generated mask onto the curve
        # *before* scoring, rather than scoring the raw generated output.
        square_mask, _ = snap_square_by_rigid_transform(generated[b], conditions[b])

        box = get_box(square_mask)
        if box is None:
            squareness[b] = 0.0
            alignment[b] = float("-inf")
            continue

        _, _, _, avg_dist = compute_alignment_score_from_box(box, square_mask, dist_transform)
        squareness[b] = squareness_metric(square_mask)
        alignment[b] = -float(avg_dist)

    return {
        "squareness_mean": float(squareness.mean()),
        "alignment_mean": float(alignment.mean()),
        "per_sample_squareness": squareness,
        "per_sample_alignment": alignment,
    }


def _fitted_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """(4, 2) int32 box points via get_box (approxPolyDP-quad-or-minAreaRect-
    fallback) — the same box compute_alignment_score_from_box actually
    scores corners against — or None if the mask has no foreground pixels.

    Note: this draws the box fit to the RAW (unsnapped) generated mask, for
    a simple visual sanity check against embedding_conditions/the curve as
    given. evaluate_squares() itself scores the mask *after*
    snap_square_by_rigid_transform's rigid search, so the drawn box can look
    slightly less curve-aligned than the actual reported alignment score
    (computed post-snap) — visualizing the post-snap mask would need
    threading that state through models/eval_callbacks.py's panel-logging
    call site, not done here since it doesn't change what's measured, only
    what's drawn."""
    box = get_box(mask)
    if box is None:
        return None
    return box.astype(np.int32)


def make_squares_panel_image(
    condition: torch.Tensor,  # (1, H, W) float in [-1,1] — curve
    generated: torch.Tensor,  # (1, H, W) float in [-1,1] — model output
    reference: torch.Tensor,  # (1, H, W) float in [-1,1] — ground-truth square
    squareness: Optional[float] = None,
    alignment: Optional[float] = None,
) -> np.ndarray:
    """Condition | Generated | Ground Truth | Overlay, each with a title bar,
    plus per-sample squareness/alignment scores as a caption strip.

    The Overlay panel draws the *fitted minAreaRect* (the same box
    squareness_metric/alignment_score actually score, via _fitted_box) of the
    generated square in red and of the ground-truth square in green, both on
    top of the curve — this makes the alignment_score number visually
    checkable at a glance (do the corners actually sit on the curve?),
    instead of requiring a mental cross-reference between separate panels.
    """
    H, W = condition.shape[-2], condition.shape[-1]
    TITLE_H = 18
    CAPTION_H = 18

    def to_gray_uint8(t: torch.Tensor) -> np.ndarray:
        arr = t.clamp(-1, 1).cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        return ((arr + 1.0) * 127.5).astype(np.uint8)

    def to_bgr(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def with_title(img_bgr: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((TITLE_H, img_bgr.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(bar, text, (2, TITLE_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        return np.concatenate([bar, img_bgr], axis=0)

    cond_gray = to_gray_uint8(condition)
    gen_gray = to_gray_uint8(generated)
    ref_gray = to_gray_uint8(reference)

    # Overlay: curve (grayscale) + fitted generated box (red) + fitted GT box (green).
    overlay = to_bgr(cond_gray)
    gen_mask = to_binary_mask(generated)
    ref_mask = to_binary_mask(reference)
    ref_box = _fitted_box(ref_mask)
    if ref_box is not None:
        cv2.polylines(overlay, [ref_box], isClosed=True, color=(0, 200, 0), thickness=1, lineType=cv2.LINE_AA)
    gen_box = _fitted_box(gen_mask)
    if gen_box is not None:
        cv2.polylines(overlay, [gen_box], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    sep = np.full((TITLE_H + H, 4, 3), 200, dtype=np.uint8)
    panels = [
        with_title(to_bgr(cond_gray), "Condition (curve)"),
        sep,
        with_title(to_bgr(gen_gray), "Generated"),
        sep,
        with_title(to_bgr(ref_gray), "Ground Truth"),
        sep,
        with_title(overlay, "Overlay"),
    ]
    grid = np.concatenate(panels, axis=1)

    caption_bits = ["red=generated  green=ground truth"]
    if squareness is not None:
        caption_bits.append(f"squareness={squareness:.3f}")
    if alignment is not None:
        caption_bits.append(f"alignment={alignment:.1f}px")
    if caption_bits:
        caption_bar = np.full((CAPTION_H, grid.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            caption_bar, "  ".join(caption_bits), (2, CAPTION_H - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
        grid = np.concatenate([grid, caption_bar], axis=0)

    return grid
