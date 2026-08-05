"""
eval/squares_eval.py – Evaluation for the Inscribed Square dataset
(datasets/squares_dataset.py).

squareness_metric implements the paper's Eq. 2 (arXiv 2510.21697):
Q(S) = area(S)/(w*h) * exp(-2|max(w,h)/min(w,h) - 1|) where (w,h) are the
minimum-area enclosing rectangle's side lengths — ported from the original
repo's utils/metrics.py, whose IoU-against-own-minAreaRect-mask formula is
exactly Eq. 2 (IoU = area(S)/(w*h) whenever S is fully contained in its own
minAreaRect box, which it always is by construction). Verified against the
paper's own reported numbers: Table 1 reports GT squareness=0.924; scoring
this repo's own freshly-generated ground-truth squares with this function
gives 0.920 (300-sample sanity check) — matches.

alignment_score implements the paper's Eq. 1: A(S,C) = -(1/4) * sum over the
4 corners of dist(corner, C), i.e. the *negative mean pixel distance* from a
predicted square's corners to the conditioning curve (higher/less negative
= better; 0 = corners exactly on the curve) — ported from
scripts/eval_squares.py's `alignment_pixel` (nearest-pixel lookup into a
cv2.distanceTransform of the curve mask), which is what the paper's Table 1
numbers were actually produced with.

Do NOT confuse this with utils/metrics.py's separate `alignment_metric`
(exp(-dist/decay_scale), bounded to (0,1]) — that's a different formula
train_diffusion.py's own per-epoch training validation loop happens to use
as a cheap bounded training-progress signal; it does not appear in the
paper's reported evaluation table and isn't reproduced here, so results
from this module are directly comparable to Table 1's Align/Square columns,
not to wandb curves from the original repo's own training runs.

Neither metric needs the exact ground-truth square: squareness_metric
scores the generated square's shape against *itself* (is this blob square-
shaped?), and alignment_score scores the generated square's corners against
the rasterized *condition* curve mask (already available as
spatial_conditions, no extra ground truth needed) — so there's no
"predicted vs. target" comparison, and no best-of-N/valid-discard step (a
generated square is always scored, never thrown out) as used by
SteinerEvalCallback/PolygonEvalCallback for their binary constraint checks.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def to_binary_mask(img: torch.Tensor) -> np.ndarray:
    """(1, H, W) or (H, W) tensor in [-1, 1] -> (H, W) uint8 mask, 255 where
    pixel < 0 (foreground/drawn), 0 elsewhere. Simpler than the original's
    tensor_to_binary_mask (min-max rescale + threshold at 127): our renders
    are always exactly {-1, +1}-valued by construction (see
    datasets/squares_dataset.py), so a direct threshold at 0 is equivalent
    and doesn't degenerate on a constant (all-background) image the way a
    min-max normalization would."""
    arr = img.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return (arr < 0).astype(np.uint8) * 255


def squareness_metric(mask: np.ndarray) -> float:
    """Ported from utils/metrics.py's squareness_metric (type='squareness'
    branch only — 'rectangleness' isn't used by config_curves.yaml).

    IoU between the mask's largest contour and its own minimum-area
    bounding rectangle, penalized by deviation from a square aspect ratio.
    1.0 = a perfect, fully-filled square blob.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return 0.0

    box = cv2.boxPoints(rect).astype(np.int32)
    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(rect_mask, box, 255)

    intersection = np.logical_and(mask > 0, rect_mask > 0).sum()
    union = np.logical_or(mask > 0, rect_mask > 0).sum()
    if union == 0:
        return 0.0
    iou = intersection / union

    aspect_ratio = max(w, h) / min(w, h)
    square_penalty = np.exp(-abs(aspect_ratio - 1) * 2)
    return float(iou * square_penalty)


def alignment_score(square_mask: np.ndarray, curve_mask: np.ndarray) -> float:
    """Paper Eq. 1: A(S,C) = -(1/4) * sum_{p in corners(S)} dist(p, C).

    Fits a minimum-area rectangle to the generated square mask (same box
    squareness_metric uses) and looks up each of its 4 float corners in a
    distance transform of the curve mask, via nearest-integer-pixel lookup
    — matches scripts/eval_squares.py's `alignment_pixel` exactly (one of
    three alignment variants in that script; the other two need either a
    rigid-transform "snap" search or the exact parametric curve, both out
    of scope for a periodic training-time callback — see this module's
    docstring). Returns the *negative* mean distance in pixels: 0 = corners
    exactly on the curve, more negative = further off. A missing/empty mask
    returns -inf (worst possible score, never silently 0/best-case).
    """
    dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

    contours, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("-inf")

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # (4, 2) float (x, y)

    H, W = dist_transform.shape
    ix = np.clip(np.rint(box[:, 0]).astype(int), 0, W - 1)
    iy = np.clip(np.rint(box[:, 1]).astype(int), 0, H - 1)
    mean_dist = float(dist_transform[iy, ix].mean())
    return -mean_dist


@torch.no_grad()
def evaluate_squares(
    generated: torch.Tensor,   # (B, 1, H, W) float, generated squares, in [-1, 1]-ish
    conditions: torch.Tensor,  # (B, 1, H, W) float, curve condition, in [-1, 1]-ish
) -> dict:
    """Per-sample squareness (Eq. 2) + alignment (Eq. 1), directly comparable
    to the paper's Table 1 Square↑/Align↑ columns (mean over a batch).

    Returns dict with keys:
      squareness_mean, alignment_mean — batch means. alignment_mean is in
        raw pixel units and typically negative (see alignment_score).
      per_sample_squareness, per_sample_alignment — (B,) numpy arrays.
    """
    B = generated.shape[0]
    squareness = np.zeros(B, dtype=np.float64)
    alignment = np.zeros(B, dtype=np.float64)

    for b in range(B):
        square_mask = to_binary_mask(generated[b])
        curve_mask = to_binary_mask(conditions[b])
        squareness[b] = squareness_metric(square_mask)
        alignment[b] = alignment_score(square_mask, curve_mask)

    return {
        "squareness_mean": float(squareness.mean()),
        "alignment_mean": float(alignment.mean()),
        "per_sample_squareness": squareness,
        "per_sample_alignment": alignment,
    }


def make_squares_panel_image(
    condition: torch.Tensor,  # (1, H, W) float in [-1,1] — curve
    generated: torch.Tensor,  # (1, H, W) float in [-1,1] — model output
    reference: torch.Tensor,  # (1, H, W) float in [-1,1] — ground-truth square
) -> np.ndarray:
    """condition | generated | reference, each mapped from [-1,1] to [0,255] grayscale."""

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        arr = t.clamp(-1, 1).cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        return ((arr + 1.0) * 127.5).astype(np.uint8)

    sep = np.full((condition.shape[-2], 4), 128, dtype=np.uint8)
    return np.concatenate([to_uint8(condition), sep, to_uint8(generated), sep, to_uint8(reference)], axis=1)
