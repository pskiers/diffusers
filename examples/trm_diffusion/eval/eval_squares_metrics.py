"""
Vendored VERBATIM from the paper's own repo (kariander1/visual-geo-solver) —
but NOT from the current HEAD of scripts/eval_squares.py, which is broken:
its own top-of-file import (`from scripts.eval_squares import (...)`) pulls
names (`get_box`, `_to_numpy01`, `_to_numpy255`,
`snap_square_by_rigid_transform`, `_largest_component_pts`,
`_min_area_rect`, `_box_points`) that don't exist anywhere in that commit —
a self-import bug introduced by a later refactor ("Added mps support, flow
matching, steiner regression", commit 6991f50) that replaced the file's
body without fixing the import, leaving the script unable to even be
imported, let alone run. The paper's actual reported Table 1 numbers could
not have come from that broken version.

These functions are instead recovered from the *initial commit* (27b7b42),
the last version of scripts/eval_squares.py that was self-contained and
actually ran end-to-end (verified: every name it uses, it defines itself).
Kept byte-for-byte identical to that commit, including its default
evaluation behavior (square-to-curve rigid-transform snapping is ON by
default — see snap_square_by_rigid_transform and its call site in that
commit's __main__ — a step the current eval/squares_eval.py had been
missing entirely: it scored the raw generated mask directly, when the
paper's own methodology first searches over small rotations/translations
for the best-aligning pose *before* scoring squareness/alignment).

Two bugs fixed, not preserved:
- get_box's `if not contours_t: None` is a no-op statement, not a
  `return None` — the function falls through and crashes on `max()` of an
  empty sequence instead of returning None the way its caller
  (`if box is None: continue`) clearly expects. Changed to an actual
  `return None`.
- snap_square_by_rigid_transform's empty-mask early return
  (`return square_mask.copy()`) returns a single array where every other
  path returns a `(mask, score)` tuple — crashes the caller's
  `square_mask, _ = snap_square_by_rigid_transform(...)` unpacking on a
  fully-blank generated square, a real occurrence early in training.
  Changed to `return square_mask.copy(), -np.inf`, matching the normal
  path's shape.
No other line changed.
"""

import cv2
import numpy as np
import torch


def _to_numpy01(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    return (img < 0)   # threshold at 0


def _to_numpy255(img):
    return (_to_numpy01(img) * 255).astype(np.uint8)


def compute_alignment_score_from_box(box, square_mask, dist_transform, decay_scale=7.5):
    """
    Computes alignment score between box corners and curve using distance transform.

    Args:
        box (np.ndarray): 4x2 array of box corners (float or int)
        square_mask (np.ndarray): binary mask from which to find closest real pixels
        dist_transform (np.ndarray): precomputed distance transform from curve mask
        decay_scale (float): decay factor for exponential decay

    Returns:
        float: alignment score (higher is better)
        list: list of distances used for scoring
        list: list of (x, y) points sampled in the mask
    """
    if not np.any(square_mask):
        return 0.0, [], []

    # Extract foreground mask points
    mask_pts = np.column_stack(np.where(square_mask > 0))[:, [1, 0]]  # (x, y) format

    distances = []
    sampled_points = []
    for corner in box:
        dists = np.linalg.norm(mask_pts - corner, axis=1)
        closest = mask_pts[np.argmin(dists)]
        x, y = int(closest[0]), int(closest[1])
        distances.append(dist_transform[y, x])
        sampled_points.append((x, y))

    avg_dist = np.mean(distances)
    # score = float(np.exp(-avg_dist / decay_scale))
    score = -avg_dist

    return score, distances, sampled_points, avg_dist


def squareness_metric(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)

    # Compute min area rectangle (rotation invariant)
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return 0.0

    area = cv2.contourArea(cnt)
    rect_area = w * h
    ratio = area / rect_area

    # Aspect ratio penalty
    aspect = max(w, h) / min(w, h)
    penalty = np.exp(-abs(aspect - 1) * 2)

    return float(ratio * penalty)


def snap_square_by_rigid_transform(
    square_mask, curve_mask,
    angle_range=(-15, 15), angle_step=0.5,
    trans_step=2, trans_range=5,
    lambda_reg: float = 0.0,
):
    """
    Snap the square to the curve using rotation + translation (rigid transform),
    with a single penalization parameter lambda_reg for larger transforms:
        penalty = lambda_reg * ( ||t||^2 + (L * theta)^2 )
    where theta is in radians and L is half the square's diagonal (in pixels).
    """
    square_mask = _to_numpy255(square_mask)
    curve_mask = _to_numpy255(curve_mask)
    h, w = square_mask.shape
    if not np.any(square_mask):
        # bugfix: original returns a bare array here, not the (mask, score)
        # tuple every other path returns -- crashes the caller's
        # `square_mask, _ = snap_square_by_rigid_transform(...)` unpacking
        # on a fully-blank generated square (a real occurrence early in
        # training). Matches the normal path's return arity; -inf score is
        # consistent with "no valid transform found."
        return square_mask.copy(), -np.inf

    # --- Distance transform from the curve (fixed) ---
    dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

    # --- Compute square center & characteristic length L from original mask ---
    ys, xs = np.where(square_mask > 0)
    center = np.mean(np.stack([xs, ys], axis=1), axis=0)  # (cx, cy)

    # Get the square's side estimate from min-area rect of the original mask
    contours, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt0 = max(contours, key=cv2.contourArea)
        rect0 = cv2.minAreaRect(cnt0)       # ((cx, cy), (w_side, h_side), angle)
        (w_side, h_side) = rect0[1]
        s = 0.5 * (w_side + h_side)         # average side length estimate
        # Half the diagonal:
        L = (s * np.sqrt(2)) / 2.0 if s > 0 else 1.0
    else:
        # Fallback: infer s from area (assuming roughly square)
        area = float(np.count_nonzero(square_mask))
        s = np.sqrt(area) if area > 0 else 1.0
        L = (s * np.sqrt(2)) / 2.0

    best_score = -np.inf
    best_mask = square_mask.copy()

    num_steps = int(np.floor((angle_range[1] - angle_range[0]) / angle_step)) + 1
    for angle_deg in np.linspace(angle_range[0], angle_range[1], num_steps):
        theta = np.deg2rad(angle_deg)  # radians for the rotation term
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)

        for dx in range(-trans_range, trans_range + 1, trans_step):
            for dy in range(-trans_range, trans_range + 1, trans_step):
                # Compose rotation + translation
                M = rot_mat.copy()
                M[0, 2] += dx
                M[1, 2] += dy

                transformed_mask = cv2.warpAffine(square_mask, M, (w, h), flags=cv2.INTER_NEAREST)

                contours_t, _ = cv2.findContours(transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours_t:
                    continue
                cnt = max(contours_t, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(cnt, True)  # tolerance factor (2% of perimeter)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:
                    box = approx.reshape(-1, 2)
                else:
                    # fallback: still use rectangle if not 4-sided
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)

                # Alignment score (your existing logic)
                align_score, _, _, _ = compute_alignment_score_from_box(box, transformed_mask, dist_transform)
                quality_score = squareness_metric(transformed_mask)
                # --- Single-parameter transform penalty (option #2) ---
                trans_norm_sq = float(dx*dx + dy*dy)          # ||t||^2 in pixels^2
                rot_term_sq = float((L * theta) * (L * theta))  # (L*theta)^2
                penalty = lambda_reg * (trans_norm_sq + rot_term_sq)

                total_score = align_score - penalty

                if total_score > best_score:
                    best_score = total_score
                    best_mask = transformed_mask.copy()

    return best_mask, best_score


def get_box(square_mask: np.ndarray) -> np.ndarray:
    contours_t, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_t:
        return None  # bugfix: original had a bare `None` statement here (no-op, not a return)
    cnt = max(contours_t, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)  # tolerance factor (2% of perimeter)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        box = approx.reshape(-1, 2)
    else:
        # fallback: still use rectangle if not 4-sided
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
    return box
