"""
eval/polygon_eval.py – Evaluation for the Max-Area Polygon dataset
(datasets/polygon_dataset.py).

2026-08-12: this now calls extract_polygon_graph.PolygonGraphExtractor
directly — a verbatim vendored copy of the paper's own
scripts/extract_polygon_graph.py (see that module's header) — rather than
an independent edge-detection/self-intersection reimplementation. Same
reasoning as eval/steiner_eval.py's equivalent change: the goal of this
reproduction is to match the paper's methodology exactly, not to
independently re-derive something that scores similarly. evaluate_polygon()
below is a direct port of scripts/evaluate_polygonization.py's
_evaluate_instance's extraction-and-scoring core (tensor->binary conversion,
extract_polygon_from_points call, is_valid_polygon check), not a
reimplementation. Two real algorithmic differences this caught vs. the
prior custom implementation: (1) their real edge detection is an exact
Bresenham line-coverage-ratio check (skip=edge_width, 3x3 patch tolerance,
threshold=0.9), not a distance-transform/percentile heuristic; (2) their
real self-intersection check runs over *every detected edge* before even
attempting to find a cycle (so a stray phantom chord that crosses another
detected edge invalidates the whole sample), not just the edges of the
Hamiltonian cycle that was eventually found — the prior implementation's
docstring explicitly (and, it turns out, incorrectly) assumed the paper's
own code tolerated non-cycle phantom edges the way ours did.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from eval.extract_polygon_graph import PolygonGraphExtractor

_EXTRACTOR = PolygonGraphExtractor()  # matches the paper's real construction (all defaults)


def _orders_equivalent(a: list[int], b: list[int]) -> bool:
    """True if cyclic vertex orders `a` and `b` describe the same polygon.

    A polygon's vertex list has no canonical start index or traversal
    direction, so two orders describe the same polygon iff one is a
    rotation of the other, or a rotation of its reversal."""
    if len(a) != len(b):
        return False
    n = len(a)
    doubled = b + b
    doubled_rev = b[::-1] + b[::-1]
    return any(doubled[s : s + n] == a or doubled_rev[s : s + n] == a for s in range(n))


@torch.no_grad()
def evaluate_polygon(
    images: torch.Tensor,             # (B, 1, H, W) float, generated
    embedding_conditions: torch.Tensor,  # (B, MAX_POINTS, 2) float [0,1] point coords
    embedding_mask: torch.Tensor,     # (B, MAX_POINTS) bool
    image_size: int,
) -> dict:
    """Extract the edge set between known points from each generated image and score it.

    Returns dict with keys:
      constraint_puzzle_acc — fraction of samples whose predicted edges form
                              a valid simple polygon using every given point
                              (Hamiltonian cycle + no self-intersections),
                              independent of whether it matches the
                              reference optimal polygon.
      per_sample_valid      — (B,) bool numpy array of that pass/fail.
      per_sample_area       — (B,) float numpy array; the valid polygon's
                              exact area (NaN where invalid) — the caller
                              combines this with the instance's exact
                              optimal area (PolygonDataset.optimal_area_for,
                              looked up by puzzle_id) to get optimality_ratio.
      per_sample_order      — length-B list; the recovered vertex order
                              (point indices into embedding_conditions'
                              non-pad slots) where valid, else None — the
                              caller compares this against
                              PolygonDataset.optimal_order_for (via
                              _orders_equivalent) to score exact-match rate.
    """
    B = images.shape[0]
    imgs = images.squeeze(1).cpu().numpy()
    emb = embedding_conditions.cpu().numpy()
    mask = embedding_mask.cpu().numpy()

    valid = np.zeros(B, dtype=bool)
    area = np.full(B, np.nan, dtype=np.float64)
    orders: list = [None] * B

    for b in range(B):
        pts_xy = emb[b][mask[b]]  # (n, 2) in [0,1]
        n = len(pts_xy)
        if n < 3:
            continue
        gt_points = [tuple(p) for p in pts_xy]

        # Direct port of evaluate_polygonization.py's tensor->binary
        # conversion: [-1,1] -> [-127,127] float (kept continuous, not
        # discretized to exact {0,127,-127} like Steiner's format).
        pred_binary = (imgs[b] * 127.0).astype(np.float32)
        if pred_binary.shape != (image_size, image_size):
            pred_binary = cv2.resize(pred_binary, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        vertices, pred_area, _pred_perimeter, _edges, vertex_indices = _EXTRACTOR.extract_polygon_from_points(
            pred_binary, gt_points
        )

        if len(vertices) >= 3:  # direct port of evaluate_polygonization.py's _is_valid_polygon
            valid[b] = True
            area[b] = pred_area
            orders[b] = vertex_indices

    return {
        "constraint_puzzle_acc": float(valid.mean()),
        "per_sample_valid": valid,
        "per_sample_area": area,
        "per_sample_order": orders,
    }


def make_polygon_panel_image(
    condition: torch.Tensor,   # (1, H, W) float in [-1,1] — puzzle (points only)
    generated: torch.Tensor,   # (1, H, W) float in [-1,1] — model output
    reference: torch.Tensor,   # (1, H, W) float in [-1,1] — ground-truth solved polygon
) -> np.ndarray:
    """condition | generated | reference, each mapped from [-1,1] back to [0,255] grayscale."""

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        arr = (t.squeeze(0).clamp(-1, 1).cpu().numpy() * 127.0 + 127.0).astype(np.uint8)
        return np.stack([arr] * 3, axis=-1)

    sep = np.full((condition.shape[-2], 4, 3), 200, dtype=np.uint8)
    return np.concatenate([to_uint8(condition), sep, to_uint8(generated), sep, to_uint8(reference)], axis=1)
