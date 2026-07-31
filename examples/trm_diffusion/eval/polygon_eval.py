"""
eval/polygon_eval.py – Evaluation for the Max-Area Polygon dataset
(datasets/polygon_dataset.py).

Simpler than Steiner Tree's eval (eval/steiner_eval.py): a polygon uses only
the *given* points as vertices — no new junction points are ever introduced
— so vertex positions are always exactly known (from embedding_conditions),
nothing to detect from pixels. Only edge *existence* between known point
pairs needs reading off the generated image; validity (simple polygon,
uses every point) and area are then computed from the exact known
coordinates, not re-derived lossily from pixels.

Mirrors eval/steiner_eval.py's overall shape: a constraint_puzzle_acc pass/
fail check independent of the reference, plus optimality_ratio looked up by
puzzle_id against the exact optimal value the offline solver already
computed (datasets/polygon_generation.py's max_area_polygon) — see
[[feedback-check-optimization-objective-not-just-validity]] in memory for
why both matter (validity alone is trivially satisfiable and says nothing
about whether the model found the *largest*-area polygon, not just *a*
simple one).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from datasets.polygon_generation import _area_signed, _segments_properly_intersect

# Discretization cutoff on the raw [-1, 1] rendered signal (matches
# datasets/polygon_dataset.py's render_polygon: background=0, edge=+1).
EDGE_THRESH = 0.5

# Max allowed distance (px) from sampled line points to the nearest edge-
# colored pixel for a candidate edge to count as "detected" — calibrated
# against ground-truth renders: true edges never exceed ~1.4px (rendering/
# sampling noise), false chords are essentially never below ~3px except for
# occasional near-collinear coincidences. See _detect_edges's docstring for
# why exact-pixel/coverage-ratio matching was tried first and didn't work.
# At this threshold, ground truth reconstructs exactly ~97.7% of the time
# (vs. Steiner Tree's ~78% — easier here since vertices are always exactly
# known, nothing to detect); area on successfully-reconstructed samples
# matches the true optimum to ~1e-4 relative error (computed from exact
# coordinates, not pixels, so the only error source is occasional phantom
# edges breaking reconstruction, not area-measurement noise itself).
EDGE_DIST_THRESH = 1.5


def _detect_edges(
    img: np.ndarray, points_rc: np.ndarray, n_samples: int = 50, margin_frac: float = 0.08
) -> list[tuple[int, int]]:
    """Check every pair of *known* points for a covered edge in the image,
    via distance-to-nearest-edge-pixel rather than exact coverage-ratio
    matching.

    Coverage-ratio matching (sample points along the p->q line, require a
    high fraction to land exactly on an edge-colored pixel) was tried first
    and didn't work: the rendered edge is only edge_width=1px wide, so any
    reasonable line-sampling algorithm (Bresenham or linspace+round) drifts
    out of exact pixel alignment with how cv2.polylines actually drew it,
    and there's no coverage threshold that cleanly separates true edges from
    chords that merely pass near the (possibly star-shaped) polygon
    boundary. Precomputing one distance-transform of the edge mask per image
    and requiring line points stay close to *some* edge pixel (not
    necessarily the exact sampled one) sidesteps the alignment-sensitivity
    entirely and calibrates cleanly (see EDGE_DIST_THRESH).

    Tolerates a handful of false-positive "phantom" edges (chords that
    happen to run near the real boundary) rather than trying to eliminate
    them outright — evaluate_polygon's Hamiltonian-cycle search (matching
    the paper's own extract_polygon_graph.py approach) tolerates these by
    searching for *a* valid tour within the detected edges rather than
    requiring the edge set to already be exactly one.
    """
    edge_mask = img > EDGE_THRESH
    dist_map = distance_transform_edt(~edge_mask)
    n = len(points_rc)
    m = int(n_samples * margin_frac)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            rows = np.linspace(points_rc[i][0], points_rc[j][0], n_samples)
            cols = np.linspace(points_rc[i][1], points_rc[j][1], n_samples)
            if n_samples > 2 * m:
                rows, cols = rows[m:-m], cols[m:-m]
            ri = np.clip(rows.round().astype(int), 0, dist_map.shape[0] - 1)
            ci = np.clip(cols.round().astype(int), 0, dist_map.shape[1] - 1)
            if len(ri) and np.percentile(dist_map[ri, ci], 90) <= EDGE_DIST_THRESH:
                edges.append((i, j))
    return edges


def _cycle_edges_self_intersect(points_xy: np.ndarray, order: list[int]) -> bool:
    """Exact check using the known point coordinates (not pixels) — same
    segment-intersection test the offline solver itself uses. Only checks
    the returned cycle's own consecutive edges, not every detected edge
    (some of those may be unused "phantom" edges the cycle search ignored)."""
    n = len(order)
    edges = [(order[i], order[(i + 1) % n]) for i in range(n)]
    for a in range(len(edges)):
        i, j = edges[a]
        for b in range(a + 1, len(edges)):
            k, l = edges[b]
            if len({i, j, k, l}) < 4:
                continue  # shares a vertex — not a proper crossing
            if _segments_properly_intersect(
                tuple(points_xy[i]), tuple(points_xy[j]), tuple(points_xy[k]), tuple(points_xy[l])
            ):
                return True
    return False


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


def _hamiltonian_cycle_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """Search for *some* Hamiltonian cycle within `edges` (does not require
    the edge set to be exactly one cycle — tolerates extra "phantom" edges
    not used by the returned tour, matching the paper's own
    extract_polygon_graph.py._find_simple_cycle approach). Plain backtracking
    DFS; fast enough at this dataset's scale (n <= 12).
    """
    if n < 3:
        return None
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    if not nx.is_connected(G):
        return None

    def dfs(path: list[int], visited: set) -> list[int] | None:
        if len(path) == n:
            return path if path[0] in G[path[-1]] else None
        for nxt in G[path[-1]]:
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                result = dfs(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None

    for start in G.nodes():
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


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
        pts_rc = pts_xy[:, ::-1] * (image_size - 1)  # (row, col) pixel coords
        edges = _detect_edges(imgs[b], pts_rc)
        order = _hamiltonian_cycle_order(n, edges)
        if order is None:
            continue
        if _cycle_edges_self_intersect(pts_xy, order):
            continue
        valid[b] = True
        area[b] = abs(_area_signed([tuple(p) for p in pts_xy], order))
        orders[b] = order

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
