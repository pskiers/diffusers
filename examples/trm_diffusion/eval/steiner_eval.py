"""
eval/steiner_eval.py – Evaluation for the Steiner Tree dataset
(datasets/steiner_dataset.py).

Extracts a graph (vertices + edges) from a generated raster image and scores
it against the tree's hard logical constraints — connectivity, valid
tree-edge count, and terminal coverage — independent of whether it matches
the reference optimal tree (a model could legitimately connect the same
terminals through a different, still-valid Steiner topology).

This is a simplified version of the paper's own eval/extraction pipeline
(scripts/evaluate_steiner.py + scripts/extract_steiner_graph.py in
https://github.com/kariander1/visual-geo-solver) — vertices are detected via
non-max-suppression on the vertex mask's distance transform (see
_detect_vertices) rather than their per-blob area-estimate + k-means
splitting, and edges are accepted via straight-line pixel-coverage sampling
with a proximity-to-other-vertex rejection. Ported the original's exact
algorithm and measured it against our own ground-truth renders to check —
it does *worse* on this data (60% exact vertex-count match vs. this file's
78%), so the simpler approach was kept. Both fail the same way (under-
counting from vertices that render within a couple pixels of each other,
never over-counting) — a genuine resolution limit of the rendering
(128px, node_radius=2), not a fixable bug in either implementation; see
_detect_vertices's docstring.

IMPORTANT — terminal snapping (2026-08-03 fix): the paper's extraction
*does* snap detected vertices onto the exact ground-truth terminal
coordinates when close enough (extract_steiner_graph.py's
_snap_vertices_to_reference), which we had NOT ported. Without it, every
detected vertex — including ones sitting exactly on a terminal — carries a
few tenths of a pixel of detection noise, and per_sample_length is measured
between these noisy positions. Since a valid tree's *true* length can never
be shorter than GeoSteiner's exact optimum, any measured length < optimum is
necessarily a measurement artifact, not a real solution — and empirically,
36.5% of our *own ground-truth* renders (no model involved) round-tripped to
a ratio < 1 before this fix (mean 1.0015, min 0.994), confirming it was a
real bug, not sampling noise. _snap_terminals_to_reference below ports their
fix: any detected vertex within snap_threshold of a known terminal is
replaced by that terminal's *exact* coordinate before length is computed,
eliminating quantization error on terminal-adjacent edges (Steiner junction
points, which have no reference to snap to, keep their raw detected
position — some residual noise there is unavoidable and matches the
paper's own approach).

optimality_ratio (generated tree length ÷ GeoSteiner's true optimal length)
is NOT computed in this module — it needs the exact optimal length, which
isn't part of the DataSample batch. Instead, SteinerEvalCallback (in
models/eval_callbacks.py) looks it up directly from the dataset's own
generation-time record via SteinerTreeDataset.optimal_length_for(puzzle_id),
and divides by the length this module's evaluate_steiner() already computes
from the generated image (`per_sample_length`) — no re-solving, no new
DataSample field, no lossy re-derivation from a rendered image.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from scipy import ndimage

# Discretization cutoffs on the raw [-1, 1] rendered signal (matches the
# paper's own eval-time thresholds): vertex <= -0.5, edge >= 0.5, else background.
VERTEX_THRESH = -0.5
EDGE_THRESH = 0.5

# Safety cap on detected vertices before attempting edge search — see its
# use in _extract_graph. Comfortably above the largest legitimate case
# (41-50 terminals + Steiner points, well under 150) while far below what
# noise can produce (~1500 spurious peaks observed on untrained output).
MAX_VERTICES = 150


def _detect_vertices(vertex_mask: np.ndarray, node_radius: float = 2.0, suppress_mult: float = 1.2) -> np.ndarray:
    """Greedy peak-picking on the vertex mask's distance transform.

    Steiner (junction) points aren't subject to the generator's terminal
    min-distance constraint, so two of them (or a Steiner point and a
    terminal) can legitimately render as one touching/overlapping blob —
    plain connected-components-centroid (one vertex per blob) then
    systematically undercounts those. Instead: repeatedly take the point
    farthest from the mask's background (the distance transform's global
    max) as a vertex, then suppress a disk of radius ~node_radius around it
    and repeat — standard non-max-suppression.

    Calibrated against ground-truth renders (datasets/steiner_data/val.ndjson):
    `suppress_mult=1.2` gives an exact vertex-count match on ~78% of them (up
    from ~53% with plain connected-components); the remainder are genuine
    near-coincidences in the solver's own geometry (e.g. a terminal and its
    optimal Steiner point landing ~2px apart at this resolution/node_radius) —
    a resolution limit, not a detection bug. constraint_puzzle_acc should be
    read as a slight underestimate of true tree validity for exactly this
    reason, same as the original paper's own pixel-based evaluation.
    """
    dt = ndimage.distance_transform_edt(vertex_mask).astype(np.float64)
    vertices = []
    H, W = dt.shape
    rr, cc = np.ogrid[:H, :W]
    suppress_r2 = (node_radius * suppress_mult) ** 2
    while True:
        peak = dt.max()
        if peak < 1.0:
            break
        r0, c0 = np.unravel_index(np.argmax(dt), dt.shape)
        vertices.append((float(r0), float(c0)))
        disk = (rr - r0) ** 2 + (cc - c0) ** 2 <= suppress_r2
        dt[disk] = 0.0
    return np.array(vertices, dtype=np.float64) if vertices else np.zeros((0, 2))


def _snap_terminals_to_reference(
    vertices_rc: np.ndarray, terminal_rc: np.ndarray, snap_threshold: float
) -> np.ndarray:
    """Greedily snap each detected vertex onto its nearest not-yet-used
    terminal reference point, if within `snap_threshold` — ports the paper's
    extract_steiner_graph.py._snap_vertices_to_reference. Each terminal
    claims at most one vertex (closest first, by terminal), and a claimed
    vertex's position is replaced by the terminal's *exact* coordinate —
    the whole point being to eliminate detection-quantization error on
    terminal-adjacent edges (see this module's docstring)."""
    if len(vertices_rc) == 0 or len(terminal_rc) == 0:
        return vertices_rc
    snapped = vertices_rc.copy()
    available = set(range(len(vertices_rc)))
    for ref in terminal_rc:
        best_idx, best_dist = None, float("inf")
        for i in available:
            d = float(np.linalg.norm(vertices_rc[i] - ref))
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is not None and best_dist <= snap_threshold:
            snapped[best_idx] = ref
            available.discard(best_idx)
    return snapped


def _extract_graph(
    img: np.ndarray,           # (H, W) float, background~0 vertex~-1 edge~+1
    node_radius: float = 2.0,
    coverage_threshold: float = 0.9,  # matches the paper's actual eval-time config (not its CLI default of 0.7)
    proximity_threshold: float = 4.0,
    terminal_rc: np.ndarray | None = None,  # (n_term, 2) exact terminal (row, col) pixel coords, for snapping
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Detect vertices (via non-max-suppression on the vertex mask's distance
    transform — see _detect_vertices) and edges (straight-line pixel coverage
    between vertex pairs) in a generated Steiner-tree raster.

    When `terminal_rc` is given, detected vertices near a known terminal are
    snapped onto its exact coordinate (see _snap_terminals_to_reference)
    before edges/length are computed from them — matches the paper's own
    extraction pipeline and is required for per_sample_length to never dip
    below the true optimum on terminal-adjacent edges.

    Returns (vertices: (n, 2) float array of (row, col) pixel coords, edges:
    list of (i, j) index pairs).
    """
    vertex_mask = img <= VERTEX_THRESH
    edge_mask = img >= EDGE_THRESH

    vertices = _detect_vertices(vertex_mask, node_radius)
    if len(vertices) == 0:
        return np.zeros((0, 2)), []
    if len(vertices) > MAX_VERTICES:
        # Noise (e.g. an early/untrained model) can produce hundreds of
        # spurious vertex-mask peaks — the O(n^2) edge search below (with an
        # O(n) passes_through_other_vertex check per candidate edge) is
        # otherwise unbounded and has been observed to take 60-140s for a
        # single sample at ~1500 vertices. Bail out immediately: no real
        # tree render ever needs anywhere near MAX_VERTICES, so exceeding it
        # is already diagnostic of garbage input — every downstream check
        # (connected/valid_tree/covers_terminals) correctly fails on an
        # empty edge list anyway.
        return vertices, []
    if terminal_rc is not None and len(terminal_rc) > 0:
        snap_threshold = max(2.0, node_radius * 2.5)
        vertices = _snap_terminals_to_reference(vertices, terminal_rc, snap_threshold)

    def line_coverage(p, q, n_samples=30):
        rows = np.linspace(p[0], q[0], n_samples)
        cols = np.linspace(p[1], q[1], n_samples)
        ri = np.clip(rows.round().astype(int), 0, edge_mask.shape[0] - 1)
        ci = np.clip(cols.round().astype(int), 0, edge_mask.shape[1] - 1)
        return float((edge_mask[ri, ci] | vertex_mask[ri, ci]).mean())

    def passes_through_other_vertex(p, q, skip_i, skip_j):
        seg = q - p
        seg_len2 = float(seg @ seg)
        if seg_len2 < 1e-9:
            return False
        for k, v in enumerate(vertices):
            if k in (skip_i, skip_j):
                continue
            t = float((v - p) @ seg) / seg_len2
            if 0.05 < t < 0.95:
                proj = p + t * seg
                if np.linalg.norm(v - proj) < proximity_threshold:
                    return True
        return False

    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if line_coverage(vertices[i], vertices[j]) >= coverage_threshold:
                if not passes_through_other_vertex(vertices[i], vertices[j], i, j):
                    edges.append((i, j))
    return vertices, edges


def _covers_terminals(vertices_rc: np.ndarray, terminal_px_rc: np.ndarray, tol_px: float) -> bool:
    if len(terminal_px_rc) == 0:
        return True
    if len(vertices_rc) == 0:
        return False
    d = np.linalg.norm(vertices_rc[None, :, :] - terminal_px_rc[:, None, :], axis=-1)  # (n_term, n_vert)
    return bool((d.min(axis=1) <= tol_px).all())


def _tree_length_normalized(vertices_rc: np.ndarray, edges: list[tuple[int, int]], image_size: int) -> float:
    """Sum of Euclidean edge lengths, in the same normalized [0,1]^2
    coordinate space datasets/steiner_generation.py's `total_length` is
    computed in (pixel distances / (image_size-1)) — so it's directly
    comparable to the exact optimal length looked up via
    SteinerTreeDataset.optimal_length_for, no unit conversion needed.
    """
    total = 0.0
    for i, j in edges:
        total += float(np.linalg.norm(vertices_rc[i] - vertices_rc[j])) / (image_size - 1)
    return total


@torch.no_grad()
def evaluate_steiner(
    images: torch.Tensor,             # (B, 1, H, W) float, generated
    embedding_conditions: torch.Tensor,  # (B, MAX_POINTS, 2) float [0,1] terminal coords
    embedding_mask: torch.Tensor,     # (B, MAX_POINTS) bool
    image_size: int,
    terminal_tol_frac: float = 0.03,  # matches the paper's ~0.03 normalized tolerance
) -> dict:
    """Extract a graph from each generated image and score it.

    Returns dict with keys:
      is_connected_acc     — fraction of samples whose extracted graph is connected.
      is_valid_tree_acc    — fraction that are additionally a valid tree (|E|=|V|-1).
      covers_terminals_acc — fraction where every true terminal has a nearby vertex.
      constraint_puzzle_acc — fraction passing ALL THREE checks at once (the
                              main "hard logical constraint" pass/fail metric,
                              analogous to Maze/Sudoku's constraint_puzzle_acc).
      per_sample_valid     — (B,) bool numpy array of the combined pass/fail.
      per_sample_length    — (B,) float numpy array; the extracted tree's
                              total edge length in normalized [0,1]^2 space
                              (NaN where extraction produced no edges) — the
                              caller combines this with the instance's exact
                              optimal length (SteinerTreeDataset.optimal_length_for,
                              looked up by puzzle_id) to get optimality_ratio;
                              not computed here since this function has no
                              access to the dataset/puzzle_id.
    """
    B = images.shape[0]
    imgs = images.squeeze(1).cpu().numpy()
    emb = embedding_conditions.cpu().numpy()
    mask = embedding_mask.cpu().numpy()
    tol_px = terminal_tol_frac * (image_size - 1)

    connected = np.zeros(B, dtype=bool)
    valid_tree = np.zeros(B, dtype=bool)
    covers = np.zeros(B, dtype=bool)
    length = np.full(B, np.nan, dtype=np.float64)

    for b in range(B):
        term_xy = emb[b][mask[b]]  # (n_term, 2) in [0,1]
        term_rc = term_xy[:, ::-1] * (image_size - 1)  # (row, col) pixel coords

        vertices, edges = _extract_graph(imgs[b], terminal_rc=term_rc)
        n = len(vertices)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        connected[b] = n > 0 and nx.is_connected(G)
        valid_tree[b] = connected[b] and G.number_of_edges() == n - 1
        if edges:
            length[b] = _tree_length_normalized(vertices, edges, image_size)

        covers[b] = _covers_terminals(vertices, term_rc, tol_px)

    valid = connected & valid_tree & covers
    return {
        "is_connected_acc": float(connected.mean()),
        "is_valid_tree_acc": float(valid_tree.mean()),
        "covers_terminals_acc": float(covers.mean()),
        "per_sample_length": length,
        "constraint_puzzle_acc": float(valid.mean()),
        "per_sample_valid": valid,
    }


def make_steiner_panel_image(
    condition: torch.Tensor,   # (1, H, W) float in [-1,1] — puzzle (points only)
    generated: torch.Tensor,   # (1, H, W) float in [-1,1] — model output
    reference: torch.Tensor,   # (1, H, W) float in [-1,1] — ground-truth solved tree
) -> np.ndarray:
    """condition | generated | reference, each mapped from [-1,1] back to [0,255] grayscale."""

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        arr = (t.squeeze(0).clamp(-1, 1).cpu().numpy() * 127.0 + 127.0).astype(np.uint8)
        return np.stack([arr] * 3, axis=-1)

    sep = np.full((condition.shape[-2], 4, 3), 200, dtype=np.uint8)
    return np.concatenate([to_uint8(condition), sep, to_uint8(generated), sep, to_uint8(reference)], axis=1)
