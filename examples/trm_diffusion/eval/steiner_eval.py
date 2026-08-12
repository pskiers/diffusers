"""
eval/steiner_eval.py – Evaluation for the Steiner Tree dataset
(datasets/steiner_dataset.py).

Extracts a graph (vertices + edges) from a generated raster image and scores
it against the tree's hard logical constraints — connectivity, valid
tree-edge count, and terminal coverage — independent of whether it matches
the reference optimal tree (a model could legitimately connect the same
terminals through a different, still-valid Steiner topology).

2026-08-12: this module used to carry an independent reimplementation of the
extraction algorithm (different vertex-detection heuristic, different edge
line-coverage check), justified at the time by it scoring better on an
internal ground-truth calibration check. That's a real trade-off but not
what was asked for — the goal of this whole reproduction effort is to match
the paper's methodology exactly, even where a different choice measures
better in isolation. So this now calls extract_steiner_graph.SteinerGraphExtractor
directly — a verbatim vendored copy of the paper's own
scripts/extract_steiner_graph.py (see that module's header) — and
evaluate_steiner() below is a direct port of their scripts/evaluate_steiner.py's
_calculate_metrics/_calculate_metrics_worker and
_convert_tensor_to_discrete/_get_terminal_pixel_coords, not a reimplementation.

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

from eval.extract_steiner_graph import SteinerGraphExtractor

# Matches the paper's own real eval-time SteinerGraphExtractor construction
# (scripts/evaluate_steiner.py's _init_graph_extractor / graph_extractor_config)
# -- vertex_radius/edge_width from the dataset config (always 2/2 here),
# coverage_threshold/proximity_threshold hardcoded to 0.9/4.0 regardless of
# the class's own (0.7/2.0) or CLI (0.7/3.0) defaults.
_EXTRACTOR = SteinerGraphExtractor(
    vertex_radius=2, edge_width=2, coverage_threshold=0.9, proximity_threshold=4.0
)

# Wrapper-level safety net, NOT part of the ported algorithm: the real
# extract_graph's edge search is O(n^2) vertex pairs, each with an O(n)
# passes-through-vertex check -- effectively cubic. A real render never
# exceeds ~60 vertices (41-50 terminals + Steiner points) x ~13px/vertex
# disc (node_radius=2) =~ 780 vertex pixels; pure noise from an untrained
# model measured 4030 vertex pixels -> 2214 vertices -> the edge search
# alone didn't finish in 2 minutes. This is the exact same failure mode a
# prior version of this file's own (non-ported) vertex-NMS hit and fixed
# with a MAX_VERTICES cap -- keeping an equivalent cap here, but at the
# call site rather than inside the vendored file, so the ported algorithm
# itself stays byte-identical to the original for every input that isn't
# adversarial/garbage.
_MAX_VERTEX_PIXELS = 2000


def _convert_tensor_to_discrete(img: np.ndarray, image_size: int) -> np.ndarray:
    """Direct port of evaluate_steiner.py's _convert_tensor_to_discrete /
    score_steiner_sample_worker's inline equivalent: [-1,1] -> {0,127,255}."""
    vertex_mask = img <= -0.5
    edge_mask = img >= 0.5
    discrete = np.full(img.shape, 127, dtype=np.uint8)
    discrete[edge_mask] = 255
    discrete[vertex_mask] = 0
    if discrete.shape != (image_size, image_size):
        import cv2
        discrete = cv2.resize(discrete, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return discrete


def _get_terminal_pixel_coords(terminals_xy: np.ndarray, image_size: int) -> list[tuple[float, float]]:
    """Direct port of evaluate_steiner.py's _get_terminal_pixel_coords:
    normalized [0,1] (x, y) -> pixel (x, y), clipped to [0, image_size-1]."""
    denom = image_size - 1
    return [
        (float(np.clip(tx * denom, 0.0, float(denom))), float(np.clip(ty * denom, 0.0, float(denom))))
        for tx, ty in terminals_xy
    ]


def _calculate_metrics(
    vertices: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    terminals_xy: np.ndarray,  # (n_term, 2) normalized [0,1] (x, y)
    image_size: int,
    threshold: float = 0.03,
) -> tuple[float | None, bool, bool, bool]:
    """Direct port of evaluate_steiner.py's _calculate_metrics /
    _calculate_metrics_worker (the two are identical in substance)."""
    if not vertices:
        return None, False, False, False

    G = nx.Graph()
    for i, _ in enumerate(vertices):
        G.add_node(i)
    G.add_edges_from(edges)

    is_connected = nx.is_connected(G) if len(vertices) > 1 else True
    is_valid_tree = is_connected and len(edges) == len(vertices) - 1

    denom = image_size - 1
    normalized_vertices = [(x / denom, y / denom) for x, y in vertices]

    if not edges:
        predicted_weight = 0.0
    else:
        predicted_weight = 0.0
        for i, j in edges:
            x1, y1 = normalized_vertices[i]
            x2, y2 = normalized_vertices[j]
            predicted_weight += float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    covered_terminals = 0
    for tx, ty in terminals_xy:
        for vx, vy in normalized_vertices:
            if np.sqrt((tx - vx) ** 2 + (ty - vy) ** 2) <= threshold:
                covered_terminals += 1
                break
    covers_terminals = covered_terminals == len(terminals_xy)

    return predicted_weight, is_valid_tree, is_connected, covers_terminals


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

    connected = np.zeros(B, dtype=bool)
    valid_tree = np.zeros(B, dtype=bool)
    covers = np.zeros(B, dtype=bool)
    length = np.full(B, np.nan, dtype=np.float64)

    for b in range(B):
        term_xy = emb[b][mask[b]]  # (n_term, 2) normalized [0,1] (x, y)

        discrete_img = _convert_tensor_to_discrete(imgs[b], image_size)
        terminal_pixels = _get_terminal_pixel_coords(term_xy, image_size)
        n_vertex_px = int((discrete_img == 0).sum())
        if n_vertex_px > _MAX_VERTEX_PIXELS:
            # Garbage input (e.g. an untrained model) -- skip the O(n^2)-to-
            # O(n^3) edge search entirely; every downstream check already
            # fails on an empty edge list anyway, same as the real
            # algorithm would eventually conclude, just without the wait.
            vertices, edges = [], []
        else:
            vertices, edges = _EXTRACTOR.extract_graph(discrete_img, reference_points=terminal_pixels)

        predicted_weight, is_valid_tree, is_connected, covers_terminals = _calculate_metrics(
            vertices, edges, term_xy, image_size, threshold=terminal_tol_frac
        )
        connected[b] = is_connected
        valid_tree[b] = is_valid_tree
        covers[b] = covers_terminals
        if predicted_weight is not None and edges:
            length[b] = predicted_weight

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
