"""
PolygonDataset – loads pre-solved Max-Area Simple Polygon instances
(generated offline by datasets/polygon_generation.py using the paper's own
exact backtracking solver) and renders condition/target images on the fly,
for the "draw the largest-area simple polygon through these points" logical-
constraint benchmark.

Based on "Visual Diffusion Models are Geometric Solvers" (arXiv 2510.21697,
Goren et al.). Unlike Steiner Tree, a polygon uses *only* the given points as
vertices — no new junction points are ever introduced — so finding the
optimum is exact search over point *orderings*, not tree topology; this also
makes evaluation simpler than Steiner's (see eval/polygon_eval.py): vertex
positions are always exactly known, nothing to detect from pixels.

Rendering reproduces the paper's own 3-level scheme (background=0,
vertex/polygon-interior=-1, edge=+1), matching Steiner Tree's convention —
paper config confirms `fill_polygon: true`, i.e. the polygon interior is
filled with the same value as vertex circles, not left as background.

Each sample:
  images             – (1, H, W) float32 in {-1, 0, +1}-ish; the optimal
                         polygon (filled interior + edge outline) through
                         the points.
  spatial_conditions – same shape; points only, no polygon — the puzzle
                         condition. Not fed to the (unconditional) stage-1
                         painter; this is the thinker's primary conditioning
                         input in stage 2 (spatial CNN), matching Steiner.
  embedding_conditions – (MAX_POINTS, 2) float32; point (x, y) in [0, 1],
                         zero-padded beyond the instance's actual point
                         count. NOT used as thinker conditioning — kept only
                         so eval/polygon_eval.py has exact ground-truth point
                         coordinates without re-deriving them from pixels.
  embedding_mask     – (MAX_POINTS,) bool; True = real point, False = pad.
  puzzle_id          – () int64.

No `solution`/`solution_mask`/`token_conditions` fields: "which point order
forms the polygon" is a variable-length permutation, which doesn't fit a
fixed per-token discrete-class schema any better than Steiner Tree's edge
list did. Trained purely through the diffusion MSE loss; eval correctness
is checked directly on the rendered image plus the known point coordinates
(see eval/polygon_eval.py), not via a per-token accuracy metric.
"""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples

MAX_POINTS = 12  # matches polygon_generation.py's default --max-points


def render_polygon(
    points: np.ndarray,   # (n, 2) float in [0, 1]
    order: list,          # permutation of range(n) — polygon vertex order; ignored if draw_polygon=False
    image_size: int = 128,
    node_radius: int = 2,
    edge_width: int = 1,
    fill_polygon: bool = True,
    draw_polygon: bool = True,
) -> np.ndarray:
    """Render a single-channel image with the paper's 3-level scheme.

    Background=127, polygon interior=0 (filled, if fill_polygon), edges=255
    (drawn on top of the fill), vertices=0 (drawn on top of everything) —
    then rescaled to background=0, vertex/interior=-1, edge=+1 as a
    (1, H, W) float32 array. `draw_polygon=False` renders the condition
    image (points only, no polygon) — matches PolygonDataset.py's
    _draw_nodes/_draw_polygon split.
    """
    img = np.full((image_size, image_size), 127, dtype=np.uint8)
    px = (points * (image_size - 1)).astype(int)

    if draw_polygon and order and len(order) >= 3:
        poly_px = px[order].astype(np.int32)
        if fill_polygon:
            cv2.fillPoly(img, [poly_px], color=0)
        cv2.polylines(img, [poly_px], isClosed=True, color=255, thickness=edge_width)

    for c in px:
        cv2.circle(img, tuple(c), node_radius, color=0, thickness=-1)

    out = (img.astype(np.float32) - 127.0) / 127.0
    return out[None, :, :]


class PolygonDataset(Dataset):
    """
    Args:
        ndjson_path: path to a file produced by datasets/polygon_generation.py.
            If None, downloads `hf_filename` from `hf_repo` instead — the
            default in configs/data/polygon.yaml, so training works on a
            fresh machine with no local data/ directory (the exact solver
            needs no build/vendoring here, but generation is still an
            offline one-time step — see polygon_generation.py's docstring).
        hf_repo, hf_filename: HuggingFace dataset repo/file to download from
            when ndjson_path is None. Uses huggingface_hub directly (not the
            `datasets` library) — see SketchVLMMazeBenchmark in
            maze_dataset.py for why (`datasets` package name collision).
        image_size, node_radius, edge_width, fill_polygon: rendering
            parameters, matching PolygonDataset.py's/config_polygonalization_max.yaml's defaults.
        max_points: pad embedding_conditions/embedding_mask to this many
            point slots (must be >= the generator's --max-points).
    """

    def __init__(
        self,
        ndjson_path: Optional[str] = None,
        hf_repo: str = "pskiers/trm-diffusion-max-area-polygon",
        hf_filename: str = "train.ndjson",
        image_size: int = 128,
        node_radius: int = 2,
        edge_width: int = 1,
        fill_polygon: bool = True,
        max_points: int = MAX_POINTS,
    ):
        super().__init__()
        self.image_size = image_size
        self.node_radius = node_radius
        self.edge_width = edge_width
        self.fill_polygon = fill_polygon
        self.max_points = max_points

        if ndjson_path is None:
            from huggingface_hub import hf_hub_download

            ndjson_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_filename)

        self.instances: list[dict] = []
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))
        self._area_by_id = {inst["instance_id"]: inst["polygon_area"] for inst in self.instances}
        self._order_by_id = {inst["instance_id"]: inst["polygon_order"] for inst in self.instances}

    def __len__(self) -> int:
        return len(self.instances)

    def optimal_area_for(self, puzzle_id: int) -> Optional[float]:
        """Exact optimal polygon area the solver computed for this instance
        at generation time (see datasets/polygon_generation.py) — used by
        eval/polygon_eval.py to score optimality without re-solving or
        re-deriving it lossily from a rendered image."""
        return self._area_by_id.get(int(puzzle_id))

    def optimal_order_for(self, puzzle_id: int) -> Optional[list]:
        """Exact optimal vertex order (point indices) the solver found for
        this instance at generation time — used by PolygonEvalCallback to
        check whether the recovered polygon matches the true optimum exactly
        (not just by area), via eval.polygon_eval._orders_equivalent
        (cyclic/reflection-invariant comparison, since a polygon's vertex
        list has no canonical start or direction)."""
        return self._order_by_id.get(int(puzzle_id))

    def __getitem__(self, idx: int) -> DataSample:
        inst = self.instances[idx]
        points = np.array(inst["points"], dtype=np.float32)
        order = inst["polygon_order"]
        n = len(points)
        if n > self.max_points:
            raise ValueError(f"instance has {n} points > max_points={self.max_points}")

        cond_img = render_polygon(
            points, order, self.image_size, self.node_radius, self.edge_width, self.fill_polygon,
            draw_polygon=False,
        )
        full_img = render_polygon(
            points, order, self.image_size, self.node_radius, self.edge_width, self.fill_polygon,
            draw_polygon=True,
        )

        emb_cond = np.zeros((self.max_points, 2), dtype=np.float32)
        emb_cond[:n] = points
        emb_mask = np.zeros(self.max_points, dtype=bool)
        emb_mask[:n] = True

        return DataSample(
            images=torch.from_numpy(full_img),
            spatial_conditions=torch.from_numpy(cond_img),
            embedding_conditions=torch.from_numpy(emb_cond),
            embedding_mask=torch.from_numpy(emb_mask),
            puzzle_id=torch.tensor(inst["instance_id"], dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)
