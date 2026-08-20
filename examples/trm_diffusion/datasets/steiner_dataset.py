"""
SteinerTreeDataset – loads pre-solved Steiner tree instances (generated
offline by datasets/steiner_generation.py using the real GeoSteiner exact
solver) and renders condition/target images on the fly, for the "draw the
optimal tree connecting these points" logical-constraint benchmark.

Based on "Visual Diffusion Models are Geometric Solvers" (arXiv 2510.21697,
Goren et al.), whose dataset/rendering convention this reproduces: a single-
channel image with a 3-level scheme — background=0, vertices=-1, tree
edges=+1 — rather than this project's usual RGB (Sudoku/Maze) or multi-
channel spatial masks (CLEVR), since that's the architecture/representation
this dataset is meant to match. A Steiner tree may introduce extra junction
("Steiner") points not in the original terminal set to shorten the tree —
that's what makes this NP-hard, unlike a plain minimum spanning tree over the
terminals — so the target image draws both terminals and any such points.

Each sample:
  images             – (1, H, W) float32 in {-1, 0, +1}-ish (bilinear-drawn
                         circles/lines, not hard-thresholded); the optimal
                         tree drawn over the terminal points.
  spatial_conditions – same shape; terminal points only, no tree — the
                         puzzle condition. Not fed to the (unconditional)
                         stage-1 painter; this is the thinker's primary
                         conditioning input in stage 2, via a CNN
                         (SpatialConditionEncoder) exactly like Maze/Sudoku —
                         spatial conditioning rather than CLEVR's abstract
                         per-object token set, since terminal points live in
                         the same pixel coordinate space as the image being
                         generated, so a CNN encoder keeps that spatial
                         correspondence instead of discarding it.
  embedding_conditions – (MAX_POINTS, 2) float32; terminal point (x, y) in
                         [0, 1], zero-padded beyond the instance's actual
                         terminal count. NOT used as thinker conditioning —
                         kept only so eval/steiner_eval.py's terminal-coverage
                         check has exact ground-truth terminal coordinates
                         without re-deriving them from spatial_conditions.
  embedding_mask     – (MAX_POINTS,) bool; True = real terminal, False = pad.
  puzzle_id          – () int64.

No `solution`/`solution_mask`/`token_conditions` fields are populated: unlike
Sudoku/Maze's per-cell class labels, "which point pairs are connected" here
is a variable-size edge list over a variable-size point set (terminals *and*
possibly new Steiner points the model must itself decide to add), which
doesn't fit a fixed per-token discrete-class schema. As with Sudoku/Maze, the
thinker is trained purely through the diffusion MSE loss (no CE term); eval
correctness is checked directly on the rendered image (see eval/steiner_eval.py),
not via a per-token accuracy metric.
"""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples

MAX_POINTS = 20  # matches steiner_generation.py's default --max-points


def render_steiner(
    terminal_points: np.ndarray,   # (n_t, 2) float in [0, 1]
    steiner_points: np.ndarray,    # (n_s, 2) float in [0, 1], may be empty
    edges: list,                   # list of [i, j] indices into terminals++steiner
    image_size: int = 128,
    node_radius: int = 2,
    edge_width: int = 2,
    draw_tree: bool = True,
) -> np.ndarray:
    """Render a single-channel image with the paper's 3-level scheme.

    Background=127, tree edges=255 (drawn first), vertices=0 (drawn on top,
    so a vertex always occludes any edge pixels under it) — then rescaled to
    background=0, vertex=-1, edge=+1 as a (1, H, W) float32 array.
    `draw_tree=False` renders the condition image (terminal points only).
    """
    img = np.full((image_size, image_size), 127, dtype=np.uint8)
    all_points = (
        np.vstack([terminal_points, steiner_points]) if len(steiner_points) > 0 else terminal_points
    )
    px = (all_points * (image_size - 1)).astype(int)

    if draw_tree and edges:
        for i, j in edges:
            cv2.line(img, tuple(px[i]), tuple(px[j]), color=255, thickness=edge_width)

    n_t = len(terminal_points)
    for idx in range(n_t if not draw_tree else len(all_points)):
        cv2.circle(img, tuple(px[idx]), node_radius, color=0, thickness=-1)

    out = (img.astype(np.float32) - 127.0) / 127.0
    return out[None, :, :]


class SteinerTreeDataset(Dataset):
    """
    Args:
        ndjson_path: path to a file produced by datasets/steiner_generation.py.
            If None, downloads `hf_filename` from `hf_repo` instead — the
            default in configs/data/steiner.yaml, so training works on a
            fresh machine (e.g. a remote cluster) with no local data/
            directory and no GeoSteiner build; generation is a one-time
            offline step (see datasets/steiner_generation.py's docstring),
            not something training ever re-runs.
        hf_repo, hf_filename: HuggingFace dataset repo/file to download from
            when ndjson_path is None. Uses huggingface_hub directly (not the
            `datasets` library) — see SketchVLMMazeBenchmark in
            maze_dataset.py for why (`datasets` package name collision).
        image_size, node_radius, edge_width: rendering parameters, matching
            generate_steiner_data.py's/GeoSteinerDataset.py's defaults.
        max_points: pad embedding_conditions/embedding_mask to this many
            terminal-point slots (must be >= the generator's --max-points).
    """

    def __init__(
        self,
        ndjson_path: Optional[str] = None,
        hf_repo: str = "pskiers/trm-diffusion-steiner-tree",
        hf_filename: str = "train.ndjson",
        image_size: int = 128,
        node_radius: int = 2,
        edge_width: int = 2,
        max_points: int = MAX_POINTS,
    ):
        super().__init__()
        self.image_size = image_size
        self.node_radius = node_radius
        self.edge_width = edge_width
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
        self._length_by_id = {inst["instance_id"]: inst["total_length"] for inst in self.instances}

    def __len__(self) -> int:
        return len(self.instances)

    def optimal_length_for(self, puzzle_id: int) -> Optional[float]:
        """Exact optimal tree length GeoSteiner computed for this instance at
        generation time (see datasets/steiner_generation.py) — used by
        eval/steiner_eval.py to score optimality without re-solving or
        re-deriving it lossily from a rendered image."""
        return self._length_by_id.get(int(puzzle_id))

    def __getitem__(self, idx: int) -> DataSample:
        inst = self.instances[idx]
        terminal_points = np.array(inst["terminal_points"], dtype=np.float32)
        steiner_points = (
            np.array(inst["steiner_points"], dtype=np.float32)
            if inst["num_steiner_points"] > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        edges = inst["edges"]
        n_t = len(terminal_points)
        if n_t > self.max_points:
            raise ValueError(f"instance has {n_t} terminals > max_points={self.max_points}")

        cond_img = render_steiner(
            terminal_points, steiner_points, edges, self.image_size, self.node_radius, self.edge_width,
            draw_tree=False,
        )
        full_img = render_steiner(
            terminal_points, steiner_points, edges, self.image_size, self.node_radius, self.edge_width,
            draw_tree=True,
        )

        emb_cond = np.zeros((self.max_points, 2), dtype=np.float32)
        emb_cond[:n_t] = terminal_points
        emb_mask = np.zeros(self.max_points, dtype=bool)
        emb_mask[:n_t] = True

        return DataSample(
            images=torch.from_numpy(full_img),
            spatial_conditions=torch.from_numpy(cond_img),
            embedding_conditions=torch.from_numpy(emb_cond),
            embedding_mask=torch.from_numpy(emb_mask),
            puzzle_id=torch.tensor(inst["instance_id"], dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)
