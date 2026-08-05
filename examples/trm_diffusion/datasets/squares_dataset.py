"""
datasets/squares_dataset.py – loads pre-generated Inscribed Square instances
(generated offline by datasets/squares_generation.py, itself a numpy port of
data/Curves.py's on-the-fly generator) and renders condition/target images on
the fly, for the "draw a square inscribed in this curve" task.

Based on "Visual Diffusion Models are Geometric Solvers" (arXiv 2510.21697,
Goren et al.). Unlike Steiner Tree/Polygon, condition and target are two
*independent* renders (no shared point set drawn into both): the condition
is just the closed curve outline, the target is just the (filled) square —
matching the original's `ellipse_imgs`/`square_imgs` split in
training/train_diffusion.py, not a "draw the solution over the puzzle"
nested scheme.

Each sample:
  images             – (1, H, W) float32; the target square, outline +
                         filled interior (background=+1, square=-1) — see
                         render_square. Matches config_curves.yaml's
                         fill_square=true.
  spatial_conditions – (1, H, W) float32; the curve outline only
                         (background=+1, curve=-1) — see render_curve.
  embedding_conditions – (4, 2) float32; the exact square corner
                         coordinates in this dataset's native [-1, 1] box
                         (same space used for rendering) — not used as model
                         conditioning (the model only ever sees the two
                         rendered images), kept for debugging/visualization
                         parity with SteinerTreeDataset's embedding_conditions.
  puzzle_id          – () int64.

No solution/solution_mask/token_conditions — same reasoning as Steiner/
Polygon (nothing here is a fixed per-token discrete-class target). Unlike
Steiner, no ratio/optimality ground truth is needed for evaluation either:
eval/squares_eval.py's squareness_metric/alignment_metric (ported from the
original's utils/metrics.py, and literally what training/train_diffusion.py
uses for its own validation loop) score the generated square against its
*own* shape and against the rasterized curve mask directly — no exact
target corners required, hence no `optimal_*_for(puzzle_id)` lookup method
on this class (contrast SteinerTreeDataset.optimal_length_for /
PolygonDataset.optimal_area_for).
"""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples


def render_curve(
    curve_points: np.ndarray,  # (N, 2) float in [-1, 1]
    image_size: int = 128,
    thickness: int = 1,
) -> np.ndarray:
    """Ported from data/Curves.py's draw_periodic_spline_image.

    Deliberately uses a *different* [-1,1]->pixel mapping than render_square
    below (`image_size // 2` scale+offset, integer-truncated) — the original
    itself uses two different pixel-coordinate helpers for the curve vs. the
    square (see draw_periodic_spline_image vs. to_pixel_coords/draw_square in
    utils/viz.py); reproduced here rather than unified, since the point is
    to match the original's actual rendering, not to fix its inconsistency.
    """
    img = np.full((image_size, image_size), 255, dtype=np.uint8)
    pts = (curve_points * (image_size // 2) + image_size // 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=0, thickness=thickness)

    out = (img.astype(np.float32) / 255.0) * 2 - 1
    return out[None, :, :]


def render_square(
    square_corners: np.ndarray,  # (4, 2) float in [-1, 1]
    image_size: int = 128,
    thickness: int = 1,
    fill: bool = True,
) -> np.ndarray:
    """Ported from data/Curves.py's draw_square + _fill_square_interior.

    Outline is rasterized first (round-to-nearest pixel mapping over
    [0, image_size-1], matching utils/viz.py's to_pixel_coords exactly —
    note this differs from render_curve's mapping, see its docstring), then
    the interior is flood-filled via binary_fill_holes on the normalized
    tensor, matching CurveImageDataset._fill_square_interior's operating
    order (fill happens post-normalization, not on the raw uint8 raster).
    """
    img = np.full((image_size, image_size), 255, dtype=np.uint8)
    px = ((square_corners + 1) / 2 * (image_size - 1)).round().astype(np.int32)
    cv2.polylines(img, [px], isClosed=True, color=0, thickness=thickness)

    out = (img.astype(np.float32) / 255.0) * 2 - 1

    if fill:
        perimeter_mask = out == -1
        filled_mask = binary_fill_holes(perimeter_mask)
        out = np.where(filled_mask, -1.0, 1.0).astype(np.float32)

    return out[None, :, :]


class SquaresDataset(Dataset):
    """
    Args:
        ndjson_path: path to a file produced by datasets/squares_generation.py.
            If None, downloads `hf_filename` from `hf_repo` instead — see
            SteinerTreeDataset for the same convention/rationale.
        hf_repo, hf_filename: HuggingFace dataset repo/file to download from
            when ndjson_path is None.
        image_size: rendering resolution — curve_points/square_corners are
            stored already fit into a padded [-1,1] box by
            squares_generation.py, independent of image_size, so this can
            differ from the resolution they were generated at.
        curve_thickness, square_thickness: line thickness in pixels, matching
            config_curves.yaml's curve_kwargs.thickness_range=[1] /
            square_kwargs.thickness=1 (both effectively constant 1; the
            original's thickness_range randomization is a no-op at that
            config value — see squares_generation.py's docstring on why it's
            not reproduced as actual per-sample randomness here).
        fill_square: whether to flood-fill the target square's interior,
            matching config_curves.yaml's fill_square=true.
    """

    def __init__(
        self,
        ndjson_path: Optional[str] = None,
        hf_repo: str = "pskiers/trm-diffusion-inscribed-square",
        hf_filename: str = "train.ndjson",
        image_size: int = 128,
        curve_thickness: int = 1,
        square_thickness: int = 1,
        fill_square: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.curve_thickness = curve_thickness
        self.square_thickness = square_thickness
        self.fill_square = fill_square

        if ndjson_path is None:
            from huggingface_hub import hf_hub_download

            ndjson_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_filename)

        self.instances: list[dict] = []
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> DataSample:
        inst = self.instances[idx]
        curve_points = np.array(inst["curve_points"], dtype=np.float32)
        square_corners = np.array(inst["square_corners"], dtype=np.float32)

        cond_img = render_curve(curve_points, self.image_size, self.curve_thickness)
        target_img = render_square(square_corners, self.image_size, self.square_thickness, self.fill_square)

        return DataSample(
            images=torch.from_numpy(target_img),
            spatial_conditions=torch.from_numpy(cond_img),
            embedding_conditions=torch.from_numpy(square_corners),
            puzzle_id=torch.tensor(inst["instance_id"], dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)
