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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
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
        out = fill_interior(out)

    return out[None, :, :]


def fill_interior(out: np.ndarray) -> np.ndarray:
    """Flood-fills the interior of a {-1,+1}-valued outline (perimeter=-1,
    background=+1), matching CurveImageDataset._fill_square_interior's
    operating order (post-normalization, not on the raw uint8 raster).
    Shared by render_square and SquaresImagePairDataset (real, pre-rendered
    unfilled outlines need the exact same fill step)."""
    perimeter_mask = out == -1
    filled_mask = binary_fill_holes(perimeter_mask)
    return np.where(filled_mask, -1.0, 1.0).astype(np.float32)


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


class SquaresImagePairDataset(Dataset):
    """Loads the paper's own real (curve, square) image pairs directly —
    github.com/kariander1/visual-geo-solver's actual released training data
    (curves_data.zip on the nirgoren/geometric-solver HF dataset repo),
    repackaged verbatim under hf_repo below. 100,000 pairs, exactly matching
    config_curves.yaml's num_samples — confirmed contiguous instance ids
    0..99999, no gaps. Unlike SquaresDataset (which renders on the fly from
    vector coordinates produced by squares_generation.py, our own numpy port
    of their generator), this reads their actual curve_N.png/square_N.png
    files directly: zero port-fidelity risk, at the cost of losing the
    exact square_corners vector (see embedding_conditions below).

    Files are already exactly 128x128, single-channel, binary {0, 255} —
    confirmed against real samples, no resizing needed regardless of
    `image_size` (passing a different image_size here is not supported;
    it's accepted only for config-interface parity with SquaresDataset).

    The square PNGs are the *unfilled* outline (config_curves.yaml's
    square_kwargs.fill_shape=false) — filling happens here at load time via
    the same fill_interior() helper render_square uses, matching
    CurveImageDataset._fill_square_interior's own post-load fill order.

    embedding_conditions is NOT the true generation-time square corners —
    this data source only retains the rasterized image, not the vector
    corners — it's a minAreaRect fit to the loaded (unfilled) square mask
    instead. Only used for debugging/visualization parity, never model
    conditioning (see SquaresDataset's docstring), so this approximation is
    harmless; may be all-zero for a degenerate (contour-less) mask.

    Args:
        image_dir: local directory containing curve_N.png/square_N.png. If
            None, downloads+extracts hf_filename (a zip) from hf_repo.
        id_range: [start, end) instance-id slice this instance of the
            dataset exposes — use disjoint ranges for train_dataset/
            val_dataset in config (e.g. [0, 98000) / [98000, 100000)), since
            this data source has no pre-split train/val files.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        hf_repo: str = "pskiers/trm-diffusion-inscribed-square",
        hf_filename: str = "curves_data.zip",
        image_size: int = 128,
        fill_square: bool = True,
        id_range: tuple[int, int] = (0, 100000),
    ):
        super().__init__()
        self.fill_square = fill_square
        self.ids = list(range(id_range[0], id_range[1]))

        if image_dir is None:
            import zipfile
            from huggingface_hub import hf_hub_download

            zip_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_filename)
            extract_dir = Path(zip_path).with_suffix("")
            marker = extract_dir / ".extracted"
            if not marker.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
                marker.touch()
            # zip's own top-level folder (e.g. "curves_data/") nested inside extract_dir
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            image_dir = str(subdirs[0]) if subdirs else str(extract_dir)

        self.image_dir = Path(image_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> DataSample:
        instance_id = self.ids[idx]
        curve_img = np.array(Image.open(self.image_dir / f"curve_{instance_id}.png").convert("L"), dtype=np.float32)
        square_img = np.array(Image.open(self.image_dir / f"square_{instance_id}.png").convert("L"), dtype=np.float32)

        cond_img = (curve_img / 255.0) * 2 - 1
        target_img = (square_img / 255.0) * 2 - 1
        if self.fill_square:
            target_img = fill_interior(target_img)

        square_corners = _approx_corners_from_mask(target_img)

        return DataSample(
            images=torch.from_numpy(target_img[None, :, :]),
            spatial_conditions=torch.from_numpy(cond_img[None, :, :]),
            embedding_conditions=torch.from_numpy(square_corners),
            puzzle_id=torch.tensor(instance_id, dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)


def _approx_corners_from_mask(square_img: np.ndarray) -> np.ndarray:
    """minAreaRect corners of the (possibly filled) square mask, in the same
    [-1,1] normalized space render_square's inputs use — an approximation
    for SquaresImagePairDataset's embedding_conditions (see its docstring),
    not the exact generation-time corners. Returns zeros for an empty mask."""
    mask = (square_img < 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((4, 2), dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(cnt))  # (4, 2) pixel coords
    H, W = square_img.shape
    box[:, 0] = (box[:, 0] / (W - 1)) * 2 - 1
    box[:, 1] = (box[:, 1] / (H - 1)) * 2 - 1
    return box.astype(np.float32)
