"""Faint-path-aware maze scoring.

Subclasses the project's AmazeMetrics (eval/amaze_eval.py, left untouched) and
swaps ONE thing: how the drawn path is extracted from the generated image.

The original detector (ear_amaze maze_metrics.extract_blue_path) keeps pixels with
HSV hue 100-130, saturation >= 50, value >= 50. A semi-transparent blue stroke on a
white background (e.g. RGB ~(228, 240, 250), saturation ~22) fails the saturation
test, so ghost branches are invisible to the scorer and never count as violations.

The faint detector keys on the blue TINT instead: a pixel is path if its hue is
blue and its blue channel exceeds its red channel by at least `tint_min`. White,
black and grey have B-R ~ 0; the red start/goal markers have B-R << 0; solid path
has B-R ~ 160 and the faint ghosts ~15-30. Everything downstream (morph-open, cell
fill threshold, endpoint tolerance, coverage/violation/pass/exact) is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

TRM_ROOT = Path("/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion")
sys.path.insert(0, str(TRM_ROOT))

from eval.amaze_eval import AmazeMetrics, MAZE_ENDPOINT_HIDDEN_WARN  # noqa: E402
from third_party.ear_amaze.infer.maze_metrics import extract_blue_path  # noqa: E402

# Hue window (OpenCV 0-179). Solid and faint strokes both sit at ~102-105.
FAINT_HUE_RANGE = (90, 130)
# Minimum B-R tint (0-255) for a pixel to count as path. Calibrated on JPEG
# ground-truth images; see calibrate.py.
FAINT_TINT_MIN = 10
# A detected stroke (8-connected component of the path mask) must span at least
# this many cell widths (bounding-box extent) to count. Drops specks of a few
# pixels that would otherwise mark a whole cell; any real branch is far longer.
FAINT_MIN_STROKE_CELLS = 0.5


def faint_blue_mask(rgb: np.ndarray, tint_min: int = FAINT_TINT_MIN,
                    hue_range=FAINT_HUE_RANGE) -> np.ndarray:
    """(H, W) uint8 {0,1} mask of solid OR semi-transparent blue path pixels."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0]
    r = rgb[..., 0].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    m = (b - r >= tint_min) & (hue >= hue_range[0]) & (hue <= hue_range[1])
    return m.astype(np.uint8)


def solid_blue_mask(rgb: np.ndarray) -> np.ndarray:
    """The original scorer's detector, for comparison."""
    return extract_blue_path(np.ascontiguousarray(rgb, dtype=np.uint8))


class FaintAmazeMetrics(AmazeMetrics):
    """AmazeMetrics with a selectable path detector ("solid" = original, "faint")."""

    def __init__(self, detector: str = "faint", tint_min: int = FAINT_TINT_MIN,
                 min_stroke_cells: float = FAINT_MIN_STROKE_CELLS, **kw):
        super().__init__(task="maze", **kw)
        if detector not in ("solid", "faint"):
            raise ValueError(detector)
        self.detector = detector
        self.tint_min = int(tint_min)
        self.min_stroke_cells = float(min_stroke_cells)

    def path_mask(self, gen_arr: np.ndarray, cell_ids: np.ndarray | None = None) -> np.ndarray:
        """Detector mask after the original 3x3 opening. The "solid" detector is the
        original scorer exactly; "faint" additionally drops strokes shorter than
        `min_stroke_cells` cell widths (needs `cell_ids` for the cell pitch)."""
        if self.detector == "solid":
            return self._morph_open(solid_blue_mask(gen_arr))
        mask = self._morph_open(faint_blue_mask(gen_arr, self.tint_min))
        if self.min_stroke_cells > 0 and cell_ids is not None:
            mask = self._drop_short_strokes(mask, cell_ids)
        return mask

    def _drop_short_strokes(self, mask: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        area = np.bincount(cell_ids.ravel())
        real = area[1:][area[1:] > 0] if area.size > 1 else area[:0]
        if real.size == 0:
            return mask
        min_len = self.min_stroke_cells * float(np.sqrt(np.median(real)))
        labeled, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        keep = np.zeros(labeled.max() + 1, dtype=bool)
        for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= min_len:
                keep[i] = True
        return keep[labeled].astype(mask.dtype)

    def analyse(self, generated_image, metadata: Dict) -> Dict:
        """Same computation as AmazeMetrics._compute_maze_metrics (copied, with the
        detector swapped), additionally returning the intermediate masks/cell sets
        for visualisation and diagnostics."""
        raw = metadata.get("metadata")
        meta = json.loads(raw) if isinstance(raw, str) else raw
        path_ids = [int(c) for c in meta.get("path_cell_ids", [])]
        gt_cell_ids = set(path_ids)

        _ori, _marked, sol_img, mask_img = self._mrf.get_reference_images("", metadata)
        cell_map = metadata.get("cell_map")
        if sol_img is None or cell_map is None:
            raise ValueError("metadata missing 'sol_img' or 'cell_map' for Maze scoring")

        cell_ids = self.decode_cell_map_ids(cell_map)
        size = (cell_ids.shape[1], cell_ids.shape[0])
        gen_arr = self._to_pixel_array(generated_image, size)
        mask = self.path_mask(gen_arr, cell_ids)
        predicted_all = self._cells_covered(mask, cell_ids)
        predicted = set(predicted_all)

        hidden, ignored = 0, set()
        if path_ids:
            ignored = self._endpoint_cells(cell_ids, (path_ids[0], path_ids[-1]))
            if ignored and gt_cell_ids - ignored:
                hidden = len((predicted ^ gt_cell_ids) & ignored)
                predicted = predicted - ignored
                gt_cell_ids = gt_cell_ids - ignored
            else:
                ignored = set()

        coverage = len(predicted & gt_cell_ids) / len(gt_cell_ids) if gt_cell_ids else 0.0
        violation = len(predicted - gt_cell_ids) / len(predicted) if predicted else 0.0

        if isinstance(sol_img, Image.Image) and isinstance(mask_img, Image.Image):
            gen_chw = torch.from_numpy(gen_arr.transpose(2, 0, 1).copy())
            sol_r = sol_img.resize(size, Image.Resampling.BILINEAR) if sol_img.size != size else sol_img
            mask_r = mask_img.resize(size, Image.Resampling.BILINEAR) if mask_img.size != size else mask_img
            mse_inside, mse_outside = self._mrf.compute_solution_space_reward(gen_chw, sol_r, sol_r, mask_r)
        else:
            mse_inside, mse_outside = 0.0, 0.0

        pass_metric = max(0.0, coverage - violation)
        exact_metric = float(bool(gt_cell_ids) and predicted == gt_cell_ids)
        if exact_metric and hidden > MAZE_ENDPOINT_HIDDEN_WARN:
            print(f"WARNING: maze board scored EXACT only because the endpoint tolerance hid {hidden} cells.")

        return {
            "scores": {
                "mse_inside": float(mse_inside), "mse_outside": float(mse_outside),
                "gt_cell_coverage": coverage, "background_violation": violation,
                "pass": pass_metric, "exact": exact_metric,
            },
            "gen_arr": gen_arr, "mask": mask, "cell_ids": cell_ids,
            "predicted": predicted, "gt": gt_cell_ids, "ignored": ignored,
            "extra": predicted - gt_cell_ids, "missing": gt_cell_ids - predicted,
        }

    def _compute_maze_metrics(self, generated_image, metadata: Dict) -> Dict[str, float]:
        return self.analyse(generated_image, metadata)["scores"]


# ── data helpers shared by the validation / scoring scripts ──────────────────
DATA_ROOT = TRM_ROOT / "data" / "amaze"
BAGEL_RUNS = {
    "square": TRM_ROOT / "runs/bagel_infer/bagel_maze_square_2026.09.22_15.57.52_50/generated_images",
    "triangle": TRM_ROOT / "runs/bagel_infer/bagel_maze_triangle_2026.09.22_15.57.52_50/generated_images",
    "hexagon": TRM_ROOT / "runs/bagel_infer/bagel_maze_hexagon_2026.09.22_15.59.11_50/generated_images",
    "circle": TRM_ROOT / "runs/bagel_infer/bagel_maze_circle_2026.09.22_15.59.11_50/generated_images",
}


def shape_scales(shape: str):
    """In-distribution test sizes present on disk for `shape` (n3 is OOD, never generated)."""
    return sorted(s for s in (int(p.stem[1:].split("_")[0]) for p in (DATA_ROOT / "maze" / shape).glob("n*_test.parquet"))
                  if s != 3)


def load_shape_index(shape: str, scales=None, wanted=None):
    """{puzzle id: (scale, metadata dict)} over every test parquet for `shape`.
    `wanted` (set of ids) keeps only those puzzles — each metadata dict holds five
    full-resolution PIL images, so indexing a whole shape at once runs out of RAM."""
    from datasets.amaze_dataset import AmazeDataset
    index = {}
    for s in (scales or shape_scales(shape)):
        pq = DATA_ROOT / "maze" / shape / f"n{s}_test.parquet"
        if not pq.exists():
            continue
        ds = AmazeDataset(str(pq), split="test", image_size=144, condition_field="m_original_img",
                          target_field="sol_img", num_channels=3, include_metadata=True)
        for i in range(len(ds)):
            md = ds[i].metadata
            if wanted is None or str(md["id"]) in wanted:
                index[str(md["id"])] = (s, md)
        del ds
    return index


def attempt_id(path: Path) -> str:
    """'7x7_<uuid>_attempt001.jpg' -> '<uuid>' (maze ids are uuid4, no underscores)."""
    return path.name.rsplit("_attempt", 1)[0].rsplit("_", 1)[1]


def load_native(path: Path) -> np.ndarray:
    """Generated image as (H, W, 3) uint8 at its native resolution (no 144px bottleneck)."""
    return np.array(Image.open(path).convert("RGB"))
