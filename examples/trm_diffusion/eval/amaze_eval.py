"""
Implements the AMAZE paper (arXiv:2604.22868) metrics so Maze and
Queen report the SAME quantities:
    Coverage  = |predicted ∩ GT| / |GT|
    Violation = |predicted different from GT| / |predicted|
    Pass@1    = fraction of exact solves (predicted == GT ⇔ Coverage=1 & Violation=0)
    Pass@5    = fraction of exact solves in 5 attempts
    MSE-In / MSE-Out = MSE inside / outside the path_mask (mask_img)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from third_party.amaze.infer.maze_metrics import MazeRewardFunction, extract_blue_path


_METRIC_KEYS: Tuple[str, ...] = (
    "mse_inside",
    "mse_outside",
    "gt_cell_coverage",
    "background_violation",
    "pass",
)


class AmazeMetrics:
    """Stateful accumulator for Amaze eval metrics.

    Score generated images with ``compute_and_accumulate_metrics`` (one batch
    per call — optionally with K attempts/sample for Pass@K), then read the
    aggregate once with ``return_metrics``.

    Per-sample keys (from _compute_{maze,queen}_metrics): mse_inside, mse_outside,
    gt_cell_coverage, background_violation and pass (1.0 iff the predicted
    cell-set exactly matches GT). With K > 1 attempts a ``pass_at_{K}`` key is
    added (1.0 iff ANY attempt is an exact solve). In the aggregate ``mean_pass``
    is the paper's Pass@1 and ``mean_pass_at_{K}`` its Pass@K.
    """

    def __init__(self, device=None, task: str = "maze"):
        if task not in ("maze", "queens"):
            raise ValueError(f"Unknown task: {task}")
        self.device = device
        self.task = task
        self._per_key: dict[str, list[float]] = {}
        self._n = 0

        if task == "maze":
            self._mrf = MazeRewardFunction()
            self._score = self._compute_maze_metrics
        else:
            self._score = self._compute_queen_metrics

    def reset(self) -> None:
        """Drop all accumulated samples (call between eval runs)."""
        self._per_key = {}
        self._n = 0

    def compute_and_accumulate_metrics(self, inputs: torch.Tensor, metadata: List[Dict]) -> List[Dict[str, float]]:
        """Score a BATCH of generated images against their GT and accumulate them.

        Args:
            inputs: generated images in [0, 1] — either (B, C, H, W) for one
                attempt/sample, or (B, K, C, H, W) for K attempts/sample (Pass@K).
            metadata: list of B AmazeDataset metadata dicts (shared across attempts).

        Per sample we record the first-attempt metrics (mse_*, coverage, violation
        and ``pass`` = Pass@1) plus, when K > 1, ``pass_at_{K}`` = 1.0 iff ANY of
        the K attempts is an exact solve (best-of-K). Returns the per-sample dicts.
        """
        if inputs.dim() == 4:                          # (B, C, H, W) -> (B, 1, C, H, W)
            inputs = inputs.unsqueeze(1)
        n_attempts = inputs.shape[1]
        all_scores: List[Dict[str, float]] = []

        for sample_attempts, meta in zip(inputs, metadata):   # sample_attempts: (K, C, H, W)
            attempts: List[Dict[str, float]] = []
            for attempt in sample_attempts:
                try:
                    attempts.append(self._score(attempt, meta))
                except Exception as e:  # keep eval alive on a single bad attempt
                    print(f"Error scoring {self.task} sample {self._n}: {e}")
                    attempts.append(dict.fromkeys(_METRIC_KEYS, 0.0))

            record = dict(attempts[0])                 # first-round metrics + pass (=Pass@1)
            if n_attempts > 1:
                record[f"pass_at_{n_attempts}"] = float(any(a["pass"] >= 1.0 for a in attempts))
            for key, value in record.items():
                self._per_key.setdefault(key, []).append(float(value))
            self._n += 1
            all_scores.append(record)

        return all_scores

    def return_metrics(self) -> Dict[str, float]:
        """Aggregate every accumulated sample: global mean/std/min/max per key.

        The global mean is over ALL samples (not a mean of per-batch means), and
        ``mean_pass`` is the paper Pass@1 (fraction of exact solves).
        """
        result: Dict[str, float] = {"generated_samples": float(self._n)}
        for key, vals in self._per_key.items():
            arr = np.asarray(vals, dtype=float)
            result[f"mean_{key}"] = float(arr.mean())
            result[f"std_{key}"] = float(arr.std())
            result[f"min_{key}"] = float(arr.min())
            result[f"max_{key}"] = float(arr.max())
        return result

    # QUEENS
    def _to_pixel_array(self, img: Union[torch.Tensor, np.ndarray, Image.Image], size: Tuple[int, int]) -> np.ndarray:
        """Return an (H, W, 3) uint8 array resized to (width, height) = size."""
        if isinstance(img, torch.Tensor):
            arr = img.detach().float().cpu()
            if arr.dim() == 4:
                arr = arr.squeeze(0)
            arr = (arr.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()
            pil = Image.fromarray(arr, mode="RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
        elif isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for queen metric: {type(img)}")
        if pil.size != size:
            pil = pil.resize(size, Image.Resampling.BILINEAR)
        return np.array(pil)

    def _solution_mask(self, n: int, cell_size: int, queen_radius: float, queens: Sequence[Sequence[int]], size: Tuple[int, int]) -> np.ndarray:
        """(H, W) boolean mask covering the GT queen-marker discs (margin=0 geometry)."""
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w]
        mask = np.zeros((h, w), dtype=bool)
        for row, col in queens:
            cx = col * cell_size + cell_size // 2
            cy = row * cell_size + cell_size // 2
            mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= queen_radius ** 2
        return mask

    def _detect_queen_cells(
        self,
        generated: np.ndarray,
        cell_ids: np.ndarray,
        n: int,
        cell_size: int,
        queen_radius: float,
        dark_threshold: int = 70,
    ) -> set:
        """Detect placed-queen cells as compact near-black blobs (not grid lines)."""
        dark = np.all(generated < dark_threshold, axis=2)
        labeled, num = ndimage.label(dark, structure=np.ones((3, 3)))  # type: ignore
        if num == 0:
            return set()

        expected_area = np.pi * (queen_radius ** 2)
        lo, hi = 0.25 * expected_area, 4.0 * expected_area
        predicted = set()
        slices = ndimage.find_objects(labeled)
        for comp_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            comp_mask = labeled[sl] == comp_id
            area = comp_mask.sum()
            if not (lo <= area <= hi):
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            if h == 0 or w == 0 or max(h, w) / min(h, w) > 2.0:
                continue  # not roughly circular -> grid-line fragment
            cy = (sl[0].start + sl[0].stop) / 2.0
            cx = (sl[1].start + sl[1].stop) / 2.0
            row = int(cy // cell_size)
            col = int(cx // cell_size)
            if 0 <= row < n and 0 <= col < n:
                predicted.add((row, col))
            else:
                cid = int(cell_ids[int(cy), int(cx)])
                if 0 <= cid < n * n:
                    predicted.add((cid // n, cid % n))
        return predicted

    def _compute_queen_metrics(self, generated_image: torch.Tensor, metadata: Dict) -> Dict[str, float]:
        raw = metadata.get("sample_json")
        if raw is None:
            raise ValueError("metadata['sample_json'] is missing — not a Queen sample?")
        meta = json.loads(raw) if isinstance(raw, str) else raw

        # Queens metadata
        n = int(meta["n"])
        cell_size = int(meta["cell_size"])
        queen_radius = float(meta.get("queen_radius") or cell_size * 0.25)
        gt_queens = [tuple(q) for q in meta["queens"]]
        size = (int(meta["width"]), int(meta["height"]))
        sol_img = metadata.get("sol_img")
        cell_map = metadata.get("cell_map")
        if sol_img is None or cell_map is None:
            raise ValueError("metadata missing 'sol_img' or 'cell_map' for Queen scoring")

        gen_arr = self._to_pixel_array(generated_image, size)
        gt_arr = self._to_pixel_array(sol_img, size)
        cell_ids = self.decode_cell_map_ids(cell_map, size)

        predicted = self._detect_queen_cells(gen_arr, cell_ids, n, cell_size, queen_radius)
        gt_set = set(gt_queens)

        coverage = len(predicted & gt_set) / len(gt_set) if gt_set else 0.0
        violation = len(predicted - gt_set) / len(predicted) if predicted else 0.0
        pass_metric = float(bool(gt_set) and predicted == gt_set)

        mask = self._solution_mask(n, cell_size, queen_radius, gt_queens, size)
        diff_sq = ((gen_arr.astype(np.float64) - gt_arr.astype(np.float64)) / 255.0) ** 2
        diff_sq = diff_sq.mean(axis=2)  # per-pixel MSE across RGB
        mse_inside = float(diff_sq[mask].mean()) if mask.any() else 0.0
        mse_outside = float(diff_sq[~mask].mean()) if (~mask).any() else 0.0

        return {
            "mse_inside": mse_inside,
            "mse_outside": mse_outside,
            "gt_cell_coverage": coverage,
            "background_violation": violation,
            "pass": pass_metric,
        }

    # MAZES
    def _morph_open(self, mask: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Binary opening (erode -> dilate) with a full ksize×ksize structuring
        element, matching maze_metrics' cv2.erode+cv2.dilate path denoising without
        pulling OpenCV into this module."""
        st = np.ones((ksize, ksize), dtype=bool)
        m = mask.astype(bool)
        return ndimage.binary_dilation(ndimage.binary_erosion(m, structure=st), structure=st)

    def _chw_uint8(self, img: Union[torch.Tensor, np.ndarray, Image.Image], size: Tuple[int, int]) -> torch.Tensor:
        """(3, H, W) uint8 tensor at (W, H) = size — the layout
        maze_metrics.compute_solution_space_reward expects for the generated image."""
        arr = self._to_pixel_array(img, size)                        # (H, W, 3) uint8
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def _compute_maze_metrics(self, generated_image: torch.Tensor, metadata: Dict) -> Dict[str, float]:
        """Score one generated Maze against its ground truth (AMAZE paper §2.2).

        Args:
            generated_image: (C, H, W) tensor in [0, 1].
            metadata: an AmazeDataset sample's metadata dict. Needs 'metadata'
                    (JSON string/dict with 'path_cell_ids'), 'sol_img', 'cell_map'
                    and (for MSE) 'mask_img' (the path_mask).

        Returns dict with mse_inside, mse_outside, gt_cell_coverage,
        background_violation, pass (=Pass@1) — identical keys/semantics to
        compute_queen_metrics so both tasks aggregate the same way downstream.
        """
        raw = metadata.get("metadata")
        if raw is None:
            raise ValueError("metadata['metadata'] is missing — not a Maze sample?")
        meta = json.loads(raw) if isinstance(raw, str) else raw
        gt_cell_ids = {int(c) for c in meta.get("path_cell_ids", [])}

        # Reuse maze_metrics' metadata unpacking: (original, marked, solution, mask).
        _ori, _marked, sol_img, mask_img = self._mrf.get_reference_images("", metadata)
        cell_map = metadata.get("cell_map")
        if sol_img is None or cell_map is None:
            raise ValueError("metadata missing 'sol_img' or 'cell_map' for Maze scoring")

        # Predicted cells: the drawn blue path -> cell ids. Decode at native
        # resolution with the shared decoder (interpolating the RGB-packed ids
        # would corrupt them); align the generated image to it.
        cell_ids = self.decode_cell_map_ids(cell_map)         # (H, W) int
        size = (cell_ids.shape[1], cell_ids.shape[0])          # (W, H)
        gen_arr = self._to_pixel_array(generated_image, size)        # (H, W, 3) uint8
        blue = self._morph_open(extract_blue_path(gen_arr))  # type: ignore[arg-type]
        predicted = {int(c) for c in cell_ids[blue].tolist()}
        predicted.discard(0)                                    # 0 = wall / background

        coverage = len(predicted & gt_cell_ids) / len(gt_cell_ids) if gt_cell_ids else 0.0
        violation = len(predicted - gt_cell_ids) / len(predicted) if predicted else 0.0

        # MSE-In / MSE-Out: reuse the benchmark's own routine, which splits by the
        # path_mask (mask_img). reference_maze is only None-checked upstream, so we
        # hand it the solution image too. (Both are PIL Images from AmazeDataset.)
        if isinstance(sol_img, Image.Image) and isinstance(mask_img, Image.Image):
            gen_chw = self._chw_uint8(generated_image, size)
            sol_r = sol_img.resize(size, Image.Resampling.BILINEAR) if sol_img.size != size else sol_img
            mask_r = mask_img.resize(size, Image.Resampling.BILINEAR) if mask_img.size != size else mask_img
            mse_inside, mse_outside = self._mrf.compute_solution_space_reward(gen_chw, sol_r, sol_r, mask_r)
        else:
            mse_inside, mse_outside = 0.0, 0.0

        pass_metric = float(bool(gt_cell_ids) and predicted == gt_cell_ids)

        return {
            "mse_inside": float(mse_inside),
            "mse_outside": float(mse_outside),
            "gt_cell_coverage": coverage,
            "background_violation": violation,
            "pass": pass_metric,
        }

    # Other Helpers
    def decode_cell_map_ids(self, cell_map: Union[Image.Image, np.ndarray, torch.Tensor], size: Tuple[int, int] | None = None) -> np.ndarray:
        """RGB-packed cell_map -> (H, W) int64 cell-id array (id = R | G<<8 | B<<16).

        Single decoder shared by the Maze and Queen scorers (same formula as
        maze_metrics._decode_cell_map). Decoded at native resolution; if ``size`` =
        (W, H) is given and differs, the map is resized with NEAREST — the packed ids
        must never be interpolated (BILINEAR would blend neighbouring ids into new,
        non-existent ones).
        """
        if isinstance(cell_map, Image.Image):
            pil = cell_map.convert("RGB")
            if size is not None and pil.size != size:
                pil = pil.resize(size, Image.Resampling.NEAREST)
            arr = np.asarray(pil)
        else:
            arr = cell_map.detach().cpu().numpy() if isinstance(cell_map, torch.Tensor) else np.asarray(cell_map)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):        # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.max() <= 1.0:                                 # [0, 1] -> [0, 255]
                arr = arr * 255.0
        arr = arr.astype(np.uint32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return (r | (g << 8) | (b << 16)).astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Shared metric aggregation + wandb-table logging, used by every AMAZE eval path:
# experiments/sample_amaze_metrics.py (TRM/DiT) and experiments/score_amaze_images.py
# (BAGEL/Janus). ``AmazeMetrics`` above scores one image; the helpers below reduce the
# per-puzzle rows into the canonical ``result`` dict and push the SAME general / OOD /
# per-geometry / per-size tables to wandb. Only the way images are produced differs.
# ──────────────────────────────────────────────────────────────────────────────

# In-distribution vs held-out (OOD) test scales — shared by every AMAZE eval path.
# Env-overridable (comma lists) so a 3×3-trained model can score with e.g.
# MAZE_SCALES=3,5,7,9,11,13,16 MAZE_OOD_SCALES=8 without touching defaults.
MAZE_SCALES = [int(x) for x in os.environ.get("MAZE_SCALES", "5,7,8,9,11,13,16").split(",") if x.strip()]
MAZE_OOD_SCALES = [int(x) for x in os.environ.get("MAZE_OOD_SCALES", "3").split(",") if x.strip()]
MAZE_GEOMETRIES = ["square", "hexagon", "triangle", "circle"]
QUEEN_SCALES = [4, 5, 6, 7, 8, 9, 10]
QUEEN_OOD_SCALES = [12]

# The six aggregated per-puzzle row keys (distinct from ``_METRIC_KEYS`` above, which
# are the raw per-image scorer keys).
METRIC_KEYS = ("violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5")


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-puzzle rows to the mean of each metric (0.0 for every key if empty)."""
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    import pandas as pd

    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in METRIC_KEYS}


def maze_sample_key(geometry: str, scale) -> str:
    """Key for the representative image pair of a maze (shape, size) combo.

    Matches the FT PNG-directory name ``{geometry}_n{scale}`` so the sample dict is
    keyed identically across the from-scratch and fine-tuned paths.
    """
    return f"{geometry}_n{scale}"


def queens_sample_key(scale) -> str:
    """Key for the representative image pair of a queens size (``n{scale}``)."""
    return f"n{scale}"


def build_maze_result(per_combo: dict, ood_combo: dict) -> dict:
    """Canonical maze ``result`` dict from per-(shape,size) rows.

    ``per_combo`` / ``ood_combo`` map ``f"{geometry}_{scale}"`` -> list of per-puzzle rows
    (in-distribution scales and OOD scales respectively).
    """
    all_rows = [r for rows in per_combo.values() for r in rows]
    return {
        "task": "maze",
        "overall": aggregate(all_rows),
        "overall_ood": aggregate([r for rows in ood_combo.values() for r in rows]),
        "per_shape": {
            g: {str(s): aggregate(per_combo[f"{g}_{s}"]) for s in MAZE_SCALES}
            for g in MAZE_GEOMETRIES
        },
        "per_shape_ood": {
            g: {str(s): aggregate(ood_combo[f"{g}_{s}"]) for s in MAZE_OOD_SCALES}
            for g in MAZE_GEOMETRIES
        },
        "per_geometry": {
            g: aggregate([r for s in MAZE_SCALES for r in per_combo[f"{g}_{s}"]])
            for g in MAZE_GEOMETRIES
        },
        "per_geometry_ood": {
            g: aggregate([r for s in MAZE_OOD_SCALES for r in ood_combo[f"{g}_{s}"]])
            for g in MAZE_GEOMETRIES
        },
        "n_puzzles": len(all_rows),
    }


def build_queens_result(per_scale_rows: dict, ood_scale_rows: dict) -> dict:
    """Canonical queens ``result`` dict from per-size rows.

    ``per_scale_rows`` / ``ood_scale_rows`` map ``str(scale)`` -> list of per-puzzle rows.
    """
    all_rows = [r for rows in per_scale_rows.values() for r in rows]
    return {
        "task": "queens",
        "overall": aggregate(all_rows),
        "overall_ood": aggregate([r for rows in ood_scale_rows.values() for r in rows]),
        "per_scale": {s: aggregate(rows) for s, rows in per_scale_rows.items()},
        "per_scale_ood": {s: aggregate(rows) for s, rows in ood_scale_rows.items()},
        "n_puzzles": len(all_rows),
    }


def make_wandb_image(tensor):
    """Convert a (C,H,W) [0,1] tensor to a wandb.Image (HWC uint8); None -> None."""
    if tensor is None:
        return None
    import numpy as np
    import wandb

    arr = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return wandb.Image((arr * 255).astype("uint8"))


def log_tables(run, task: str, result: dict, samples: dict | None = None) -> None:
    """Log the general / OOD / per-geometry / per-size AMAZE tables to an open wandb run.

    ``run`` is an already-initialised wandb run (the caller owns init/finish).
    ``samples`` maps ``maze_sample_key`` / ``queens_sample_key`` -> a
    ``{"generated", "condition"}`` image pair for the per-combo image tables.
    """
    import wandb

    samples = samples or {}
    prefix = f"amaze_eval/{task}"
    for key, val in result["overall"].items():
        run.summary[f"{prefix}/overall/{key}"] = val
    for key, val in result.get("overall_ood", {}).items():
        run.summary[f"{prefix}/overall_ood/{key}"] = val

    img_cols = ["group", "generated", "condition",
                "violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5"]
    metric_cols = ["group", "violation", "coverage", "mse_inside", "mse_outside", "pass1", "pass5"]

    def _img_table(named_pairs):
        t = wandb.Table(columns=img_cols)
        for name, agg, pair in named_pairs:
            pair = pair or {}
            t.add_data(name, make_wandb_image(pair.get("generated")),
                       make_wandb_image(pair.get("condition")),
                       agg["violation"], agg["coverage"], agg["mse_inside"],
                       agg["mse_outside"], agg["pass1"], agg["pass5"])
        return t

    def _metric_table(named_aggs):
        t = wandb.Table(columns=metric_cols)
        for name, agg in named_aggs:
            t.add_data(name, agg["violation"], agg["coverage"], agg["mse_inside"],
                       agg["mse_outside"], agg["pass1"], agg["pass5"])
        return t

    if task == "maze":
        for g, by_scale in result["per_shape"].items():
            rows = [(f"{s}x{s}", by_scale[str(s)], samples.get(maze_sample_key(g, s))) for s in MAZE_SCALES]
            run.log({f"{prefix}/{g}_table": _img_table(rows)})
        for g, by_scale in result["per_shape_ood"].items():
            rows = [(f"{s}x{s}", by_scale[str(s)], samples.get(maze_sample_key(g, s))) for s in MAZE_OOD_SCALES]
            run.log({f"{prefix}/{g}_ood_table": _img_table(rows)})

        per_geometry = result["per_geometry"]
        for g, agg in per_geometry.items():
            for key, val in agg.items():
                run.summary[f"{prefix}/per_geometry/{g}/{key}"] = val
        run.log({f"{prefix}/per_geometry_table": _metric_table([(g, per_geometry[g]) for g in MAZE_GEOMETRIES])})

        per_geometry_ood = result["per_geometry_ood"]
        for g, agg in per_geometry_ood.items():
            for key, val in agg.items():
                run.summary[f"{prefix}/per_geometry_ood/{g}/{key}"] = val
        run.log({f"{prefix}/per_geometry_ood_table": _metric_table([(g, per_geometry_ood[g]) for g in MAZE_GEOMETRIES])})
    else:
        combined = {**result["per_scale"], **result["per_scale_ood"]}
        rows = [(f"{s}x{s}", combined[s], samples.get(queens_sample_key(s))) for s in combined]
        run.log({f"{prefix}/per_scale_table": _img_table(rows)})
