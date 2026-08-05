#!/usr/bin/env python3
"""
datasets/squares_generation.py — offline, one-time Inscribed Square instance
generator, adapted from `data/Curves.py`'s `CurveImageDataset._generate_curve_and_square`
in https://github.com/kariander1/visual-geo-solver (the code release for
"Visual Diffusion Models are Geometric Solvers", arXiv 2510.21697).

Unlike Steiner Tree/Max-Area Polygon, there is no NP-hard solve here: a
random closed curve is built by taking a circle (10% of the time) or a
smooth harmonic (Fourier) perturbation of one, then deforming it just enough
(via a periodic cubic-spline correction) so it passes exactly through the
corners of 1-5 randomly placed squares — the "solving" is done by
construction, not search. This script ports that construction (numpy-only;
the original does the same math in torch, but nothing here needs
autograd/GPU) so training data is reproducible/offline like Steiner/Polygon,
rather than generated on-the-fly as the original's CurveImageDataset does
when save_on_generate=False.

One curve draw yields between 1 and `max_squares` (x, y curve, square)
pairs — consecutive instance_ids may share the same curve_points with a
different square_corners, exactly matching the original's own generator
(`_generate_curve_and_square` is a Python generator yielding one square at a
time per curve; see `_generate_worker`'s chunked-parallel consumption
pattern, ported below as `_generate_chunk`). This is intentional: it trains
the model on the fact that a given curve generally admits more than one
valid inscribed square.

Output is a single NDJSON file (one JSON object per line):
{instance_id, is_circle, curve_points, square_corners}. Both point lists are
already fit into a padded [-1, 1]^2 box (matching SquaresDataset's rendering
convention directly, no further normalization needed at __getitem__ time).
curve_points has 1000 entries when --use-spline (the default, matching the
original's hardcoded `u_fine = linspace(0, 1, 1000)` resampling resolution),
or num_points+1 otherwise.

Usage:
    python datasets/squares_generation.py --num-instances 20000 \
        --output data/squares_data/train.ndjson --seed-offset 0
    python datasets/squares_generation.py --num-instances 2000 \
        --output data/squares_data/val.ndjson --seed-offset 1000000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline, splev, splprep
from tqdm import tqdm


def is_self_intersecting(x: np.ndarray, y: np.ndarray) -> bool:
    """Ported from utils/viz.py's is_self_intersecting."""
    from shapely.geometry import LineString

    curve = LineString(np.column_stack([x, y]))
    return not curve.is_simple


def _fit_to_unit_box_with_padding(
    curve_pts: np.ndarray,   # (N, 2)
    square_pts: np.ndarray,  # (4, 2)
    padding_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Ported from data/Curves.py's _fit_to_unit_box_with_padding (numpy
    instead of torch — pure min/max/broadcast, no autograd needed here).
    Scales+shifts both point sets together so their combined bounding box
    fits as large as possible inside [-1, 1]^2 with uniform padding on all
    sides. Does NOT enforce centering beyond that."""
    all_points = np.concatenate([curve_pts, square_pts], axis=0)

    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    size_xy = max_xy - min_xy

    target_size = 2.0 * (1.0 - padding_ratio)
    scale = target_size / size_xy.max()

    scaled = (all_points - min_xy) * scale
    final_min = scaled.min(axis=0)
    final_max = scaled.max(axis=0)
    shift_to_center = -((final_min + final_max) / 2)
    scaled_and_shifted = scaled + shift_to_center

    curve_scaled = scaled_and_shifted[: curve_pts.shape[0]]
    square_scaled = scaled_and_shifted[curve_pts.shape[0] :]
    return curve_scaled, square_scaled


def generate_curve_and_square(
    rng: np.random.Generator,
    H_range: tuple[int, int] = (6, 30),
    num_points: int = 500,
    radius_range: tuple[float, float] = (0.3, 0.7),
    rotation_range: tuple[float, float] = (0.0, 2 * np.pi),
    use_spline: bool = True,
    shift_range: float = 0.5,
    circles_ratio: float = 0.1,
    max_squares: int = 5,
):
    """Ported near-verbatim from data/Curves.py's
    CurveImageDataset._generate_curve_and_square (numpy instead of torch;
    `rng` replaces the class's persistent self._rng / seed handling — callers
    own their own np.random.Generator instance for reproducibility).

    Yields (curve_pts, square_pts) pairs — (N, 2) and (4, 2) float arrays in
    a padded [-1, 1]^2 box — one per square, 1 to max_squares per curve draw
    (always 1 for circles).
    """
    is_circle = rng.random() < circles_ratio
    while True:
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        num_squares = int(rng.integers(1, max_squares + 1))
        if is_circle:
            num_squares = 1

        square_centers = rng.uniform(-0.7, 0.7, size=(num_squares, 2))
        square_rotations = rng.uniform(rotation_range[0], rotation_range[1], size=num_squares)

        square_points_all = []
        target_radius = None
        for center, rotation in zip(square_centers, square_rotations):
            target_radius = rng.uniform(radius_range[0], radius_range[1])
            base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64) * (target_radius / np.sqrt(2))
            rot_mat = np.array(
                [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
            )
            rotated_square = base @ rot_mat.T
            rotated_square += center
            square_points_all.append(rotated_square)

        square_points = np.concatenate(square_points_all, axis=0)  # (4*num_squares, 2)

        if is_circle:
            r_final = np.full_like(t, target_radius)

            square_points_all = []
            center_angles = rng.uniform(0, 2 * np.pi, size=num_squares)
            for theta_center in center_angles:
                angle_offsets = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
                rotation = rng.uniform(0, 2 * np.pi)
                square_angles = (theta_center + angle_offsets + rotation) % (2 * np.pi)

                x_c = target_radius * np.cos(square_angles)
                y_c = target_radius * np.sin(square_angles)
                square_points_all.append(np.stack([x_c, y_c], axis=1))

            square_points = np.concatenate(square_points_all, axis=0)
        else:
            H = int(rng.integers(H_range[0], H_range[1] + 1))
            rho = rng.random(H) * np.logspace(-0.5, -2.5, H)
            phi = rng.random(H) * 2 * np.pi
            r_base = np.ones_like(t)
            for h in range(1, H + 1):
                r_base += rho[h - 1] * np.sin(h * t + phi[h - 1])

            x_pts, y_pts = square_points[:, 0], square_points[:, 1]
            square_radii = np.sqrt(x_pts**2 + y_pts**2)
            square_angles = np.arctan2(y_pts, x_pts) % (2 * np.pi)

            square_indices = [int(np.argmin(np.abs(t - angle))) for angle in square_angles]
            r_current = r_base[square_indices]
            delta = square_radii - r_current

            sort_idx = np.argsort(square_angles)
            square_angles_sorted = square_angles[sort_idx]
            delta_sorted = delta[sort_idx]

            square_angles_sorted = np.append(square_angles_sorted, square_angles_sorted[0] + 2 * np.pi)
            delta_sorted = np.append(delta_sorted, delta_sorted[0])

            correction_spline = CubicSpline(square_angles_sorted, delta_sorted, bc_type="periodic")
            r_final = r_base + correction_spline(t)

        x = r_final * np.cos(t)
        y = r_final * np.sin(t)

        if use_spline:
            tck, _ = splprep([x, y], s=0, per=True)
            u_fine = np.linspace(0, 1, 1000)
            x, y = splev(u_fine, tck)
            x, y = np.asarray(x), np.asarray(y)
        else:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        if is_circle or not is_self_intersecting(x, y):
            break

    # Snap nearest curve points to exact square corners so the polyline
    # passes through every corner exactly (alignment = 0).
    for j in range(square_points.shape[0]):
        sx, sy = square_points[j]
        dists_sq = (x - sx) ** 2 + (y - sy) ** 2
        closest_idx = int(np.argmin(dists_sq))
        x[closest_idx] = sx
        y[closest_idx] = sy

    curve_pts = np.stack([x, y], axis=1)

    shift = rng.uniform(-shift_range, shift_range, size=(2,))
    curve_pts = curve_pts + shift

    num_squares = square_points.shape[0] // 4
    for i in range(num_squares):
        square_pts = square_points[4 * i : 4 * (i + 1), :] + shift
        yield _fit_to_unit_box_with_padding(curve_pts.copy(), square_pts.copy()), is_circle


def _generate_chunk(args) -> list[dict]:
    """Ported from data/Curves.py's _generate_worker: consumes a contiguous
    instance_id range from one curve generator, starting a fresh curve
    (fresh seed) whenever the current one's squares run out — so instance
    ids within a chunk may share curve_points, exactly like the original."""
    start, end, seed, gen_kwargs = args
    rng = np.random.default_rng(seed)
    gen = generate_curve_and_square(rng, **gen_kwargs)

    records = []
    for instance_id in range(start, end):
        try:
            (curve_pts, square_pts), is_circle = next(gen)
        except StopIteration:
            rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            gen = generate_curve_and_square(rng, **gen_kwargs)
            (curve_pts, square_pts), is_circle = next(gen)
        records.append(
            {
                "instance_id": instance_id,
                "is_circle": bool(is_circle),
                "curve_points": curve_pts.tolist(),
                "square_corners": square_pts.tolist(),
            }
        )
    return records


def main():
    p = argparse.ArgumentParser(description="Generate Inscribed Square instances (offline, one-time)")
    p.add_argument("--num-instances", type=int, default=20000)
    p.add_argument("--output", type=str, required=True, help="output .ndjson path")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0, help="base seed for this file's chunks")
    p.add_argument("--chunk-size", type=int, default=200)
    # Generation params — defaults match config_curves.yaml (the config the
    # paper's released Inscribed-Square checkpoint was actually trained
    # with), not CurveImageDataset's own class defaults where they differ
    # (H_range and side_length_range).
    p.add_argument("--h-range", type=int, nargs=2, default=(6, 30))
    p.add_argument("--num-points", type=int, default=500)
    p.add_argument("--side-length-range", type=float, nargs=2, default=(0.3, 0.7),
                   help="range for the square's CIRCUMSCRIBED radius (distance center->corner), "
                        "despite the name — matches the original config key verbatim.")
    p.add_argument("--rotation-range", type=float, nargs=2, default=(0.0, 2 * np.pi))
    p.add_argument("--shift-range", type=float, default=0.5)
    p.add_argument("--circles-ratio", type=float, default=0.1)
    p.add_argument("--max-squares", type=int, default=5)
    p.add_argument("--no-spline", action="store_true", help="disable spline resampling (use_spline=False)")
    args = p.parse_args()

    gen_kwargs = dict(
        H_range=tuple(args.h_range),
        num_points=args.num_points,
        radius_range=tuple(args.side_length_range),
        rotation_range=tuple(args.rotation_range),
        use_spline=not args.no_spline,
        shift_range=args.shift_range,
        circles_ratio=args.circles_ratio,
        max_squares=args.max_squares,
    )

    num_workers = args.num_workers or min(mp.cpu_count(), 12)
    chunk_size = args.chunk_size
    chunks = []
    base_rng = np.random.default_rng(args.seed_offset)
    for start in range(0, args.num_instances, chunk_size):
        end = min(start + chunk_size, args.num_instances)
        chunks.append((start, end, int(base_rng.integers(0, 2**31)), gen_kwargs))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_generate_chunk, c): c for c in chunks}
        for fut in tqdm(as_completed(futures), total=len(chunks), desc="Generating Inscribed Square instances"):
            all_records.extend(fut.result())

    all_records.sort(key=lambda d: d["instance_id"])
    with open(out_path, "w") as f:
        for d in all_records:
            f.write(json.dumps(d))
            f.write("\n")

    print(f"Wrote {len(all_records)} instances to {out_path}")


if __name__ == "__main__":
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    main()
