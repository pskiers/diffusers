#!/usr/bin/env python3
"""
datasets/polygon_generation.py — offline, one-time Max-Area Simple Polygon
instance generator, adapted from `polygon_generation/max_area_polygonization.py`
and `generate_polygons_data.py` in https://github.com/kariander1/visual-geo-solver
(the code release for "Visual Diffusion Models are Geometric Solvers",
arXiv 2510.21697).

Given n random points, finds the simple (non-self-intersecting) polygon of
*maximum area* using every point exactly once as a vertex — unlike Steiner
Tree, no new points are ever introduced, so this is exact search over point
*orderings*, not tree topology. Solved via the paper's own exact backtracking
DFS (angle-sorted candidate order + segment-intersection pruning) — pure
Python, no external solver/build needed (unlike Steiner's GeoSteiner
dependency), fast enough directly at this dataset's scale (7-12 points,
matching the paper's own reported experiments: worst case ~1.5s at n=12).

Output is a single NDJSON file (one JSON object per line):
{instance_id, num_points, points, polygon_order, polygon_area}.
Coordinates are normalized to [0, 1]^2. No images are generated — rendering
happens at __getitem__ time (datasets/polygon_dataset.py), exactly like
Steiner Tree.

Usage:
    python datasets/polygon_generation.py --num-instances 20000 \
        --min-points 7 --max-points 12 \
        --output data/polygon_data/train.ndjson --seed-offset 0
    python datasets/polygon_generation.py --num-instances 2000 \
        --min-points 7 --max-points 12 \
        --output data/polygon_data/val.ndjson --seed-offset 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

Point = tuple[float, float]
EPS = 1e-12


def generate_random_points(
    num_points: int,
    rng: np.random.RandomState,
    min_dist: float = 0.15,
    pad: float = 0.03,
) -> np.ndarray:
    """Rejection-sample `num_points` in [pad, 1-pad]^2 at least `min_dist`
    apart — mirrors generate_polygons_data.py's generate_random_points
    (Python fallback path; min_dist=0.15 matches its default)."""
    points: list[np.ndarray] = []
    attempts = 0
    max_attempts = 10000
    while len(points) < num_points and attempts < max_attempts:
        candidate = rng.rand(2) * (1 - 2 * pad) + pad
        if all(np.linalg.norm(candidate - p) > min_dist for p in points):
            points.append(candidate)
        attempts += 1
    if len(points) < num_points:
        raise ValueError(f"could not place {num_points} points after {max_attempts} attempts")
    return np.array(points)


def _orientation(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax, ay, bx, by, cx, cy):
    return min(ax, bx) - EPS <= cx <= max(ax, bx) + EPS and min(ay, by) - EPS <= cy <= max(ay, by) + EPS


def _segments_properly_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Ported verbatim (module-level helpers, not the class) from
    max_area_polygonization.py — proper segment-intersection test used both
    by the DFS solver's pruning and (separately, in our own eval module) to
    validate a *generated* polygon's edge set for self-intersections."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)
    if (o1 * o2 < -EPS) and (o3 * o4 < -EPS):
        return True
    if abs(o1) <= EPS and _on_segment(ax, ay, bx, by, cx, cy):
        if (abs(cx - ax) > EPS or abs(cy - ay) > EPS) and (abs(cx - bx) > EPS or abs(cy - by) > EPS):
            return True
    if abs(o2) <= EPS and _on_segment(ax, ay, bx, by, dx, dy):
        if (abs(dx - ax) > EPS or abs(dy - ay) > EPS) and (abs(dx - bx) > EPS or abs(dy - by) > EPS):
            return True
    if abs(o3) <= EPS and _on_segment(cx, cy, dx, dy, ax, ay):
        if (abs(ax - cx) > EPS or abs(ay - cy) > EPS) and (abs(ax - dx) > EPS or abs(ay - dy) > EPS):
            return True
    if abs(o4) <= EPS and _on_segment(cx, cy, dx, dy, bx, by):
        if (abs(bx - cx) > EPS or abs(by - cy) > EPS) and (abs(bx - dx) > EPS or abs(by - dy) > EPS):
            return True
    return False


def _area_signed(points: list[Point], order: list[int]) -> float:
    area2 = 0.0
    n = len(order)
    for i in range(n):
        x1, y1 = points[order[i]]
        x2, y2 = points[order[(i + 1) % n]]
        area2 += x1 * y2 - x2 * y1
    return area2 * 0.5


def _choose_anchor(points: list[Point]) -> int:
    best = 0
    bx, by = points[0]
    for i, (x, y) in enumerate(points):
        if (y < by - EPS) or (abs(y - by) <= EPS and x < bx - EPS):
            best = i
            bx, by = x, y
    return best


def _precompute_cross(points: list[Point]):
    n = len(points)
    cross = [[[[False] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = points[i]
            b = points[j]
            for k in range(n):
                for l in range(n):
                    if k == l or i in (k, l) or j in (k, l):
                        continue
                    if _segments_properly_intersect(a, b, points[k], points[l]):
                        cross[i][j][k][l] = True
    return cross


def max_area_polygon(
    points: list[Point], time_limit: Optional[float] = None, precompute_cross: bool = True
) -> tuple[list[int], float]:
    """Exact backtracking solver for Maximum-Area simple polygonization
    (n <~ 12) — ported near-verbatim from max_area_polygonization.py. Angle-
    sorted candidate order + segment-crossing pruning keeps this fast enough
    to run directly (no C++/ILP acceleration needed): worst case ~1.5s at
    n=12 (measured), well under `time_limit`.
    """
    n = len(points)
    assert n >= 3
    start = time.time()
    anchor = _choose_anchor(points)
    cross = _precompute_cross(points) if precompute_cross else None

    used = [False] * n
    used[anchor] = True
    path = [anchor]
    best_order: list[int] = []
    best_area = -1.0

    cx = sum(x for x, _ in points) / n
    cy = sum(y for _, y in points) / n
    angles = [math.atan2(y - cy, x - cx) for (x, y) in points]
    angle_order = sorted(range(n), key=lambda i: angles[i])

    def last_edge_intersects(v: int) -> bool:
        if len(path) < 2:
            return False
        u = path[-1]
        if cross is not None:
            for i in range(len(path) - 2):
                p, q = path[i], path[i + 1]
                if cross[u][v][p][q]:
                    return True
        else:
            a, b = points[u], points[v]
            for i in range(len(path) - 2):
                p, q = path[i], path[i + 1]
                if _segments_properly_intersect(a, b, points[p], points[q]):
                    return True
        return False

    def closing_edge_intersects() -> bool:
        u, v = path[-1], path[0]
        if cross is not None:
            for i in range(len(path) - 1):
                p, q = path[i], path[i + 1]
                if p in (u, v) or q in (u, v):
                    continue
                if cross[u][v][p][q]:
                    return True
        else:
            a, b = points[u], points[v]
            for i in range(len(path) - 1):
                p, q = path[i], path[i + 1]
                if p in (u, v) or q in (u, v):
                    continue
                if _segments_properly_intersect(a, b, points[p], points[q]):
                    return True
        return False

    def dfs():
        nonlocal best_order, best_area
        if time_limit is not None and (time.time() - start) > time_limit:
            return
        if len(path) == n:
            if closing_edge_intersects():
                return
            s = _area_signed(points, path)
            if s <= EPS:
                return
            if s > best_area + 1e-15:
                best_area = s
                best_order = path.copy()
            return
        for v in angle_order:
            if used[v]:
                continue
            if len(path) == n - 1 and v == anchor:
                continue
            if last_edge_intersects(v):
                continue
            used[v] = True
            path.append(v)
            dfs()
            path.pop()
            used[v] = False

    dfs()
    return best_order, best_area


def _generate_one(args) -> dict:
    instance_id, num_points, seed = args
    try:
        rng = np.random.RandomState(seed)
        points = generate_random_points(num_points, rng)
        points_list = [(float(x), float(y)) for x, y in points]
        order, area = max_area_polygon(points_list, time_limit=30, precompute_cross=True)
        if not order or area <= 0:
            raise RuntimeError("no valid polygonalization found")
        return {
            "status": "success",
            "data": {
                "instance_id": instance_id,
                "num_points": num_points,
                "points": points.tolist(),
                "polygon_order": order,
                "polygon_area": area,
            },
        }
    except Exception as e:  # noqa: BLE001 — surfaced via the "failed" record, not raised
        return {"status": "failed", "id": instance_id, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description="Generate Max-Area Polygon instances (offline, one-time)")
    p.add_argument("--num-instances", type=int, default=20000)
    p.add_argument("--min-points", type=int, default=7)
    p.add_argument("--max-points", type=int, default=12)
    p.add_argument("--output", type=str, required=True, help="output .ndjson path")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0, help="added to instance_id for the per-instance RNG seed")
    args = p.parse_args()

    num_workers = args.num_workers or min(mp.cpu_count(), 12)
    rng = np.random.RandomState(args.seed_offset - 1 if args.seed_offset else 0)
    tasks = [
        (i, int(rng.randint(args.min_points, args.max_points + 1)), args.seed_offset + i)
        for i in range(args.num_instances)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, failed = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_generate_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Generating polygon instances"):
            r = fut.result()
            if r["status"] == "success":
                results.append(r["data"])
            else:
                failed.append(r)

    results.sort(key=lambda d: d["instance_id"])
    with open(out_path, "w") as f:
        for d in results:
            f.write(json.dumps(d))
            f.write("\n")

    print(f"Wrote {len(results)}/{len(tasks)} instances to {out_path}")
    if failed:
        print(f"{len(failed)} instances failed, e.g.: {failed[:3]}")


if __name__ == "__main__":
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    main()
