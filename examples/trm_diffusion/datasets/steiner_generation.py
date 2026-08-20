#!/usr/bin/env python3
"""
datasets/steiner_generation.py — offline, one-time Steiner tree instance
generator, adapted from `steiner_generation/generate_steiner_data.py` in
https://github.com/kariander1/visual-geo-solver (the code release for
"Visual Diffusion Models are Geometric Solvers", arXiv 2510.21697).

Solves each instance to the exact optimum with the real GeoSteiner solver
(third_party/geosteiner-5.3, CC BY-NC 4.0 — academic/non-commercial use;
see third_party/geosteiner-5.3/README.md) via the classic `efst | bb`
pipeline, then parses `bb`'s solution-certificate text output into an
explicit graph: terminal points, any extra Steiner (junction) points the
optimal solution introduces, and the edge list connecting them.

Unlike the original script, this does NOT render any images or write
per-instance visualization — SteinerTreeDataset (steiner_dataset.py) renders
condition/target images on the fly from the saved point/edge data, exactly
like MazeDataset does for mazes. Output is a single NDJSON file (one JSON
object per line): {instance_id, num_terminals, num_steiner_points,
terminal_points, steiner_points, edges, edge_weights, total_length}.
Coordinates are normalized to [0, 1]^2, not pixel space.

This script is a one-time, offline step — it requires the compiled `efst`/
`bb` binaries (see third_party/geosteiner-5.3/build_without_libtool.sh).
Training/eval never runs this script or needs GeoSteiner; they only read the
resulting NDJSON file.

Usage:
    python datasets/steiner_generation.py --num-instances 20000 \
        --min-points 10 --max-points 20 \
        --output data/steiner_data/train.ndjson --seed-offset 0
    python datasets/steiner_generation.py --num-instances 2000 \
        --min-points 10 --max-points 20 \
        --output data/steiner_data/val.ndjson --seed-offset 1000000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

_DEFAULT_GEOSTEINER_PATH = Path(__file__).resolve().parent.parent / "third_party" / "geosteiner-5.3"


def generate_random_points(
    num_points: int,
    rng: np.random.RandomState,
    min_dist: float = 0.1,
    pad: float = 0.03,
) -> np.ndarray:
    """Rejection-sample `num_points` in [pad, 1-pad]^2 at least `min_dist` apart.

    Matches generate_steiner_data.py's generate_random_points (there, pad is
    derived from node_radius/edge_width/image_size — since we render on our
    own canvas at __getitem__ time rather than baking pixel geometry in here,
    `pad` is just passed directly as a normalized-coordinate margin).
    """
    points: list[np.ndarray] = []
    attempts = 0
    max_attempts = 1000
    while len(points) < num_points and attempts < max_attempts:
        candidate = rng.rand(2) * (1 - 2 * pad) + pad
        if all(np.linalg.norm(candidate - p) > min_dist for p in points):
            points.append(candidate)
        attempts += 1
    if len(points) < num_points:
        raise ValueError(f"could not place {num_points} points after {max_attempts} attempts")
    return np.array(points)


def solve_steiner_tree(points: np.ndarray, geosteiner_path: Path, timeout: int = 30):
    """Run the `efst | bb` GeoSteiner pipeline; returns (steiner_points, total_length, raw_bb_output)."""
    efst = str((geosteiner_path / "efst").absolute())
    bb = str((geosteiner_path / "bb").absolute())
    points_str = "\n".join(f"{x} {y}" for x, y in points)

    efst_proc = subprocess.Popen(
        [efst], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=geosteiner_path
    )
    bb_proc = subprocess.Popen(
        [bb], stdin=efst_proc.stdout, stdout=subprocess.PIPE, text=True, cwd=geosteiner_path
    )
    efst_proc.stdin.write(points_str)
    efst_proc.stdin.close()
    efst_proc.stdout.close()

    try:
        bb_output, _ = bb_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        bb_proc.kill()
        efst_proc.kill()
        raise RuntimeError("GeoSteiner solve timed out")
    if bb_proc.returncode != 0:
        raise RuntimeError(f"GeoSteiner solve failed (bb exit {bb_proc.returncode})")

    steiner_points = []
    length = None
    for line in bb_output.split("\n"):
        if line.strip().startswith("% @C"):
            parts = line.strip().split()
            if len(parts) >= 3:
                steiner_points.append([float(parts[2]), float(parts[3])])
        if "length =" in line:
            m = re.search(r"length = ([\d.]+)", line)
            if m:
                length = float(m.group(1))
    return np.array(steiner_points), length, bb_output


def extract_graph_structure(terminal_points: np.ndarray, steiner_points: np.ndarray, bb_output: str):
    """Parse bb's solution-certificate text into an explicit edge list.

    Ported near-verbatim from generate_steiner_data.py's
    extract_graph_structure — this is the part that actually decodes
    GeoSteiner's terse `T`/`S` connection-line format, not something worth
    re-deriving independently. Edge indices: [0, num_terminals) are
    terminals in the original order, [num_terminals, ...) are Steiner points.
    """
    edges: list[list[int]] = []
    edge_weights: list[float] = []
    all_points = (
        np.vstack([terminal_points, steiner_points]) if len(steiner_points) > 0 else terminal_points
    )

    lines = bb_output.split("\n")
    for line in lines:
        line = line.strip()
        if "T" in line and "S" in line and not line.startswith("%"):
            parts = line.split()
            try:
                if len(parts) == 5 and parts[1] == "T" and parts[3] == "T" and parts[4] == "S":
                    t1, t2 = int(parts[0]), int(parts[2])
                    if 0 <= t1 < len(terminal_points) and 0 <= t2 < len(terminal_points):
                        edge = [t1, t2]
                        if edge not in edges and [t2, t1] not in edges:
                            edges.append(edge)
                            edge_weights.append(float(np.linalg.norm(all_points[t1] - all_points[t2])))
                elif len(parts) >= 4:
                    if parts[1] == "T":
                        terminal_idx = int(parts[0])
                        sx, sy = float(parts[2]), float(parts[3])
                    else:
                        sx, sy = float(parts[0]), float(parts[1])
                        terminal_idx = int(parts[2])
                    steiner_idx = None
                    for s_idx, sp in enumerate(steiner_points):
                        if abs(sp[0] - sx) < 1e-6 and abs(sp[1] - sy) < 1e-6:
                            steiner_idx = len(terminal_points) + s_idx
                            break
                    if steiner_idx is not None and 0 <= terminal_idx < len(terminal_points):
                        edge = [terminal_idx, steiner_idx]
                        if edge not in edges and [steiner_idx, terminal_idx] not in edges:
                            edges.append(edge)
                            edge_weights.append(
                                float(np.linalg.norm(all_points[terminal_idx] - all_points[steiner_idx]))
                            )
            except (ValueError, IndexError):
                continue

    for line in lines:
        line = line.strip()
        if not line.startswith("%") and "S" in line and "T" not in line:
            parts = line.split()
            try:
                if len(parts) >= 5 and parts[-1] == "S":
                    s1x, s1y = float(parts[0]), float(parts[1])
                    s2x, s2y = float(parts[2]), float(parts[3])
                    s1_idx = s2_idx = None
                    for s_idx, sp in enumerate(steiner_points):
                        if abs(sp[0] - s1x) < 1e-6 and abs(sp[1] - s1y) < 1e-6:
                            s1_idx = len(terminal_points) + s_idx
                        if abs(sp[0] - s2x) < 1e-6 and abs(sp[1] - s2y) < 1e-6:
                            s2_idx = len(terminal_points) + s_idx
                    if s1_idx is not None and s2_idx is not None:
                        edge = [s1_idx, s2_idx]
                        if edge not in edges and [s2_idx, s1_idx] not in edges:
                            edges.append(edge)
                            edge_weights.append(float(np.linalg.norm(all_points[s1_idx] - all_points[s2_idx])))
            except (ValueError, IndexError):
                continue

    return edges, edge_weights


def _generate_one(args) -> dict:
    instance_id, num_points, geosteiner_path, seed = args
    try:
        rng = np.random.RandomState(seed)
        points = generate_random_points(num_points, rng)
        steiner_points, length, raw_output = solve_steiner_tree(points, geosteiner_path)
        edges, edge_weights = extract_graph_structure(points, steiner_points, raw_output)
        return {
            "status": "success",
            "data": {
                "instance_id": instance_id,
                "num_terminals": len(points),
                "num_steiner_points": len(steiner_points),
                "terminal_points": points.tolist(),
                "steiner_points": steiner_points.tolist(),
                "edges": edges,
                "edge_weights": edge_weights,
                "total_length": length,
            },
        }
    except Exception as e:  # noqa: BLE001 — surfaced via the "failed" record, not raised
        return {"status": "failed", "id": instance_id, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description="Generate Steiner tree instances via GeoSteiner (offline, one-time)")
    p.add_argument("--num-instances", type=int, default=20000)
    p.add_argument("--min-points", type=int, default=10)
    p.add_argument("--max-points", type=int, default=20)
    p.add_argument("--geosteiner-path", type=str, default=str(_DEFAULT_GEOSTEINER_PATH))
    p.add_argument("--output", type=str, required=True, help="output .ndjson path")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0, help="added to instance_id for the per-instance RNG seed")
    args = p.parse_args()

    geosteiner_path = Path(args.geosteiner_path)
    for binary in ("efst", "bb"):
        if not (geosteiner_path / binary).exists():
            raise FileNotFoundError(
                f"{geosteiner_path / binary} not found — build it first: "
                f"cd {geosteiner_path} && ./build_without_libtool.sh"
            )

    num_workers = args.num_workers or min(mp.cpu_count(), 12)
    rng = np.random.RandomState(args.seed_offset - 1 if args.seed_offset else 0)
    tasks = [
        (i, int(rng.randint(args.min_points, args.max_points + 1)), geosteiner_path, args.seed_offset + i)
        for i in range(args.num_instances)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, failed = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_generate_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Generating Steiner instances"):
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
