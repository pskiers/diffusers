#!/usr/bin/env python3
"""
datasets/ball_drop_generation.py — offline, one-time Ball Drop instance
generator, using pymunk (2D rigid-body physics) rather than any code ported
from a paper — unlike Steiner Tree/Max-Area Polygon (arXiv 2510.21697), this
task's source ("SketchVLM", arXiv 2604.22875) publishes no generation code,
only benchmark analysis scripts, and its own dataset
(loganbolton/sketchvlm-physics-ball-drop) is a 198-row VLM-evaluation set,
far too small to train on.

Task: a ball drops under gravity through a scene with a fixed floor split
into 4 bucket compartments by 3 internal dividers, into one of 1-3 straight
"solution" line segments (long ramps, one per height "tier") that redirect
it. Unlike Steiner Tree/Max-Area Polygon, this is NOT an optimization
problem (no "shortest"/"largest" objective) — it's a reachability/success
problem, matching PHYRE-style physics tasks: any line placement that lands
the ball in a well-defined target bucket is an equally valid ground-truth
solution. This means generation needs no solver/search at all: randomly
place the ball and solution lines in the (fixed) scene, simulate, and
whichever bucket the ball actually settles in *becomes* the recorded target
for that instance — see generate_instance's docstring. Instances where the
ball never settles cleanly (still bouncing at the step budget, or ends up
ambiguously close to a divider) are rejected and resampled with a fresh
sub-seed, mirroring Steiner/Polygon's own point-generation rejection
sampling (~44% yield per attempt here — cheap to resample since one
simulation is ~5ms).

The fixed scene layout (divider x positions, divider height, no peg
obstacles, floor) and the line-placement convention (long ramps at one of 3
height "tiers", not small local obstacles) were reverse-engineered from the
real dataset's own `metadata_json` (its `scene.bodies` list) rather than
invented — this keeps the domain gap small enough that
ball_drop_real_data.py can later import the real 198 instances as a
held-out generalization eval set, reusing this module's exact simulate/
bucket_of/DIVIDER_X/LINE_TIERS conventions. See that module for the import
pipeline and how well it matches (settle/consistency check results).

World coordinates are y-up, origin bottom-left, in "physics units" numerically
equal to pixel units at `image_size` resolution (i.e. a scene generated at
image_size=128 occupies a 128x128 physics world) — a 1:1 scale chosen to
match this project's other datasets, not a physical unit; it corresponds to
the real dataset's own 512x512 world at a fixed 4x downscale (their divider
x's 126.5/254.5/382.5 are 512's quarters; ours are exact 128 quarters).

All geometry (ball_start_x, lines) is stored normalized to [0, 1] (divided by
`size`), matching Steiner/Polygon's convention — datasets/ball_drop_dataset.py
and eval/ball_drop_eval.py both rescale by their own image_size when
rendering/re-simulating. The fixed scene geometry (divider positions/height,
floor, tiers) is NOT stored per-instance (it's a module-level constant,
scaled by whatever `size` the caller uses) — see build_scene.

Usage:
    python datasets/ball_drop_generation.py --num-instances 20000 \
        --output data/ball_drop_data/train.ndjson --seed-offset 0
    python datasets/ball_drop_generation.py --num-instances 2000 \
        --output data/ball_drop_data/val.ndjson --seed-offset 1000000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pymunk
from tqdm import tqdm

SIZE = 128  # canonical physics/pixel unit scale; see module docstring.

# Fixed scene layout — reverse-engineered from the real dataset's
# metadata_json (see module docstring): divider x exactly at quarters,
# divider_height ~15 (60/4 from their 512-scale), no pegs, thin floor.
FLOOR_Y_FRAC = 0.5 / SIZE
DIVIDER_X_FRAC = [0.25, 0.5, 0.75]
DIVIDER_HEIGHT_FRAC = 15.0 / SIZE
LINE_TIER_FRAC = [0.25, 0.45, 0.65]

MAX_STEPS = 3000
DT = 1.0 / 60
SETTLE_STEPS = 20
SETTLE_VEL = 2.0
SETTLE_Y_MAX_FRAC = 30.0 / SIZE
DIVIDER_MARGIN_FRAC = 2.5 / SIZE  # reject if final ball x is this close to a divider (ambiguous bucket).

MIN_LINES = 1
MAX_LINES = 3
LINE_THICKNESS_FRAC = 1.0 / SIZE
LINE_LENGTH_FRAC_RANGE = (0.25, 0.8)  # of `size` — real lines span large fractions of the width.
LINE_ANGLE_RANGE = (-0.6, 0.6)  # radians — near-horizontal ramps, matches observed real angles.
LINE_TIER_Y_JITTER_FRAC = 3.0 / SIZE
LINE_X_MARGIN_FRAC = 20.0 / SIZE
BALL_START_X_MARGIN_FRAC = 10.0 / SIZE
BALL_RADIUS_FRAC = 3.0 / SIZE
BALL_START_Y_FRAC = 1.0 - 6.0 / SIZE


def _make_static_segment(space: pymunk.Space, p1, p2, radius: float, elasticity: float = 0.3, friction: float = 0.5):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, p1, p2, radius)
    shape.elasticity = elasticity
    shape.friction = friction
    space.add(body, shape)


def bucket_bounds_for(size: int) -> list[float]:
    return [0.0] + [f * size for f in DIVIDER_X_FRAC] + [float(size)]


def build_scene(size: int = SIZE) -> pymunk.Space:
    """Build the fixed scene: outer walls, floor, and the 3 bucket dividers.
    Deterministic given `size` — no per-instance randomization (matches the
    real dataset's own fixed layout, per project decision)."""
    space = pymunk.Space()
    space.gravity = (0, -900)

    _make_static_segment(space, (0, 0), (0, size), 1.5)
    _make_static_segment(space, (size, 0), (size, size), 1.5)

    floor_y = FLOOR_Y_FRAC * size
    _make_static_segment(space, (0, floor_y), (size, floor_y), 1.5, elasticity=0.2, friction=0.6)

    divider_height = DIVIDER_HEIGHT_FRAC * size
    for xf in DIVIDER_X_FRAC:
        dx = xf * size
        _make_static_segment(space, (dx, floor_y), (dx, floor_y + divider_height), 1.2, elasticity=0.2, friction=0.6)

    return space


def add_ball(space: pymunk.Space, start_x: float, size: int = SIZE) -> pymunk.Body:
    start_y = BALL_START_Y_FRAC * size
    radius = BALL_RADIUS_FRAC * size
    mass = 1.0
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = (start_x, start_y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.4
    shape.friction = 0.5
    space.add(body, shape)
    return body


def add_lines(space: pymunk.Space, lines: list[tuple[tuple[float, float], tuple[float, float]]], size: int = SIZE) -> None:
    """Add solution/candidate line segments as static bodies. `lines` are
    (p1, p2) pairs in physics units (not normalized)."""
    radius = LINE_THICKNESS_FRAC * size
    for p1, p2 in lines:
        _make_static_segment(space, p1, p2, radius, elasticity=0.3, friction=0.5)


def sample_lines(rng: random.Random, n_lines: int, size: int = SIZE) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Sample `n_lines` ramps, one per randomly chosen distinct height tier
    out of the 3 fixed LINE_TIER_FRAC levels (matches the real dataset's
    apparent one-line-per-tier convention)."""
    tiers = rng.sample(LINE_TIER_FRAC, n_lines)
    x_margin = LINE_X_MARGIN_FRAC * size
    lines = []
    for tier_f in tiers:
        cx = rng.uniform(x_margin, size - x_margin)
        cy = tier_f * size + rng.uniform(-LINE_TIER_Y_JITTER_FRAC * size, LINE_TIER_Y_JITTER_FRAC * size)
        angle = rng.uniform(*LINE_ANGLE_RANGE)
        length = rng.uniform(*LINE_LENGTH_FRAC_RANGE) * size
        dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
        lines.append(((cx - dx, cy - dy), (cx + dx, cy + dy)))
    return lines


def bucket_of(x: float, bucket_bounds: list[float]) -> int:
    for i in range(4):
        if bucket_bounds[i] <= x < bucket_bounds[i + 1]:
            return i
    return 3 if x >= bucket_bounds[-1] else 0


def simulate(
    space: pymunk.Space,
    ball: pymunk.Body,
    size: int = SIZE,
    max_steps: int = MAX_STEPS,
    dt: float = DT,
    settle_steps: int = SETTLE_STEPS,
    settle_vel: float = SETTLE_VEL,
) -> tuple[int, pymunk.Vec2d, bool]:
    """Step the simulation until the ball is at rest near the bottom (its
    speed stays below `settle_vel` for `settle_steps` consecutive steps while
    y < SETTLE_Y_MAX_FRAC * size), or `max_steps` is reached without settling.

    Returns (step_settled_at, final_position, did_settle).
    """
    settle_y_max = SETTLE_Y_MAX_FRAC * size
    low_speed_run = 0
    for step in range(max_steps):
        space.step(dt)
        speed = (ball.velocity.x ** 2 + ball.velocity.y ** 2) ** 0.5
        if speed < settle_vel and ball.position.y < settle_y_max:
            low_speed_run += 1
            if low_speed_run >= settle_steps:
                return step, ball.position, True
        else:
            low_speed_run = 0
    return max_steps, ball.position, False


def generate_instance(seed: int, size: int = SIZE, max_resample: int = 40) -> Optional[dict]:
    """Generate one Ball Drop instance: fixed scene + random ball start +
    random 1-3 solution lines (one per randomly chosen height tier),
    simulated to find out which bucket the ball actually lands in. No search
    for a *specific* target bucket — whichever bucket the random lines
    happen to route the ball into simply becomes the ground truth for this
    (ball_start, lines) pair, which is always a mutually consistent, valid
    instance by construction (this is a reachability task, not an
    optimization one — see this module's docstring).

    Retries with a fresh sub-seed (up to max_resample times, ~44% yield per
    attempt at this task's default parameters) if the ball doesn't settle
    cleanly in a bucket within the step budget, or settles ambiguously close
    to a divider boundary.
    """
    bucket_bounds = bucket_bounds_for(size)
    for attempt in range(max_resample):
        rng = random.Random(f"{seed}_{attempt}")
        space = build_scene(size)
        n_lines = rng.randint(MIN_LINES, MAX_LINES)
        lines = sample_lines(rng, n_lines, size)
        add_lines(space, lines, size)
        margin = BALL_START_X_MARGIN_FRAC * size
        ball_start_x = rng.uniform(margin, size - margin)
        ball = add_ball(space, ball_start_x, size)

        step, final_pos, settled = simulate(space, ball, size)
        if not settled:
            continue
        divider_margin = DIVIDER_MARGIN_FRAC * size
        if any(abs(final_pos.x - b) < divider_margin for b in bucket_bounds[1:-1]):
            continue  # ambiguous — too close to a divider

        bucket = bucket_of(final_pos.x, bucket_bounds)
        return {
            "instance_id": seed,
            "size": size,
            "ball_start_x": ball_start_x / size,
            "num_lines": n_lines,
            "lines": [[p1[0] / size, p1[1] / size, p2[0] / size, p2[1] / size] for p1, p2 in lines],
            "target_bucket": bucket,
            "settle_step": step,
            "final_ball_x": final_pos.x / size,
            "final_ball_y": final_pos.y / size,
        }
    return None


def _worker(seed: int) -> dict:
    inst = generate_instance(seed)
    if inst is None:
        return {"status": "failed", "id": seed, "error": "did not settle after max_resample attempts"}
    inst["status"] = "ok"
    return inst


def main():
    p = argparse.ArgumentParser(description="Generate Ball Drop instances (offline, one-time)")
    p.add_argument("--num-instances", type=int, default=20000)
    p.add_argument("--output", type=str, required=True, help="output .ndjson path")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0, help="added to instance_id for the per-instance RNG seed")
    args = p.parse_args()

    num_workers = args.num_workers or mp.cpu_count()
    seeds = list(range(args.seed_offset, args.seed_offset + args.num_instances))

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_worker, s): s for s in seeds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating ball-drop instances"):
            r = fut.result()
            if r.get("status") == "failed":
                failures.append(r)
            else:
                results.append(r)

    results.sort(key=lambda r: r["instance_id"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            r = {k: v for k, v in r.items() if k != "status"}
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)}/{args.num_instances} instances to {out_path}")
    if failures:
        print(f"{len(failures)} instances failed, e.g.: {failures[:3]}")


if __name__ == "__main__":
    main()
