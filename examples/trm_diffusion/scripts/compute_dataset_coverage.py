#!/usr/bin/env python3
"""Reproduce the dataset-coverage tables from the report (# Dataset section).

S (training samples per size) is recomputed from gen_amaze.py so it stays exact.
M is exact for maze (spanning trees of the n×n grid) and an estimate for queens
(layouts × region colourings with a unique solution — no closed form).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_amaze import (  # noqa: E402  authoritative dataset constants + split logic
    MAZE_GEOMETRIES,
    MAZE_SCALES,
    MAZE_TRAIN,
    QUEEN_SCALES,
    QUEEN_TRAIN,
    _queen_mixed_counts,
    _split_count,
)

# Queens distinct-board space — ESTIMATE only (no closed form); report figures for n=4,5,6.
QUEENS_M_ESTIMATE: dict[int, float | None] = {4: 3_000, 5: 200_000, 6: 1.3e7}


def maze_spanning_trees(n: int) -> float:
    # Spanning trees of the n×n grid via Laplacian eigenvalues μ_{i,j}=λ_i+λ_j, λ_k=4·sin²(kπ/2n);
    # T=(1/n²)·Π_{(i,j)≠(0,0)} μ_{i,j}. Verified n=3→192, n=5→557 568 000. float64 holds up to n=16.
    lam = [4.0 * math.sin(k * math.pi / (2 * n)) ** 2 for k in range(n)]
    log_t = -2.0 * math.log10(n)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            log_t += math.log10(lam[i] + lam[j])
    return 10.0**log_t


def _fmt_count(x: float | None) -> str:
    if x is None:
        return ">>"
    if x < 1e15:
        return f"{int(round(x)):,}"
    return f"{x:.1e}"


def _fmt_cov(s: int, m: float | None) -> str:
    if m is None or m == 0:
        return "<<"
    frac = s / m
    return f"{frac * 100:.2g}%" if frac >= 1e-4 else f"~1e{math.floor(math.log10(frac)):d}"


def _table(title: str, rows: list[tuple[int, float | None, int]]) -> None:
    print(f"\n## {title}")
    print(f"| {'n':>3} | {'distinct boards M':>18} | {'training S':>10} | {'S/M coverage':>12} |")
    print(f"| {'-'*3} | {'-'*18} | {'-'*10} | {'-'*12} |")
    for n, m, s in rows:
        print(f"| {n:>3} | {_fmt_count(m):>18} | {s:>10,} | {_fmt_cov(s, m):>12} |")


def queens_rows() -> list[tuple[int, float | None, int]]:
    s_per_scale = _queen_mixed_counts(QUEEN_TRAIN)
    return [(n, QUEENS_M_ESTIMATE.get(n), s_per_scale.get(n, 0)) for n in QUEEN_SCALES]


def _maze_per_combo() -> dict[tuple[str, int], int]:
    combos = [(g, s) for g in MAZE_GEOMETRIES for s in MAZE_SCALES]
    return dict(zip(combos, _split_count(MAZE_TRAIN, len(combos))))


def maze_rows() -> list[tuple[int, float | None, int]]:
    per_combo = _maze_per_combo()
    # representative per-size S (shapes differ by ≤1 for the even 30k split).
    return [(n, maze_spanning_trees(n), per_combo[("square", n)]) for n in MAZE_SCALES]


def main() -> None:
    q = queens_rows()
    m = maze_rows()
    print(f"MAZE_TRAIN={MAZE_TRAIN:,}  QUEEN_TRAIN={QUEEN_TRAIN:,}")
    _table("Queens (M is an ESTIMATE — see docstring)", q)
    print(f"  sum S (queens) = {sum(s for _, _, s in q):,}  (should be {QUEEN_TRAIN:,})")
    _table("Maze — S is per (shape, size); M = spanning trees of the n×n grid", m)
    print(
        f"  sum S (maze) = {sum(_maze_per_combo().values()):,}"
        f"  (over all {len(MAZE_GEOMETRIES)}×{len(MAZE_SCALES)} shape×size combos;"
        f" should be {MAZE_TRAIN:,})"
    )


if __name__ == "__main__":
    main()
