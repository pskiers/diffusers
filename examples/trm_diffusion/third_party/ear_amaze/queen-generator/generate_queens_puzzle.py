#!/usr/bin/env python3
"""
Batch generator for Queens Puzzle levels.

Rules:
1) Place N queens on an N x N grid.
2) Exactly one queen per row, per column, per region.
3) No two queens can touch in 8-neighborhood (king adjacency).

Generation strategy:
- First place queens satisfying row/col/adjacency constraints.
- Then grow N connected regions from queen seed cells until all cells are covered.
- Finally, run a backtracking solver to ensure the puzzle has a unique solution.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

Pos = Tuple[int, int]


@dataclass
class Level:
    n: int
    queens: List[Pos]
    regions: List[List[int]]


def neighbors4(r: int, c: int, n: int) -> List[Pos]:
    out: List[Pos] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def touching_any_queen(r: int, c: int, queens_set: Set[Pos]) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if (r + dr, c + dc) in queens_set:
                return True
    return False


def random_queen_layout(n: int, rng: random.Random, max_tries: int = 3000) -> Optional[List[Pos]]:
    """Randomized backtracking: one queen per row with col + adjacency constraints."""
    for _ in range(max_tries):
        cols_used: Set[int] = set()
        queens: List[Pos] = []
        queens_set: Set[Pos] = set()

        def bt(row: int) -> bool:
            if row == n:
                return True
            col_order = list(range(n))
            rng.shuffle(col_order)
            for col in col_order:
                if col in cols_used:
                    continue
                if touching_any_queen(row, col, queens_set):
                    continue
                cols_used.add(col)
                queens.append((row, col))
                queens_set.add((row, col))

                if bt(row + 1):
                    return True

                cols_used.remove(col)
                queens.pop()
                queens_set.remove((row, col))
            return False

        if bt(0):
            return queens
    return None


def grow_regions_from_seeds(n: int, queens: Sequence[Pos], rng: random.Random) -> Optional[List[List[int]]]:
    """Random flood-fill growth from N queen seeds, creating N connected regions."""
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    frontiers: List[Set[Pos]] = [set() for _ in range(n)]
    sizes = [0 for _ in range(n)]

    for rid, (r, c) in enumerate(queens):
        grid[r][c] = rid
        frontiers[rid].add((r, c))
        sizes[rid] = 1

    unassigned = n * n - n

    for _ in range(n * n * 20):
        if unassigned == 0:
            return grid

        expandable: List[int] = []
        for rid in range(n):
            found = False
            for r, c in frontiers[rid]:
                for nr, nc in neighbors4(r, c, n):
                    if grid[nr][nc] == -1:
                        found = True
                        break
                if found:
                    break
            if found:
                expandable.append(rid)

        if not expandable:
            break

        # Bias towards smaller regions for better balance.
        min_size = min(sizes[rid] for rid in expandable)
        small = [rid for rid in expandable if sizes[rid] <= min_size + 1]
        rid = rng.choice(small if rng.random() < 0.7 else expandable)

        boundary_cells = list(frontiers[rid])
        rng.shuffle(boundary_cells)

        picked: Optional[Pos] = None
        for r, c in boundary_cells:
            cands = [(nr, nc) for nr, nc in neighbors4(r, c, n) if grid[nr][nc] == -1]
            if cands:
                picked = rng.choice(cands)
                break

        if picked is None:
            continue

        pr, pc = picked
        grid[pr][pc] = rid
        frontiers[rid].add((pr, pc))

        # Keep frontier sets compact: if a cell has no unassigned neighbor, drop it.
        for rr, cc in [(pr, pc)] + neighbors4(pr, pc, n):
            owner = grid[rr][cc]
            if owner == -1:
                continue
            if (rr, cc) in frontiers[owner]:
                if all(grid[nr][nc] != -1 for nr, nc in neighbors4(rr, cc, n)):
                    frontiers[owner].discard((rr, cc))
            else:
                if any(grid[nr][nc] == -1 for nr, nc in neighbors4(rr, cc, n)):
                    frontiers[owner].add((rr, cc))

        sizes[rid] += 1
        unassigned -= 1

    return None


def solve_solutions(level_regions: List[List[int]], max_count: int = 2) -> List[Tuple[int, ...]]:
    """Backtracking solver for the puzzle; returns up to max_count solutions."""
    n = len(level_regions)
    cols_used: Set[int] = set()
    regions_used: Set[int] = set()
    queens_set: Set[Pos] = set()
    cols_solution = [-1 for _ in range(n)]
    sols: List[Tuple[int, ...]] = []

    def bt(row: int) -> None:
        if len(sols) >= max_count:
            return
        if row == n:
            sols.append(tuple(cols_solution))
            return

        for col in range(n):
            rid = level_regions[row][col]
            if col in cols_used or rid in regions_used:
                continue
            if touching_any_queen(row, col, queens_set):
                continue

            cols_used.add(col)
            regions_used.add(rid)
            queens_set.add((row, col))
            cols_solution[row] = col

            bt(row + 1)

            cols_solution[row] = -1
            cols_used.remove(col)
            regions_used.remove(rid)
            queens_set.remove((row, col))

    bt(0)
    return sols


def count_solutions(level_regions: List[List[int]], max_count: int = 2) -> int:
    return len(solve_solutions(level_regions, max_count=max_count))


def is_region_connected(
    regions: List[List[int]],
    rid: int,
    seed: Pos,
    skip_cell: Optional[Pos] = None,
) -> bool:
    n = len(regions)
    sr, sc = seed
    if skip_cell == seed:
        return False
    if regions[sr][sc] != rid:
        return False

    total = 0
    for r in range(n):
        for c in range(n):
            if (r, c) == skip_cell:
                continue
            if regions[r][c] == rid:
                total += 1
    if total == 0:
        return False

    stack = [(sr, sc)]
    seen = {(sr, sc)}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c, n):
            if (nr, nc) == skip_cell:
                continue
            if regions[nr][nc] != rid or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))

    return len(seen) == total


def can_move_cell_region(
    regions: List[List[int]],
    cell: Pos,
    dst_rid: int,
    queen_cells: Set[Pos],
    seed_by_rid: Sequence[Pos],
) -> bool:
    r, c = cell
    n = len(regions)
    src_rid = regions[r][c]
    if src_rid == dst_rid:
        return False
    if cell in queen_cells:
        return False
    if not any(regions[nr][nc] == dst_rid for nr, nc in neighbors4(r, c, n)):
        return False
    if not is_region_connected(regions, src_rid, seed_by_rid[src_rid], skip_cell=cell):
        return False
    if not is_region_connected(regions, dst_rid, seed_by_rid[dst_rid], skip_cell=None):
        return False
    return True


def apply_move(regions: List[List[int]], cell: Pos, dst_rid: int) -> None:
    r, c = cell
    regions[r][c] = dst_rid


def refine_regions_to_unique(
    regions: List[List[int]],
    queens: Sequence[Pos],
    rng: random.Random,
    max_iters: int = 4000,
) -> Optional[List[List[int]]]:
    n = len(regions)
    queen_cells = set(queens)
    gt_cols = tuple(c for _, c in queens)
    seed_by_rid = list(queens)

    sols = solve_solutions(regions, max_count=2)
    if len(sols) == 1:
        return regions

    for _ in range(max_iters):
        sols = solve_solutions(regions, max_count=2)
        if len(sols) == 1:
            return regions

        alt = sols[0] if sols[0] != gt_cols else sols[1]
        moved = False

        # Targeted move: try to invalidate this alternative solution.
        rows = [r for r in range(n) if alt[r] != gt_cols[r]]
        rng.shuffle(rows)
        for r in rows:
            ac = alt[r]
            gc = gt_cols[r]
            cell = (r, ac)
            dst = regions[r][gc]
            if can_move_cell_region(regions, cell, dst, queen_cells, seed_by_rid):
                src = regions[r][ac]
                apply_move(regions, cell, dst)
                if count_solutions(regions, max_count=2) >= 1:
                    moved = True
                    break
                regions[r][ac] = src

        if moved:
            continue

        # Fallback move: random valid boundary transfer to escape local minima.
        candidates: List[Tuple[Pos, int]] = []
        for r in range(n):
            for c in range(n):
                if (r, c) in queen_cells:
                    continue
                src = regions[r][c]
                neigh_rids = {regions[nr][nc] for nr, nc in neighbors4(r, c, n)}
                for dst in neigh_rids:
                    if dst != src:
                        candidates.append(((r, c), dst))
        rng.shuffle(candidates)

        for cell, dst in candidates[: min(len(candidates), 200)]:
            if can_move_cell_region(regions, cell, dst, queen_cells, seed_by_rid):
                apply_move(regions, cell, dst)
                moved = True
                break

        if not moved:
            return None

    return None


def hsv_palette(k: int, sat: float = 0.38, val: float = 0.95) -> List[Tuple[int, int, int]]:
    colors: List[Tuple[int, int, int]] = []
    for i in range(k):
        h = i / max(1, k)
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw_level(
    level: Level,
    path: Path,
    show_queens: bool,
    cell_size: int = 64,
    margin: int = 0,
    line_width: int = 2,
    queen_radius: Optional[int] = None,
) -> None:
    if not PIL_OK:
        raise RuntimeError("PIL backend not available")

    n = level.n
    w = margin * 2 + n * cell_size
    h = margin * 2 + n * cell_size

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    palette = hsv_palette(n)

    for r in range(n):
        for c in range(n):
            rid = level.regions[r][c]
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=palette[rid])

    # Grid lines
    for i in range(n + 1):
        x = margin + i * cell_size
        y = margin + i * cell_size
        draw.line([x, margin, x, margin + n * cell_size], fill=(45, 45, 45), width=line_width)
        draw.line([margin, y, margin + n * cell_size, y], fill=(45, 45, 45), width=line_width)

    if show_queens:
        radius = int(cell_size * 0.25) if queen_radius is None else int(queen_radius)
        for r, c in level.queens:
            cx = margin + c * cell_size + cell_size // 2
            cy = margin + r * cell_size + cell_size // 2
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(15, 15, 15),
                outline=(250, 250, 250),
                width=max(1, line_width),
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_level_svg(
    level: Level,
    path: Path,
    show_queens: bool,
    cell_size: int = 64,
    margin: int = 0,
    line_width: int = 2,
    queen_radius: Optional[int] = None,
) -> None:
    n = level.n
    width = margin * 2 + n * cell_size
    height = margin * 2 + n * cell_size
    palette = hsv_palette(n)

    def rgb(c: Tuple[int, int, int]) -> str:
        return f"rgb({c[0]},{c[1]},{c[2]})"

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')

    for r in range(n):
        for c in range(n):
            rid = level.regions[r][c]
            x = margin + c * cell_size
            y = margin + r * cell_size
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{rgb(palette[rid])}"/>'
            )

    stroke = "rgb(45,45,45)"
    for i in range(n + 1):
        x = margin + i * cell_size
        y = margin + i * cell_size
        lines.append(
            f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + n * cell_size}" '
            f'stroke="{stroke}" stroke-width="{line_width}"/>'
        )
        lines.append(
            f'<line x1="{margin}" y1="{y}" x2="{margin + n * cell_size}" y2="{y}" '
            f'stroke="{stroke}" stroke-width="{line_width}"/>'
        )

    if show_queens:
        radius = int(cell_size * 0.25) if queen_radius is None else int(queen_radius)
        for r, c in level.queens:
            cx = margin + c * cell_size + cell_size // 2
            cy = margin + r * cell_size + cell_size // 2
            lines.append(
                f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="rgb(15,15,15)" '
                f'stroke="rgb(250,250,250)" stroke-width="{max(1, line_width)}"/>'
            )

    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_unique_level(
    n: int,
    rng: random.Random,
    max_attempts: int,
) -> Level:
    for _ in range(max_attempts):
        queens = random_queen_layout(n, rng)
        if queens is None:
            continue

        regions = grow_regions_from_seeds(n, queens, rng)
        if regions is None:
            continue

        refined = refine_regions_to_unique(regions, queens, rng)
        if refined is not None and count_solutions(refined, max_count=2) == 1:
            return Level(n=n, queens=list(queens), regions=refined)

    raise RuntimeError(
        f"Failed to generate a unique level after {max_attempts} attempts. "
        "Try increasing --max-attempts or changing --seed."
    )


def save_level_json(level: Level, path: Path, cell_size: int, queen_radius: Optional[int]) -> None:
    n = level.n
    payload = {
        "n": n,
        "width": n * cell_size,
        "height": n * cell_size,
        "cell_size": cell_size,
        "queen_radius": int(cell_size * 0.25) if queen_radius is None else int(queen_radius),
        "cell_id_rule": "left_to_right_then_top_to_bottom",
        "cell_id_formula": "cell_id = row * n + col",
        "queens": [[r, c] for r, c in level.queens],
        "regions": level.regions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_cell_map(level: Level, path: Path, cell_size: int) -> None:
    """
    Save per-pixel cell ownership map as PGM (P5).
    Pixel value = cell_id = row * N + col.
    """
    n = level.n
    width = n * cell_size
    height = n * cell_size
    maxval = n * n - 1

    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{width} {height}\n{maxval}\n".encode("ascii")

    if maxval <= 255:
        data = bytearray(width * height)
        idx = 0
        for y in range(height):
            row = y // cell_size
            for x in range(width):
                col = x // cell_size
                data[idx] = row * n + col
                idx += 1
        path.write_bytes(header + bytes(data))
    else:
        data = bytearray(width * height * 2)
        idx = 0
        for y in range(height):
            row = y // cell_size
            for x in range(width):
                col = x // cell_size
                cid = row * n + col
                data[idx] = (cid >> 8) & 0xFF
                data[idx + 1] = cid & 0xFF
                idx += 2
        path.write_bytes(header + bytes(data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate Queens Puzzle levels.")
    parser.add_argument("--n", type=int, default=8, help="Grid size N for N x N")
    parser.add_argument("--count", type=int, default=10, help="Number of levels to generate")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./output_queens"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2000,
        help="Max attempts per level for uniqueness search",
    )
    parser.add_argument("--cell-size", type=int, default=64, help="Cell size in pixels")
    parser.add_argument(
        "--queen-radius",
        type=int,
        default=None,
        help="Queen circle radius in pixels. Default is 25% of cell size.",
    )
    parser.add_argument(
        "--image-format",
        choices=["auto", "png", "svg"],
        default="auto",
        help="Image export format. auto=png when PIL exists else svg.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 4:
        raise SystemExit("N should be >= 4 for meaningful puzzles.")
    if args.count <= 0:
        raise SystemExit("count should be positive.")

    rng = random.Random(args.seed)

    outdir: Path = args.outdir
    puzzle_dir = outdir / "puzzle"
    gt_dir = outdir / "gt"
    json_dir = outdir / "json"
    cell_map_dir = outdir / "cell_map"

    if args.image_format == "auto":
        image_fmt = "png" if PIL_OK else "svg"
    else:
        image_fmt = args.image_format
    if image_fmt == "png" and not PIL_OK:
        raise SystemExit("Pillow not found; cannot export png. Use --image-format svg.")

    for i in range(args.count):
        level = generate_unique_level(args.n, rng, max_attempts=args.max_attempts)

        stem = f"level_{args.n}_{i:04d}"
        if image_fmt == "png":
            draw_level(
                level,
                puzzle_dir / f"{stem}.png",
                show_queens=False,
                cell_size=args.cell_size,
                queen_radius=args.queen_radius,
            )
            draw_level(
                level,
                gt_dir / f"{stem}.png",
                show_queens=True,
                cell_size=args.cell_size,
                queen_radius=args.queen_radius,
            )
        else:
            draw_level_svg(
                level,
                puzzle_dir / f"{stem}.svg",
                show_queens=False,
                cell_size=args.cell_size,
                queen_radius=args.queen_radius,
            )
            draw_level_svg(
                level,
                gt_dir / f"{stem}.svg",
                show_queens=True,
                cell_size=args.cell_size,
                queen_radius=args.queen_radius,
            )
        save_level_json(
            level,
            json_dir / f"{stem}.json",
            cell_size=args.cell_size,
            queen_radius=args.queen_radius,
        )
        save_cell_map(level, cell_map_dir / f"{stem}.pgm", cell_size=args.cell_size)

        print(f"[{i + 1}/{args.count}] generated {stem}")

    print(f"Done. Output written to: {outdir}")


if __name__ == "__main__":
    main()
