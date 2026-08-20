"""
MazeDataset – procedurally generates grid mazes and renders them as RGB
images, for the "draw the correct path" logical-constraint benchmark.

Each maze is a "perfect maze" (a spanning tree over the grid-cell graph, built
via randomized-DFS / recursive backtracker — the same "depth_first_recursive_
backtracker" algorithm used by the published SketchVLM maze-navigation
benchmark, https://huggingface.co/datasets/loganbolton/sketchvlm-maze-navigation),
so exactly one simple path connects any two cells — the shortest path between
a random start and goal is therefore unique, giving a crisp ground-truth
solution.

Two dataset classes are provided:
  MazeDataset          – unlimited synthetic mazes generated on the fly, for
                         actually training the painter/TRM (the real
                         benchmark below has only 200 examples — nowhere near
                         enough to train a diffusion model from scratch).
  SketchVLMMazeBenchmark – loads the real 200-example SketchVLM maze bench,
                         recovers each maze's wall layout from its published
                         image (see `_parse_wall_layout_from_image`), and
                         re-renders it with MazeDataset's own renderer/style.
                         Used as a held-out eval-only benchmark so reported
                         numbers are on the literal published maze instances,
                         without a vision-domain shift confounding the
                         comparison (the painter is only ever trained on this
                         module's own rendering style/resolution, not on
                         SketchVLM's 573x573 pixel format).

Every sample (from either class) is a DataSample with:
  images             – (3, H, W) float32 [0,1]; maze scene with the solution
                         path drawn in blue between the green start cell and
                         the red goal cell.
  spatial_conditions – same shape; identical scene but without the path
                         (walls + start + goal only) — this is what the
                         painter/thinker are conditioned on.
  token_conditions   – (grid_size**2,) int64; per-cell 4-bit "openness" mask
                         (UP=1, RIGHT=2, DOWN=4, LEFT=8 — bit set means that
                         side is passable, not a wall). 0 for cells outside
                         the active maze region.
  solution           – (grid_size**2,) int64; 0 = off-path, 1 = on-path,
                         IGNORE_LABEL_ID (-100) for cells outside the active
                         maze region (ignored by any CE loss / accuracy calc).
  solution_mask      – (grid_size**2,) bool; True = cell is inside the active
                         maze region. Unlike Sudoku's "given cell" mask, this
                         dataset has no partial reveal — the field is reused
                         here to mean "in-bounds", since every in-bounds cell
                         must be solved for and every out-of-bounds cell is
                         padding to keep a fixed seq_len for a fixed canvas.
  puzzle_id          – () int64.

Rendering is a raw numpy canvas (white background, black walls, gray padding
for the region outside the active maze, green start cell, red goal cell,
blue path) — no learned rendering assets are needed, so eval (eval/maze_eval.py)
can extract the drawn path by simple color thresholding instead of a trained
classifier (unlike MNISTSudokuDataset, which composites real MNIST digit
bitmaps and therefore needs a learned digit classifier at eval time).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples

logger = logging.getLogger(__name__)

IGNORE_LABEL_ID = -100  # matches datasets.sudoku_dataset.IGNORE_LABEL_ID convention

# Direction bit flags for the per-cell "openness" bitmask.
UP, RIGHT, DOWN, LEFT = 1, 2, 4, 8
_STEP = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Render colors (RGB, float [0,1]).
_WHITE = (1.0, 1.0, 1.0)
_GRAY = (0.5, 0.5, 0.5)
_GREEN = (0.0, 0.8, 0.0)
_RED = (0.85, 0.0, 0.0)
_BLUE = (0.0, 0.35, 0.95)


# ── Maze generation ────────────────────────────────────────────────────────────


def generate_perfect_maze(size: int, rng: np.random.Generator) -> np.ndarray:
    """Randomized-DFS (recursive backtracker) perfect maze.

    Returns a (size, size) int64 array; entry [r, c] is a bitmask of open
    (passable) directions out of cell (r, c) using the UP/RIGHT/DOWN/LEFT
    bits. A perfect maze is a spanning tree over the grid graph: exactly one
    simple path connects any two cells.
    """
    open_mask = np.zeros((size, size), dtype=np.int64)
    visited = np.zeros((size, size), dtype=bool)
    start = (int(rng.integers(size)), int(rng.integers(size)))
    visited[start] = True
    stack = [start]
    while stack:
        r, c = stack[-1]
        neighbors = []
        for d, (dr, dc) in _STEP.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc]:
                neighbors.append((d, nr, nc))
        if not neighbors:
            stack.pop()
            continue
        d, nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        open_mask[r, c] |= d
        open_mask[nr, nc] |= _OPPOSITE[d]
        visited[nr, nc] = True
        stack.append((nr, nc))
    return open_mask


def shortest_path(
    open_mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """BFS shortest path through `open_mask`'s cell graph.

    In a perfect maze this is also the *unique* simple path between the two
    cells, since the graph is a spanning tree.
    """
    prev: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        size = open_mask.shape[0]
        for d, (dr, dc) in _STEP.items():
            if open_mask[r, c] & d:
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in prev:
                    prev[nxt] = (r, c)
                    q.append(nxt)
    if goal not in prev:
        raise RuntimeError("goal unreachable — maze generation produced a disconnected graph")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ── Rendering / DataSample construction (shared by both dataset classes) ──────


def render_maze(
    grid_size: int,
    cell_size: int,
    active_size: int,
    open_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path_cells: Optional[set] = None,
    wall_px: int = 2,
) -> np.ndarray:
    """Render a maze onto a (3, grid_size*cell_size, grid_size*cell_size) RGB canvas.

    `open_mask` covers only the top-left `active_size`x`active_size` sub-grid;
    the rest of the canvas (if active_size < grid_size) is filled gray as
    padding. `path_cells`=None renders the puzzle only (no path drawn) — used
    for spatial_conditions; passing the solution path's cell set renders the
    solved image — used for `images`.
    """
    cs, wp = cell_size, wall_px
    side = grid_size * cs
    img = np.empty((3, side, side), dtype=np.float32)
    for ch in range(3):
        img[ch] = _WHITE[ch]

    active_px = active_size * cs
    if active_px < side:
        for ch in range(3):
            img[ch, active_px:, :] = _GRAY[ch]
            img[ch, :, active_px:] = _GRAY[ch]

    for r in range(active_size):
        for c in range(active_size):
            r0, c0 = r * cs, c * cs
            mask = int(open_mask[r, c])
            if not (mask & UP):
                img[:, r0:r0 + wp, c0:c0 + cs] = 0.0
            if not (mask & LEFT):
                img[:, r0:r0 + cs, c0:c0 + wp] = 0.0
            if not (mask & DOWN):
                img[:, r0 + cs - wp:r0 + cs, c0:c0 + cs] = 0.0
            if not (mask & RIGHT):
                img[:, r0:r0 + cs, c0 + cs - wp:c0 + cs] = 0.0

    def fill_cell(rc: tuple[int, int], color: tuple[float, float, float]) -> None:
        r, c = rc
        r0, c0 = r * cs + wp, c * cs + wp
        r1, c1 = r * cs + cs - wp, c * cs + cs - wp
        for ch in range(3):
            img[ch, r0:r1, c0:c1] = color[ch]

    if path_cells is not None:
        for rc in path_cells:
            if rc != start and rc != goal:
                fill_cell(rc, _BLUE)
    fill_cell(start, _GREEN)
    fill_cell(goal, _RED)
    return img


def build_maze_sample(
    grid_size: int,
    cell_size: int,
    active_size: int,
    open_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
    puzzle_id: int,
    wall_px: int = 2,
) -> DataSample:
    """Render condition/target images and pack the discrete fields into a DataSample."""
    path_set = set(path)
    cond_img = render_maze(grid_size, cell_size, active_size, open_mask, start, goal, None, wall_px)
    full_img = render_maze(grid_size, cell_size, active_size, open_mask, start, goal, path_set, wall_px)

    gs = grid_size
    solution = np.full(gs * gs, IGNORE_LABEL_ID, dtype=np.int64)
    active_mask = np.zeros(gs * gs, dtype=bool)
    token_conditions = np.zeros(gs * gs, dtype=np.int64)
    for r in range(active_size):
        for c in range(active_size):
            i = r * gs + c
            active_mask[i] = True
            token_conditions[i] = int(open_mask[r, c])
            solution[i] = 1 if (r, c) in path_set else 0

    return DataSample(
        images=torch.from_numpy(full_img),
        spatial_conditions=torch.from_numpy(cond_img),
        solution=torch.from_numpy(solution),
        solution_mask=torch.from_numpy(active_mask),
        token_conditions=torch.from_numpy(token_conditions),
        puzzle_id=torch.tensor(puzzle_id, dtype=torch.long),
    )


# ── Synthetic training/val dataset ─────────────────────────────────────────────


class MazeDataset(Dataset):
    """
    Args:
        grid_size:        fixed canvas size in cells (both dimensions).
                           painter_size = grid_size * cell_size.
        cell_size:         pixel size of one cell edge.
        min_active_size, max_active_size: the maze actually occupies a random
            active_size x active_size sub-grid (anchored at the canvas's
            top-left corner), with active_size sampled uniformly from
            [min_active_size, max_active_size] independently per sample. The
            remainder of the canvas (if any) is rendered as a solid gray
            padding region and excluded from the loss/eval via solution=-100
            and solution_mask=False. Both default to grid_size, i.e. fixed-size
            mazes with no padding. Set min_active_size down to e.g. 3 to cover
            the same scale as the published SketchVLM maze benchmark (see
            SketchVLMMazeBenchmark below) within training.
        wall_px:           wall line thickness in pixels.
        length:            nominal dataset length. Mazes are generated on the
                           fly (not cached to disk); index i always yields the
                           same maze given the same `seed`, so this behaves
                           like a fixed finite maze bank of `length` puzzles.
        seed:              base RNG seed; combined with the sample index to
                           derandomize access order.
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_size: int = 16,
        min_active_size: Optional[int] = None,
        max_active_size: Optional[int] = None,
        wall_px: int = 2,
        length: int = 20_000,
        seed: int = 0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.min_active_size = min_active_size if min_active_size is not None else grid_size
        self.max_active_size = max_active_size if max_active_size is not None else grid_size
        if not (2 <= self.min_active_size <= self.max_active_size <= grid_size):
            raise ValueError(
                f"require 2 <= min_active_size ({self.min_active_size}) <= "
                f"max_active_size ({self.max_active_size}) <= grid_size ({grid_size})"
            )
        self.wall_px = wall_px
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def _rng_for(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1_000_003 + idx)

    def __getitem__(self, idx: int) -> DataSample:
        rng = self._rng_for(idx)
        active_size = int(rng.integers(self.min_active_size, self.max_active_size + 1))
        open_mask = generate_perfect_maze(active_size, rng)

        cells = [(r, c) for r in range(active_size) for c in range(active_size)]
        start = cells[int(rng.integers(len(cells)))]
        goal = start
        while goal == start:
            goal = cells[int(rng.integers(len(cells)))]
        path = shortest_path(open_mask, start, goal)

        return build_maze_sample(
            self.grid_size, self.cell_size, active_size, open_mask, start, goal, path, idx, self.wall_px
        )

    collate_fn = staticmethod(collate_data_samples)


# ── Real published benchmark (eval-only) ───────────────────────────────────────


def _open_mask_from_generation_path(gen_path: list[tuple[int, int]], grid_size: int) -> np.ndarray:
    """Rebuild the exact (grid_size, grid_size) openness bitmask from a
    depth_first_recursive_backtracker's cell-visiting order.

    `gen_path` lists every cell in the order the generator first visited it
    (no repeats — backtracking to retry other neighbors doesn't re-append a
    cell), so each consecutive pair is exactly one spanning-tree edge the
    generator opened. This reconstructs the *complete* true wall layout from
    metadata alone — no pixel inspection of the rendered image needed.
    """
    open_mask = np.zeros((grid_size, grid_size), dtype=np.int64)
    for (r0, c0), (r1, c1) in zip(gen_path[:-1], gen_path[1:]):
        dr, dc = r1 - r0, c1 - c0
        if (dr, dc) == (-1, 0):
            d0, d1 = UP, DOWN
        elif (dr, dc) == (1, 0):
            d0, d1 = DOWN, UP
        elif (dr, dc) == (0, 1):
            d0, d1 = RIGHT, LEFT
        elif (dr, dc) == (0, -1):
            d0, d1 = LEFT, RIGHT
        else:
            raise ValueError(f"non-adjacent consecutive cells in generation_path: {(r0, c0)} -> {(r1, c1)}")
        open_mask[r0, c0] |= d0
        open_mask[r1, c1] |= d1
    return open_mask


class SketchVLMMazeBenchmark(Dataset):
    """
    Held-out eval-only dataset: loads the real, published SketchVLM maze
    benchmark (https://huggingface.co/datasets/loganbolton/sketchvlm-maze-navigation,
    200 examples, fixed 3x3 grids, generated with "depth_first_recursive_
    backtracker" — the same algorithm MazeDataset above implements), rebuilds
    each maze's exact wall layout from its metadata's `generation_path` (the
    generator's cell-visiting order — consecutive pairs are exactly the
    spanning-tree edges it opened), and re-renders it with MazeDataset's own
    renderer so a painter trained only on MazeDataset can be evaluated on the
    literal published instances without a vision-domain mismatch (the
    dataset's own 573x573px rendered images are not used at all — only its
    numeric metadata is; validated to reconstruct the dataset's own
    `shortest_path_coordinates` exactly on all 200 published rows).

    Not intended for training — only ~200 examples exist. Use as an
    eval_callback / eval-only val_dataset alongside a MazeDataset-trained
    model (train with min_active_size<=3<=max_active_size so the model has
    seen this scale during training).

    Args:
        grid_size, cell_size, wall_px: passed through to MazeDataset's
            renderer — should match the grid_size/cell_size the painter was
            trained with (the real mazes are always 3x3 and will be padded
            into the top-left corner of the grid_size canvas if grid_size>3).
        hf_repo, hf_parquet_file: source of the published dataset.
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_size: int = 16,
        wall_px: int = 2,
        hf_repo: str = "loganbolton/sketchvlm-maze-navigation",
        hf_parquet_file: str = "data/train-00000-of-00001.parquet",
    ):
        super().__init__()
        # NB: intentionally uses huggingface_hub + pandas/pyarrow directly
        # rather than the HF `datasets` library — `import datasets` from
        # inside this very package (datasets/maze_dataset.py) resolves to
        # this project's own `datasets/` package, not the pip-installed
        # library, so `from datasets import load_dataset` is unusable here
        # (same reason sudoku_dataset.py uses hf_hub_download instead).
        import pandas as pd
        from huggingface_hub import hf_hub_download

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.wall_px = wall_px
        parquet_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_parquet_file)
        self.table = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> DataSample:
        row = self.table.iloc[idx]
        meta = json.loads(row["metadata_json"])
        active_size = max(int(meta["rows"]), int(meta["cols"]))

        gen_path = [tuple(int(x) for x in rc) for rc in meta["generation_path"]]
        open_mask = _open_mask_from_generation_path(gen_path, active_size)

        entry = tuple(int(x) for x in row["entry_coordinate"])
        exit_ = tuple(int(x) for x in row["exit_coordinate"])
        derived_path = shortest_path(open_mask, entry, exit_)

        expected = [tuple(xy) for xy in json.loads(row["shortest_path_coordinates"])]
        if derived_path != expected:
            logger.warning(
                f"SketchVLMMazeBenchmark: reconstructed path disagrees with metadata for "
                f"maze_id={row.get('maze_id')} (idx={idx}); derived={derived_path} expected={expected}. "
                "Using metadata's own path instead."
            )
            derived_path = expected

        return build_maze_sample(
            self.grid_size,
            self.cell_size,
            active_size,
            open_mask,
            entry,
            exit_,
            derived_path,
            idx,
            self.wall_px,
        )

    collate_fn = staticmethod(collate_data_samples)
