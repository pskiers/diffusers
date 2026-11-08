"""
ABOUTME: Per-board cell structure for AMAZE Maze / Queens (rebuttal exp 08): cell index, the thinker token at each
ABOUTME: cell's centre, per-cell labels read from an image by the AMAZE scorer, and a solution-free violation count.

Cells and labels come from the scorer's own machinery (eval/amaze_eval.py AmazeMetrics): Maze cells are the non-zero
ids of the board's cell map (0 = wall / background) and a cell is "on the path" when the scorer's blue-path mask
covers it (`_cells_covered`); Queens cells are the n x n grid and a cell holds a queen when `_detect_queen_cells`
finds one there. No endpoint tolerance is applied to the labels (the scorer drops the start/goal neighbourhood only
when comparing against the ground truth).

Violations (no ground truth used, the analogue of the Sudoku unit check):
- Maze: two cells are joined by an open passage when their boundary in the cell map is long enough and mostly free of
  wall (dark) pixels in the unsolved maze image (`maze_passages`; the cell map alone does not mark walls). A predicted path S must be a simple start-goal path through passages: every cell of S has
  exactly 2 neighbours in S (start and goal: 1), and start and goal are in S. Violations = cells of S whose degree in
  S is wrong + missing endpoints. The ground-truth path of a perfect maze scores 0 (checked over all boards).
- Queens: rows, columns and regions with a queen count other than 1, plus pairs of touching queens (8-neighbourhood).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

TOKEN_GRID = 12  # the AMAZE thinker's token grid over the 144 px image (12 px per token)


@dataclass
class Board:
    task: str
    n: int                      # puzzle size (maze n / queens n)
    n_cells: int
    token_idx: torch.Tensor     # (N,) thinker token holding each cell's centre
    adj: torch.Tensor           # (N, N) 0/1: maze open passages / queens touching cells
    gt: torch.Tensor            # (N,) 0/1 ground truth (maze path cells / queen cells)
    endpoints: tuple = ()       # maze: (start, goal) cell indices
    groups: torch.Tensor | None = None  # queens: (N, n_rows + n_cols + n_regions) one-hot unit membership
    cell_ids: np.ndarray | None = None  # maze: (H, W) cell-id image at native resolution
    id_to_idx: dict | None = None
    size: tuple = ()            # (W, H) native resolution
    queen: dict | None = None   # queens: cell_size, queen_radius


PASSAGE_DARK_MAX = 0.5     # a boundary at most this dark (wall pixels) is open
PASSAGE_MIN_CONTACT = 0.2  # ...and at least this many cell widths long (circle cells also touch at corners)


def maze_passages(cell_ids: np.ndarray, maze_rgb: np.ndarray, id_to_idx: dict, min_contact: float) -> np.ndarray:
    """(N, N) 0/1 open passages: cells whose pixels touch (4-neighbourhood) along a boundary of at least min_contact
    pixel contacts of which at most PASSAGE_DARK_MAX are wall (dark in the unsolved maze image). The cell map assigns
    wall pixels to cells, so touching alone does not mean open."""
    dark = maze_rgb.max(axis=2) < 100
    lut = np.full(int(cell_ids.max()) + 1, -1)
    for k, v in id_to_idx.items():
        lut[k] = v
    idx = lut[cell_ids]
    n = len(id_to_idx)
    contacts = np.zeros((n, n))
    dark_contacts = np.zeros((n, n))
    for a, b, da, db in ((idx[:, :-1], idx[:, 1:], dark[:, :-1], dark[:, 1:]),
                         (idx[:-1], idx[1:], dark[:-1], dark[1:])):
        m = (a != b) & (a >= 0) & (b >= 0)
        lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        np.add.at(contacts, (lo, hi), 1)
        np.add.at(dark_contacts, (lo, hi), (da[m] | db[m]).astype(float))
    is_open = (contacts >= min_contact) & (dark_contacts <= PASSAGE_DARK_MAX * np.maximum(contacts, 1))
    is_open = is_open | is_open.T
    return is_open.astype(np.float32)


def _token_of(cx: np.ndarray, cy: np.ndarray, w: int, h: int) -> np.ndarray:
    tx = np.clip((cx / w * TOKEN_GRID).astype(int), 0, TOKEN_GRID - 1)
    ty = np.clip((cy / h * TOKEN_GRID).astype(int), 0, TOKEN_GRID - 1)
    return ty * TOKEN_GRID + tx


def build_board(metadata: dict, task: str, scorer) -> Board:
    """metadata: an AmazeDataset sample's metadata (include_metadata=True)."""
    if task == "maze":
        meta = metadata["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        path_ids = [int(c) for c in (json.loads(meta["path_cell_ids"]) if isinstance(meta["path_cell_ids"], str)
                                     else meta["path_cell_ids"])]
        cell_ids = scorer.decode_cell_map_ids(metadata["cell_map"])
        h, w = cell_ids.shape
        ids, cx, cy, area = scorer._cell_centroids(cell_ids)
        keep = (area > 0) & (ids != 0)
        ids, cx, cy = ids[keep], cx[keep], cy[keep]
        id_to_idx = {int(i): k for k, i in enumerate(ids)}
        n_cells = len(ids)
        adj = maze_passages(cell_ids, scorer._to_pixel_array(metadata["original_img"], (w, h)), id_to_idx,
                            min_contact=PASSAGE_MIN_CONTACT * float(np.sqrt(np.median(area[keep]))))
        gt = np.zeros(n_cells, dtype=np.float32)
        gt[[id_to_idx[c] for c in path_ids]] = 1.0
        maze_cfg = meta.get("maze_config", {})
        if isinstance(maze_cfg, str):
            import ast
            maze_cfg = ast.literal_eval(maze_cfg)
        adj = torch.from_numpy(adj)
        return Board(task="maze", n=int(maze_cfg.get("width", 0)), n_cells=n_cells,
                     token_idx=torch.from_numpy(_token_of(cx, cy, w, h)), adj=adj,
                     gt=torch.from_numpy(gt), endpoints=(id_to_idx[path_ids[0]], id_to_idx[path_ids[-1]]),
                     cell_ids=cell_ids, id_to_idx=id_to_idx, size=(w, h))
    raw = metadata["sample_json"]
    meta = json.loads(raw) if isinstance(raw, str) else raw
    n, cs = int(meta["n"]), int(meta["cell_size"])
    w, h = int(meta["width"]), int(meta["height"])
    regions = np.array(json.loads(meta["regions"]) if isinstance(meta["regions"], str) else meta["regions"])
    queens = json.loads(meta["queens"]) if isinstance(meta["queens"], str) else meta["queens"]
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    token_idx = _token_of((cc + 0.5) * cs, (rr + 0.5) * cs, w, h)
    touching = (np.abs(rr[:, None] - rr[None]) <= 1) & (np.abs(cc[:, None] - cc[None]) <= 1)
    np.fill_diagonal(touching, False)
    n_reg = int(regions.max()) + 1
    groups = np.zeros((n * n, 2 * n + n_reg), dtype=np.float32)
    groups[np.arange(n * n), rr] = 1
    groups[np.arange(n * n), n + cc] = 1
    groups[np.arange(n * n), 2 * n + regions.ravel()] = 1
    gt = np.zeros(n * n, dtype=np.float32)
    gt[[int(r) * n + int(c) for r, c in queens]] = 1.0
    return Board(task="queens", n=n, n_cells=n * n, token_idx=torch.from_numpy(token_idx),
                 adj=torch.from_numpy(touching.astype(np.float32)), gt=torch.from_numpy(gt),
                 groups=torch.from_numpy(groups), size=(w, h),
                 cell_ids=scorer.decode_cell_map_ids(metadata["cell_map"], (w, h)),
                 queen={"cell_size": cs, "queen_radius": float(meta.get("queen_radius") or cs * 0.25)})


def violations(S: torch.Tensor, board: Board) -> torch.Tensor:
    """S: (P, N) 0/1 predicted cells for P positions -> (P,) violation counts (see module docstring)."""
    S = S.float()
    adj = board.adj.to(S.device)
    if board.task == "maze":
        deg = S @ adj
        req = torch.full((board.n_cells,), 2.0, device=S.device)
        req[list(board.endpoints)] = 1.0
        bad = (S * (deg != req).float()).sum(1)
        return bad + (1 - S[:, board.endpoints[0]]) + (1 - S[:, board.endpoints[1]])
    counts = S @ board.groups.to(S.device)
    touching = ((S @ adj) * S).sum(1) / 2
    return (counts != 1).float().sum(1) + touching


def labels_from_image(img: torch.Tensor, board: Board, scorer) -> torch.Tensor:
    """img: (C, H, W) in [0, 1] -> (N,) 0/1 per-cell reading by the AMAZE scorer (no endpoint tolerance)."""
    from eval.amaze_eval import extract_blue_path

    gen = scorer._to_pixel_array(img, board.size)
    lab = torch.zeros(board.n_cells)
    if board.task == "maze":
        covered = scorer._cells_covered(scorer._morph_open(extract_blue_path(gen)), board.cell_ids)
        idx = [board.id_to_idx[c] for c in covered if c in board.id_to_idx]
    else:
        cells = scorer._detect_queen_cells(gen, board.cell_ids, board.n, board.queen["cell_size"],
                                           board.queen["queen_radius"])
        idx = [r * board.n + c for r, c in cells]
    lab[idx] = 1.0
    return lab
