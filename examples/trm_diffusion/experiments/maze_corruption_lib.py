"""experiments/maze_corruption_lib.py — pure-CV maze path corruption operators.

Grounded on the AMAZE square-maze data format (verified on a real sample):
  * Raw images are 728x728 RGB; the model works at 144 (AmazeDataset resizes).
  * ``sol_img`` == ``m_original_img`` + a blue solution path drawn on top
    (pixel-aligned; blue absent in m_original). Blue ~ RGB(15,116,187).
  * ``cell_map`` is an RGB-packed (id = R|G<<8|B<<16) *grid tiling*: for an
    n×n square maze every pixel belongs to one of n² grid cells (NO id-0 wall
    pixels — walls are black lines drawn inside m_original, not in cell_map).
  * ``metadata['metadata']`` JSON holds ``path_cell_ids`` ALREADY ORDERED
    start→goal (verified: consecutive cells are grid-adjacent), plus
    ``maze_config.width/height`` and ``start_cell``/``end_cell``.

Because walls live only in m_original (black lines on cell borders), the
grid-adjacency graph is recovered by sampling the shared border between two
grid-adjacent cells (``_is_open``). This is SQUARE-maze specific; triangle /
circle / hexagon geometries are out of scope for ADD/WALL.

All functions are numpy/cv2/PIL only (no torch) so the corruption pipeline is
unit-testable offline without a GPU.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

BLUE_LO = np.array([100, 50, 50])
BLUE_HI = np.array([130, 255, 255])
_WALL_DARK = 100          # a pixel darker than this (max channel) counts as wall
_WALL_FRAC = 0.25         # >this fraction dark along the shared border => wall (closed)


def to_rgb(x) -> np.ndarray:
    """PIL Image / ndarray -> (H, W, 3) uint8 RGB."""
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return a.astype(np.uint8)


def decode_cell_ids(cell_map) -> np.ndarray:
    """RGB-packed cell_map -> (H, W) int64 cell ids (id = R | G<<8 | B<<16)."""
    a = to_rgb(cell_map).astype(np.uint32)
    return (a[..., 0] | (a[..., 1] << 8) | (a[..., 2] << 16)).astype(np.int64)


def blue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of blue path pixels (same HSV band as the scorer)."""
    return cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), BLUE_LO, BLUE_HI) > 0


@dataclass
class MazeSample:
    sol: np.ndarray          # (H, W, 3) uint8 — full solution (path drawn)
    morig: np.ndarray        # (H, W, 3) uint8 — maze, no path (background)
    ids: np.ndarray          # (H, W) int64 — grid cell id per pixel
    bmask: np.ndarray        # (H, W) bool — blue path mask of sol
    path: list               # ordered cell ids start->goal
    id2rc: dict              # cell id -> (row, col)
    rc2id: dict              # (row, col) -> cell id
    cell_px: float           # pixels per grid cell
    blue_color: tuple        # (R, G, B) ints — sampled path color
    stroke: int              # path stroke width in px
    width: int               # grid width (cols)
    height: int              # grid height (rows)


def build_maze_sample(metadata: dict) -> MazeSample | None:
    """Build a MazeSample from an AmazeDataset sample's ``.metadata`` dict.

    Returns None when the sample lacks the fields we need (so the caller can
    skip it) rather than raising.
    """
    if not metadata:
        return None
    raw = metadata.get("metadata")
    meta = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(meta, dict) or "path_cell_ids" not in meta:
        return None
    sol_img = metadata.get("sol_img")
    morig_img = metadata.get("m_original_img")
    cell_map = metadata.get("cell_map")
    if sol_img is None or morig_img is None or cell_map is None:
        return None
    mc = meta.get("maze_config", {})
    width = int(mc.get("width", 0))
    height = int(mc.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    sol = to_rgb(sol_img)
    morig = to_rgb(morig_img)
    ids = decode_cell_ids(cell_map)
    if sol.shape[:2] != ids.shape or morig.shape[:2] != ids.shape:
        return None
    bmask = blue_mask(sol)
    path = [int(x) for x in meta["path_cell_ids"]]

    cell_px = ids.shape[0] / height
    id2rc, rc2id = {}, {}
    for cid in np.unique(ids):
        if cid == 0:
            continue
        ys, xs = np.where(ids == cid)
        r = int(round(ys.mean() / cell_px - 0.5))
        c = int(round(xs.mean() / cell_px - 0.5))
        id2rc[int(cid)] = (r, c)
        rc2id[(r, c)] = int(cid)

    # Drop path cells missing from the grid map (robustness); keep order.
    path = [p for p in path if p in id2rc]
    if len(path) < 3:
        return None

    blue_px = sol[bmask]
    blue_color = tuple(int(v) for v in blue_px.mean(0)) if len(blue_px) else (15, 116, 187)
    plen = _path_len_px(path, id2rc, cell_px)
    stroke = max(3, int(round(len(blue_px) / max(1.0, plen)))) if len(blue_px) else max(3, int(cell_px * 0.1))

    return MazeSample(sol, morig, ids, bmask, path, id2rc, rc2id, cell_px,
                      blue_color, stroke, width, height)


def _center(rc, cell_px) -> tuple:
    r, c = rc
    return (int((c + 0.5) * cell_px), int((r + 0.5) * cell_px))   # (x, y) for cv2


def _path_len_px(path, id2rc, cell_px) -> float:
    tot = 0.0
    for a, b in zip(path[:-1], path[1:]):
        (xa, ya), (xb, yb) = _center(id2rc[a], cell_px), _center(id2rc[b], cell_px)
        tot += ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
    return tot


def _is_open(morig: np.ndarray, r, c, r2, c2, cell_px: float) -> bool:
    """True if there is NO wall between grid-adjacent cells (r,c)-(r2,c2).

    Samples the middle 40% of the shared border in the (path-free) maze image;
    a mostly-black border means a wall (closed), mostly-white means an opening.
    """
    if (r2, c2) == (r, c + 1):
        x = int((c + 1) * cell_px); band = morig[int((r + 0.3) * cell_px):int((r + 0.7) * cell_px), max(0, x - 4):x + 4]
    elif (r2, c2) == (r, c - 1):
        x = int(c * cell_px); band = morig[int((r + 0.3) * cell_px):int((r + 0.7) * cell_px), max(0, x - 4):x + 4]
    elif (r2, c2) == (r + 1, c):
        y = int((r + 1) * cell_px); band = morig[max(0, y - 4):y + 4, int((c + 0.3) * cell_px):int((c + 0.7) * cell_px)]
    elif (r2, c2) == (r - 1, c):
        y = int(r * cell_px); band = morig[max(0, y - 4):y + 4, int((c + 0.3) * cell_px):int((c + 0.7) * cell_px)]
    else:
        return False
    if band.size == 0:
        return False
    return (band.max(axis=2) < _WALL_DARK).mean() < _WALL_FRAC


def _open_nonblocked(ms: MazeSample, cell: int, blocked: set) -> list:
    """Grid-adjacent cells reachable from ``cell`` (no wall) that are not blocked."""
    r, c = ms.id2rc[cell]
    out = []
    for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        nid = ms.rc2id.get(nb)
        if nid is not None and nid not in blocked and _is_open(ms.morig, r, c, nb[0], nb[1], ms.cell_px):
            out.append(nid)
    return out


def render_partial(ms: MazeSample, p: float) -> tuple:
    """Maze background with only the first ``p`` fraction of the true path drawn.

    Returns (img728 uint8, shown_cells ordered list). The path is copied
    pixel-exact from ``sol`` (real stroke/colour) for the shown cells.
    """
    k = max(1, int(round(len(ms.path) * p)))
    shown = ms.path[:k]
    img = ms.morig.copy()
    sel = np.isin(ms.ids, shown) & ms.bmask
    img[sel] = ms.sol[sel]
    return img, shown


def apply_gap(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random,
              gap_frac: float = 0.15, margin: float = 0.25) -> tuple:
    """Erase a contiguous INTERIOR chunk of the shown prefix (bracketed both
    sides). Returns (new_img, erased_cells)."""
    n = len(shown)
    glen = max(1, int(round(n * gap_frac)))
    lo = int(n * margin)
    hi = int(n * (1 - margin)) - glen
    if hi <= lo:
        start = max(1, n // 2 - glen // 2)
    else:
        start = rng.randint(lo, hi)
    erased = shown[start:start + glen]
    if not erased:
        return img, []
    out = img.copy()
    sel = np.isin(ms.ids, erased)
    out[sel] = ms.morig[sel]           # restore background (remove path)
    return out, erased


def apply_add(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random) -> tuple:
    """Add a legal off-path dead-end spur near the frontier. Random-walks
    through open corridors, avoiding the whole GT path and its own trail, until
    stuck. Branches from the nearest prior shown cell that has a side-branch.
    Returns (new_img, added_cells)."""
    blocked = set(ms.path)
    branch = next((cell for cell in reversed(shown) if _open_nonblocked(ms, cell, blocked)), None)
    if branch is None:
        return img, []
    walk = [branch]
    cur = branch
    while True:
        cand = _open_nonblocked(ms, cur, blocked)
        if not cand:
            break
        nxt = rng.choice(cand)
        walk.append(nxt)
        blocked.add(nxt)
        cur = nxt
    added = walk[1:]
    if not added:
        return img, []
    out = img.copy()
    for a, b in zip(walk[:-1], walk[1:]):
        cv2.line(out, _center(ms.id2rc[a], ms.cell_px), _center(ms.id2rc[b], ms.cell_px),
                 ms.blue_color, ms.stroke)
    return out, added


def apply_wall(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random,
               ext: float = 0.30) -> tuple:
    """Draw a straight blue line from the frontier through the nearest wall and a
    few px past its far side. Returns (new_img, wall_line_mask, in_wall_mask)
    where wall_line_mask is the drawn pixels and in_wall_mask is the drawn pixels
    that land on an actual wall (black in m_original)."""
    frontier = shown[-1]
    r, c = ms.id2rc[frontier]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    walled = [(dr, dc) for dr, dc in dirs if not _is_open(ms.morig, r, c, r + dr, c + dc, ms.cell_px)]
    if not walled:
        return img, None, None
    internal = [(dr, dc) for dr, dc in walled if (r + dr, c + dc) in ms.rc2id]
    dr, dc = rng.choice(internal or walled)
    ca = _center((r, c), ms.cell_px)
    cb = (int(ca[0] + dc * ms.cell_px * (0.5 + ext)), int(ca[1] + dr * ms.cell_px * (0.5 + ext)))
    out = img.copy()
    cv2.line(out, ca, cb, ms.blue_color, ms.stroke)
    line_mask = np.zeros(ms.ids.shape, np.uint8)
    cv2.line(line_mask, ca, cb, 1, ms.stroke)
    line_mask = line_mask > 0
    wall_dark = ms.morig.max(axis=2) < _WALL_DARK
    return out, line_mask, (line_mask & wall_dark)


def predicted_cells(img728: np.ndarray, ms: MazeSample) -> set:
    """Blue pixels of a 728-res image -> set of grid cell ids they fall in."""
    bm = blue_mask(img728)
    s = set(int(cid) for cid in ms.ids[bm].tolist())
    s.discard(0)
    return s
