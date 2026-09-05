"""experiments/maze_corruption_lib.py — pure-CV maze path corruption operators."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

BLUE_LO = np.array([100, 50, 50])
BLUE_HI = np.array([130, 255, 255])
_WALL_DARK = 100          # a pixel darker than this counts as wall
_WALL_FRAC = 0.25         # >this fraction dark along the border => wall


def to_rgb(x) -> np.ndarray:
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return a.astype(np.uint8)


def decode_cell_ids(cell_map) -> np.ndarray:
    a = to_rgb(cell_map).astype(np.uint32)
    return (a[..., 0] | (a[..., 1] << 8) | (a[..., 2] << 16)).astype(np.int64)


def blue_mask(rgb: np.ndarray, morph_open: bool = False) -> np.ndarray:
    """HSV blue-path mask.

    ``morph_open=True`` adds the 3x3 binary opening that eval/amaze_eval.py's
    AmazeMetrics applies before mapping pixels to cell ids. Use it wherever the
    probe derives a *predicted cell set*, so recovery_rate and the paper metrics
    (coverage / violation / pass) are read off the same mask. The raw mask is
    kept for building corrupted inputs from the GT solution image, where the
    opening would only eat into the drawn stroke.
    """
    m = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), BLUE_LO, BLUE_HI) > 0
    if morph_open:
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    return m


@dataclass
class MazeSample:
    sol: np.ndarray
    morig: np.ndarray
    ids: np.ndarray
    bmask: np.ndarray
    path: list
    id2rc: dict
    rc2id: dict
    cell_px: float
    blue_color: tuple
    stroke: int
    width: int
    height: int


def build_maze_sample(metadata: dict) -> MazeSample | None:
    if not metadata:
        return None
    raw = metadata.get("metadata")
    meta = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(meta, dict) or "path_cell_ids" not in meta:
        return None
    sol_img, morig_img, cell_map = metadata.get("sol_img"), metadata.get("m_original_img"), metadata.get("cell_map")
    if sol_img is None or morig_img is None or cell_map is None:
        return None
    mc = meta.get("maze_config", {})
    width, height = int(mc.get("width", 0)), int(mc.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    sol, morig, ids = to_rgb(sol_img), to_rgb(morig_img), decode_cell_ids(cell_map)
    if sol.shape[:2] != ids.shape or morig.shape[:2] != ids.shape:
        return None
    bmask, path = blue_mask(sol), [int(x) for x in meta["path_cell_ids"]]

    cell_px = ids.shape[0] / height
    id2rc, rc2id = {}, {}
    for cid in np.unique(ids):
        if cid == 0: continue
        ys, xs = np.where(ids == cid)
        r, c = int(round(ys.mean() / cell_px - 0.5)), int(round(xs.mean() / cell_px - 0.5))
        id2rc[int(cid)], rc2id[(r, c)] = (r, c), int(cid)

    path = [p for p in path if p in id2rc]
    if len(path) < 3: return None

    blue_px = sol[bmask]
    blue_color = tuple(int(v) for v in blue_px.mean(0)) if len(blue_px) else (15, 116, 187)
    plen = sum(((id2rc[a][0]-id2rc[b][0])**2 + (id2rc[a][1]-id2rc[b][1])**2)**0.5 for a, b in zip(path[:-1], path[1:])) * cell_px
    stroke = max(3, int(round(len(blue_px) / max(1.0, plen)))) if len(blue_px) else max(3, int(cell_px * 0.1))

    return MazeSample(sol, morig, ids, bmask, path, id2rc, rc2id, cell_px, blue_color, stroke, width, height)


def _center(rc, cell_px) -> tuple:
    r, c = rc
    return (int((c + 0.5) * cell_px), int((r + 0.5) * cell_px))


def _is_open(morig: np.ndarray, r, c, r2, c2, cell_px: float) -> bool:
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
    if band.size == 0: return False
    return (band.max(axis=2) < _WALL_DARK).mean() < _WALL_FRAC


def _open_nonblocked(ms: MazeSample, cell: int, blocked: set) -> list:
    r, c = ms.id2rc[cell]
    out = []
    for nb in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        nid = ms.rc2id.get(nb)
        if nid is not None and nid not in blocked and _is_open(ms.morig, r, c, nb[0], nb[1], ms.cell_px):
            out.append(nid)
    return out


def render_partial(ms: MazeSample, p: float) -> tuple:
    # Special case for 0.0 - always return at least the start node
    k = max(1, int(round(len(ms.path) * p)))
    shown = ms.path[:k]
    img = ms.morig.copy()
    sel = np.isin(ms.ids, shown) & ms.bmask
    img[sel] = ms.sol[sel]
    return img, shown


def apply_gap(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random, gap_frac: float) -> tuple:
    """Wycina zadany % całkowitej ścieżki w losowym miejscu."""
    n = len(shown)
    glen = max(1, int(round(n * gap_frac)))

    if n - glen <= 0:
        start, glen = 0, n
    else:
        start = rng.randint(0, n - glen)

    erased = shown[start:start + glen]
    if not erased:
        return img, []

    out = img.copy()
    sel = np.isin(ms.ids, erased)
    out[sel] = ms.morig[sel]
    return out, erased


def apply_add(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random) -> tuple:
    """Rysuje złą ścieżkę startującą z czoła narysowanego prefixu, aż do dead-endu.

    Walk rules (as specified):
      a. while the only way forward is the GT path itself, follow the GT path;
      b. the moment a non-GT opening exists, take it and never re-enter a wall,
         the GT path, or a cell the walk already used — run until a dead end.

    So at level 0% the walk leaves the start cell (a dead end on most AMAZE
    boards) along the solution until the first junction, then diverges. The GT
    cells crossed in step (a) are *correct*, so they are drawn but reported
    separately from the wrong ones.

    Returns (img, wrong_cells, gt_walked, diverge_offset):
      wrong_cells    — the off-path cells, i.e. what the model has to erase.
      gt_walked      — GT cells drawn by rule (a) beyond `shown`; correct, so
                       they join the "must survive" set, not the corruption.
      diverge_offset — how many GT cells rule (a) had to cover before a branch
                       existed (0 = diverged at the frontier itself).
    wrong_cells is empty only when the walk reaches the goal without ever
    finding an opening — the caller must treat that board as *not corrupted*
    rather than as a perfect recovery.
    """
    on_path = set(ms.path)
    idx = {c: i for i, c in enumerate(ms.path)}
    visited = {shown[-1]}

    # (a) follow the GT path forward while it is the only option.
    cur = shown[-1]
    gt_walked: list = []
    while True:
        free = [n for n in _open_nonblocked(ms, cur, on_path) if n not in visited]
        if free:
            break
        nxt_i = idx[cur] + 1
        if nxt_i >= len(ms.path):
            return img, [], [], -1          # walked to the goal, never a branch
        cur = ms.path[nxt_i]
        visited.add(cur)
        gt_walked.append(cur)

    # (b) diverge and random-walk off-path to a dead end.
    blocked = on_path | visited
    walk = [cur]
    while True:
        cand = _open_nonblocked(ms, cur, blocked)
        if not cand:
            break
        nxt = rng.choice(cand)
        walk.append(nxt)
        blocked.add(nxt)
        cur = nxt

    wrong = walk[1:]
    if not wrong:
        return img, [], [], -1

    out = img.copy()
    full = [shown[-1]] + gt_walked + wrong
    for x, y in zip(full[:-1], full[1:]):
        cv2.line(out, _center(ms.id2rc[x], ms.cell_px), _center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
    return out, wrong, gt_walked, len(gt_walked)


def apply_shortcut(img: np.ndarray, ms: MazeSample, shown: list, rng: random.Random) -> tuple:
    """Rysuje chamską linię bezpośrednio z punktu `p` prosto do mety.

    Returns (img, line_mask, in_wall, off_path_line):
      line_mask     — every pixel of the drawn shortcut.
      in_wall       — the subset that lands on maze wall pixels.
      off_path_line — the subset that lies on cells NOT on the GT solution. This
                      is what "did the model erase the shortcut?" must be scored
                      on: the line leaves the frontier along legitimately-blue
                      cells, and can graze GT cells on the way, so scoring the
                      full line_mask penalises a perfect answer.
    """
    if not shown:
        return img, None, None, None

    frontier = shown[-1]
    goal = ms.path[-1]

    ca = _center(ms.id2rc[frontier], ms.cell_px)
    cb = _center(ms.id2rc[goal], ms.cell_px)

    out = img.copy()
    cv2.line(out, ca, cb, ms.blue_color, ms.stroke)

    line_mask = np.zeros(ms.ids.shape, np.uint8)
    cv2.line(line_mask, ca, cb, 1, ms.stroke)
    line_mask = line_mask > 0
    wall_dark = ms.morig.max(axis=2) < _WALL_DARK
    off_path = line_mask & ~np.isin(ms.ids, ms.path)

    return out, line_mask, (line_mask & wall_dark), off_path


def predicted_cells(img_native: np.ndarray, ms: MazeSample) -> set:
    """Cell ids covered by blue in a native-resolution image.

    Uses the same opened mask as eval/amaze_eval.py so the probe's own
    recovery/collateral counters and the paper's coverage/violation/pass agree
    on what "the model drew here" means.
    """
    bm = blue_mask(img_native, morph_open=True)
    s = set(int(cid) for cid in ms.ids[bm].tolist())
    s.discard(0)
    return s
