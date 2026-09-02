"""
experiments/conditioning_lib.py — shared helpers for interpreting the TRM's
ControlNet conditioning on the 12x12 grid (amaze, maze task).

Four experiment scripts build on this module:
    conditioning_heatmaps.py    — per-cell energy + 11-channel maps overlaid on the maze
    conditioning_trajectory.py  — painter x0(t) alongside the conditioning(t)
    conditioning_swap.py         — inject maze-B's conditioning onto maze-A's canvas
    decode_conditioning.py       — linear probes: what maze facts the conditioning encodes

Key architecture facts this module relies on (verified against the code):
  * TRM logits (B, 144, 11) -> ControlNetTranslator._logits_to_spatial
    (bridge_mode="normalized" == per-channel BatchNorm1d, deterministic in eval)
    -> spatial (B, 11, 12, 12) == "Tap2" -> bilinear upsample -> pyramid -> residuals.
  * The bilinear upsample adds NO information: analyse on the native 12x12,
    overlay with NEAREST x12 (honest, matches the encoder's factor-12 tiling).
  * vocab_size=11 is vestigial (Sudoku); the 11 channels are continuous, NOT classes.

Capture mechanics:
  * ConditioningCapture monkeypatches translator._logits_to_spatial to stash the
    post-bridge map (Tap2) and its input logits (Tap1), one entry per model forward.
  * Under CFG the model is called twice per denoising step (conditional first), so
    the translator fires twice/step -> use ConditioningCapture.conditional(...) to
    keep the conditional pass only.
  * DiT (ConcatDiTPainter) has no translator -> capture is a no-op (enabled=False).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

TRM_ROOT = Path(__file__).resolve().parent.parent

GRID = 12  # amaze thinker V2 reasoning grid (seq_len = 144 = 12*12)


# ── Conditioning capture ──────────────────────────────────────────────────────


class ConditioningCapture:
    """Context manager that records the TRM's conditioning per model forward.

    On __enter__ it replaces ``model.thinker_painter_translator._logits_to_spatial``
    with a wrapper that appends the post-bridge ``(B, C, grid, grid)`` map (Tap2)
    and its input logits ``(B, N, C)`` (Tap1) to ``self.spatial`` / ``self.logits``,
    then delegates to the original. Restored on __exit__.

    No-op when the model exposes no translator (e.g. the DiT baseline): ``enabled``
    is False and the stores stay empty.
    """

    def __init__(self, model):
        self.translator = getattr(model, "thinker_painter_translator", None)
        self.spatial: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self._orig = None

    @property
    def enabled(self) -> bool:
        return self.translator is not None

    def __enter__(self) -> "ConditioningCapture":
        if self.translator is None:
            return self
        orig = self.translator._logits_to_spatial
        self._orig = orig
        store_s, store_l = self.spatial, self.logits

        def _patched(logits):
            out = orig(logits)
            store_l.append(logits.detach().float().cpu())
            store_s.append(out.detach().float().cpu())
            return out

        self.translator._logits_to_spatial = _patched  # instance attr shadows the method
        return self

    def __exit__(self, *exc) -> bool:
        if self._orig is not None:
            # remove the instance attribute -> restores the bound class method
            try:
                del self.translator._logits_to_spatial
            except AttributeError:
                self.translator._logits_to_spatial = self._orig
        self._orig = None
        return False

    def clear(self) -> None:
        self.spatial.clear()
        self.logits.clear()

    def conditional(self, per_step: int = 2) -> list[torch.Tensor]:
        """The conditional-pass Tap2 maps, one per denoising step.

        CFGPredictor calls model(sample) [conditional] then model(null) [uncond]
        each step, so translator calls come in groups of ``per_step`` with the
        conditional one first -> ``self.spatial[::per_step]``.
        """
        return self.spatial[::per_step]

    def conditional_logits(self, per_step: int = 2) -> list[torch.Tensor]:
        return self.logits[::per_step]


@torch.no_grad()
def capture_teacher_forced(
    model, conditions, t: int, device, seed: int = 0
) -> dict:
    """One teacher-forced forward at a FIXED timestep ``t``: noise the GT solution
    to level t, run the thinker, return its conditioning + recurrent state.

    Using a high t (little/no solution leaked through x_noisy) makes any decoding
    result mean "the TRM reasoned this", not "it copied a visible answer".

    Returns dict with:
        spatial : (B, 11, 12, 12) post-bridge conditioning (Tap2)
        logits  : (B, 144, 11)   raw thinker logits (Tap1)
        z_H     : (B, 144, 512)  final reasoning-state carry
    """
    import dataclasses

    images = conditions.images.to(device)
    B = images.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(images.shape, device=device, generator=gen)
    t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
    x_noisy = model.scheduler.add_noise(images, noise, t_batch)
    sample = dataclasses.replace(conditions, x_noisy=x_noisy, timesteps=t_batch)

    zH_out: list[torch.Tensor] = []
    with ConditioningCapture(model) as cap:
        pred, _z_H, _z_L = model.forward_with_carry(sample, zH_out=zH_out)
        spatial = cap.spatial[-1] if cap.spatial else None
    return {
        "spatial": spatial,
        "logits": pred.logits.detach().float().cpu() if pred.logits is not None else None,
        "z_H": zH_out[-1].detach().float().cpu() if zH_out else None,
    }


# ── Image / heatmap helpers ───────────────────────────────────────────────────


def to_hwc_uint8(img) -> np.ndarray:
    """Any (C,H,W)/(H,W,3)/(H,W) tensor|ndarray|PIL in [0,1] or [0,255] -> (H,W,3) uint8."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    # .float() first: bf16 (the painter's dtype) has no numpy equivalent.
    t = img.detach().cpu().float().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    t = t.astype(np.float32)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = np.transpose(t, (1, 2, 0))
    if t.ndim == 2:
        t = np.stack([t] * 3, axis=-1)
    if t.shape[-1] == 1:
        t = np.repeat(t, 3, axis=-1)
    if float(t.max()) <= 1.0 + 1e-6:
        t = t * 255.0
    return np.clip(t, 0, 255).astype(np.uint8)


def energy_map(spatial) -> np.ndarray:
    """(C,H,W) or (B,C,H,W) -> per-cell L2 norm over the channel dim, (H,W) / (B,H,W).

    L2 (energy), NOT a signed sum: the normalized bridge centres each channel near
    zero, so a signed sum cancels; energy shows which cells deviate from baseline.
    """
    t = spatial if isinstance(spatial, torch.Tensor) else torch.as_tensor(np.asarray(spatial))
    return t.float().pow(2).sum(dim=-3).sqrt().cpu().numpy()


def upscale_nn(grid2d: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour block upscale (honest: each token stays one factor-sized block)."""
    return np.kron(np.asarray(grid2d, dtype=np.float32), np.ones((factor, factor), np.float32))


def _resize_nn(arr_hwc: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    im = Image.fromarray(arr_hwc)
    if im.size != (W, H):
        im = im.resize((W, H), Image.NEAREST)
    return np.asarray(im)


def heat_to_rgb(heat2d: np.ndarray, cmap: str = "magma",
                vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Scalar field -> (H,W,3) uint8 via a matplotlib colormap (min-max normalised)."""
    import matplotlib
    from matplotlib.colors import Normalize

    try:
        cmap_fn = matplotlib.colormaps[cmap]          # matplotlib >= 3.5 (Helios)
    except (AttributeError, KeyError):
        import matplotlib.cm as cm                      # fallback for < 3.9
        cmap_fn = cm.get_cmap(cmap)

    h = np.asarray(heat2d, dtype=np.float32)
    lo = float(np.min(h)) if vmin is None else vmin
    hi = float(np.max(h)) if vmax is None else vmax
    rgba = cmap_fn(Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-8)(h))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_grid_on_maze(grid2d: np.ndarray, maze_img, alpha: float = 0.55,
                         cmap: str = "magma", vmin=None, vmax=None) -> np.ndarray:
    """NEAREST-upscale a (grid,grid) heat map to the maze size and alpha-blend it.

    Returns (H,W,3) uint8. The grid tiles the image uniformly (as the bridge's
    upsample assumes); for maze scale n != grid the heat cells straddle maze cells.
    """
    maze = to_hwc_uint8(maze_img)
    H, W = maze.shape[:2]
    factor = max(1, H // int(np.asarray(grid2d).shape[0]))
    heat_up = upscale_nn(grid2d, factor)
    heat_rgb = _resize_nn(heat_to_rgb(heat_up, cmap, vmin, vmax), (H, W))
    return np.clip(alpha * heat_rgb + (1.0 - alpha) * maze, 0, 255).astype(np.uint8)


def channel_montage(spatial_cHW, cols: int = 4, cmap: str = "viridis",
                    factor: int = 12, pad: int = 2) -> np.ndarray:
    """The 11 channel maps as a padded montage, each min-max normalised on its own
    colour scale (so near-dead channels look flat, not amplified). (H,W,3) uint8.
    """
    s = spatial_cHW.detach().cpu().numpy() if isinstance(spatial_cHW, torch.Tensor) else np.asarray(spatial_cHW)
    C = s.shape[0]
    tiles = [_resize_nn(heat_to_rgb(upscale_nn(s[c], factor), cmap), (s.shape[1] * factor, s.shape[2] * factor))
             for c in range(C)]
    th, tw = tiles[0].shape[:2]
    rows = (C + cols - 1) // cols
    canvas = np.full((rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3), 255, np.uint8)
    for k, tile in enumerate(tiles):
        r, cc = divmod(k, cols)
        y = pad + r * (th + pad)
        x = pad + cc * (tw + pad)
        canvas[y:y + th, x:x + tw] = tile
    return canvas


# ── Cell-map decode + 12x12 alignment + maze labels ───────────────────────────


def decode_cell_ids(cell_map) -> np.ndarray:
    """RGB-packed cell_map (PIL/ndarray/tensor) -> (H,W) int64 cell ids (id = R|G<<8|B<<16).
    Same formula as eval.amaze_eval.AmazeMetrics.decode_cell_map_ids; wall/background = 0.
    """
    if isinstance(cell_map, Image.Image):
        arr = np.asarray(cell_map.convert("RGB")).astype(np.uint32)
    else:
        arr = cell_map.detach().cpu().numpy() if isinstance(cell_map, torch.Tensor) else np.asarray(cell_map)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r | (g << 8) | (b << 16)).astype(np.int64)


def token_cell_grid(cell_ids_hw: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Assign each of the grid*grid tokens the majority cell id in its image block."""
    H, W = cell_ids_hw.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    out = np.zeros((grid, grid), np.int64)
    for i in range(grid):
        for j in range(grid):
            block = cell_ids_hw[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel()
            if block.size:
                vals, counts = np.unique(block, return_counts=True)
                out[i, j] = int(vals[counts.argmax()])
    return out


def _cell_adjacency(cell_ids_hw: np.ndarray) -> dict:
    """Passage graph over cells: two nonzero cells are adjacent iff their regions
    touch directly (a 4-neighbour pixel pair, one in each), i.e. no wall between.
    """
    a = cell_ids_hw
    pair_lists = []
    left, right = a[:, :-1], a[:, 1:]
    mh = (left > 0) & (right > 0) & (left != right)
    if mh.any():
        pair_lists.append(np.stack([left[mh], right[mh]], axis=1))
    top, bot = a[:-1, :], a[1:, :]
    mv = (top > 0) & (bot > 0) & (top != bot)
    if mv.any():
        pair_lists.append(np.stack([top[mv], bot[mv]], axis=1))
    adj: dict = defaultdict(set)
    if pair_lists:
        pairs = np.unique(np.sort(np.concatenate(pair_lists, axis=0), axis=1), axis=0)
        for u, v in pairs:
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))
    return adj


def _bfs_dist(adj: dict, source: int) -> dict:
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _red_marker_grid(m_original_img, grid: int = GRID) -> np.ndarray:
    """Fraction of red (start/goal marker) pixels per token -> (grid,grid) float in [0,1]."""
    arr = to_hwc_uint8(m_original_img).astype(np.int32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red = ((r > 150) & (g < 100) & (b < 100)).astype(np.float32)
    H, W = red.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    out = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            blk = red[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = float(blk.mean()) if blk.size else 0.0
    return out


def maze_token_labels(metadata: dict, grid: int = GRID, with_dist: bool = True) -> dict:
    """Per-token maze labels aligned to the grid*grid reasoning canvas.

    Returns dict of (grid,grid) arrays:
        on_path      : {0,1}  cell is on the solution path
        is_wall      : {0,1}  token maps to wall/background (cell id 0)
        path_pos     : index along the ordered path (-1 off path); 0 at path[0] end
        is_marker    : {0,1}  token overlaps a red start/goal marker
        dist_to_goal : geodesic cell distance to path end path[-1]  (with_dist, v2)
        dist_to_start: geodesic cell distance to path end path[0]   (with_dist, v2)
    plus token_cells: (grid,grid) majority cell id per token.

    Start/goal are both red, so which path end is "goal" is a convention (path[-1]);
    both distance fields are provided so the decoder can use either / the min.
    """
    meta = metadata.get("metadata")
    meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    path_ids = [int(c) for c in meta.get("path_cell_ids", [])]
    pos_of = {cid: k for k, cid in enumerate(path_ids)}
    pset = set(path_ids)

    cell_ids_hw = decode_cell_ids(metadata.get("cell_map"))
    tok = token_cell_grid(cell_ids_hw, grid)

    on_path = np.isin(tok, list(pset) if pset else [-1]).astype(np.float32)
    is_wall = (tok == 0).astype(np.float32)
    L = len(path_ids)
    path_pos = np.full((grid, grid), -1.0, np.float32)
    for i in range(grid):
        for j in range(grid):
            cid = int(tok[i, j])
            if cid in pos_of:
                path_pos[i, j] = float(pos_of[cid])

    labels = {"on_path": on_path, "is_wall": is_wall, "path_pos": path_pos, "token_cells": tok}
    if metadata.get("m_original_img") is not None:
        labels["is_marker"] = (_red_marker_grid(metadata["m_original_img"], grid) > 0.02).astype(np.float32)

    if with_dist and L >= 2:
        try:
            adj = _cell_adjacency(cell_ids_hw)
            for name, endpoint in (("dist_to_goal", path_ids[-1]), ("dist_to_start", path_ids[0])):
                d = _bfs_dist(adj, endpoint)
                field = np.full((grid, grid), np.nan, np.float32)
                for i in range(grid):
                    for j in range(grid):
                        field[i, j] = d.get(int(tok[i, j]), np.nan)
                labels[name] = field
        except Exception:
            pass
    return labels


# ── Good / bad puzzle selection (Pass@1) ──────────────────────────────────────


@torch.no_grad()
def select_good_bad(model, ds, device, n_each: int, seed: int = 0,
                    batch_size: int = 24, max_scan: Optional[int] = None) -> tuple[list[int], list[int]]:
    """Generate one attempt per puzzle, score with AmazeMetrics, and return
    (solved_idxs, failed_idxs), each up to ``n_each`` long.
    """
    from eval.amaze_eval import AmazeMetrics

    scorer = AmazeMetrics(device=device, task="maze")
    pipeline = model.sampling_pipeline
    good: list[int] = []
    bad: list[int] = []
    limit = len(ds) if max_scan is None else min(max_scan, len(ds))

    for start in range(0, limit, batch_size):
        if len(good) >= n_each and len(bad) >= n_each:
            break
        idxs = list(range(start, min(start + batch_size, limit)))
        puzzles = [ds[i] for i in idxs]
        conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)
        gen = pipeline.sample_one_batch(
            model, conditions, device,
            generator=torch.Generator(device=device).manual_seed(seed + start),
        )
        decoded = model.decode_for_eval(gen).cpu()
        metadata = [p.metadata if p.metadata is not None else {} for p in puzzles]
        recs = scorer.compute_and_accumulate_metrics(decoded, metadata)
        for i, rec in zip(idxs, recs):
            if rec.get("pass", 0.0) >= 1.0:
                if len(good) < n_each:
                    good.append(i)
            elif len(bad) < n_each:
                bad.append(i)
    return good, bad


# ── Model loading + wandb attach (shared across scripts) ──────────────────────


def load_model(cfg, checkpoint: str):
    """Build the model from cfg and load ``checkpoint`` (EMA per cfg.use_ema). eval()."""
    from eval.checkpoint_utils import load_checkpoint as _load_checkpoint
    from factory import build_model
    from hydra.utils import instantiate

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    return model


def resolve_run_id(cfg, checkpoint: str) -> Optional[str]:
    explicit = cfg.get("wandb_run_id", None)
    id_file = Path(checkpoint).parent / "wandb_run_id.txt"
    if explicit:
        return str(explicit)
    return id_file.read_text().strip() if id_file.exists() else None


def wandb_attach(cfg, checkpoint: str, logger=None):
    """Attach to the training run (same id file as the metrics/trajectory scripts).
    Returns a wandb run or None; never raises.
    """
    project = cfg.run.get("wandb_project", None) if hasattr(cfg, "run") else None
    run_id = resolve_run_id(cfg, checkpoint)
    if not project or not run_id:
        if logger:
            logger.info("wandb: skipping (need run.wandb_project and a run id via "
                        "+wandb_run_id= or <ckpt-dir>/wandb_run_id.txt).")
        return None
    import wandb

    try:
        return wandb.init(project=project, id=run_id, resume="allow",
                          settings=wandb.Settings(init_timeout=int(cfg.get("wandb_init_timeout", 60))))
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning(f"wandb: init failed/timed out ({e!r}); PNGs still saved on disk.")
        return None
