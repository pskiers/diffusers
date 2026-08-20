"""
eval/maze_eval.py – Evaluation for the Maze dataset (datasets/maze_dataset.py).

Unlike MNIST Sudoku (eval/mnist_eval.py), no learned classifier is needed:
MazeDataset renders every cell as a flat color from a small fixed palette
(white background, gray padding, blue path, green start, red goal), so a
generated cell can be classified by simple nearest-color matching on its
interior pixels.

Provides:
  extract_cell_colors – average interior RGB color of every cell.
  classify_cells       – nearest-palette-color label per cell.
  evaluate_mazes       – classifies a batch of generated images and scores
                         them against ground truth (cell/puzzle accuracy) and
                         against the maze's own wall structure (constraint
                         validity — a valid, connected start->goal path,
                         independent of whether it matches the reference).
  make_maze_panel_image – condition | generated | true-solution panel for
                         WandB logging.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from datasets.maze_dataset import DOWN, LEFT, RIGHT, UP, render_maze

# Cell-label ids (order matters: used as the "closest palette color" index).
BACKGROUND, PADDING, PATH, START, GOAL = 0, 1, 2, 3, 4
_PALETTE = torch.tensor(
    [
        [1.0, 1.0, 1.0],   # BACKGROUND (white)
        [0.5, 0.5, 0.5],   # PADDING (gray)
        [0.0, 0.35, 0.95],  # PATH (blue)
        [0.0, 0.8, 0.0],   # START (green)
        [0.85, 0.0, 0.0],  # GOAL (red)
    ]
)
_STEP = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


@torch.no_grad()
def extract_cell_colors(
    images: torch.Tensor, grid_size: int, cell_size: int, margin: int = 4
) -> torch.Tensor:
    """Average RGB color in the interior of every cell (avoiding wall-line
    pixels near cell borders).

    images: (B, 3, grid_size*cell_size, grid_size*cell_size) float [0, 1]
    Returns (B, grid_size*grid_size, 3).
    """
    B, C, H, W = images.shape
    m = min(margin, cell_size // 2 - 1) if cell_size > 2 else 0
    cells = images.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    # (B, C, grid_size, grid_size, cell_size, cell_size)
    interior = cells[..., m:cell_size - m, m:cell_size - m] if m > 0 else cells
    mean = interior.mean(dim=(-1, -2))  # (B, C, grid_size, grid_size)
    return mean.permute(0, 2, 3, 1).reshape(B, grid_size * grid_size, C)


@torch.no_grad()
def classify_cells(images: torch.Tensor, grid_size: int, cell_size: int) -> torch.Tensor:
    """Classify every cell to the nearest palette color.

    Returns (B, grid_size*grid_size) int64, values in
    {BACKGROUND, PADDING, PATH, START, GOAL}.
    """
    colors = extract_cell_colors(images, grid_size, cell_size)  # (B, N, 3)
    palette = _PALETTE.to(colors.device, colors.dtype)  # (5, 3)
    dists = (colors.unsqueeze(2) - palette.view(1, 1, -1, 3)).pow(2).sum(-1)  # (B, N, 5)
    return dists.argmin(dim=-1)


def _path_validity(
    on_path_mask: np.ndarray,    # (N,) bool, this sample's predicted on-path cells
    token_conditions: np.ndarray,  # (N,) int, true openness bitmask
    active_mask: np.ndarray,     # (N,) bool
    start_idx: int,
    goal_idx: int,
    grid_size: int,
) -> bool:
    """True iff the predicted on-path cells (within active cells) form a
    single simple path connecting start_idx to goal_idx, using only truly
    open (non-wall) edges — the maze's hard logical constraint, independent
    of whether it matches the unique reference shortest path.
    """
    on_path = {
        i for i in range(grid_size * grid_size)
        if active_mask[i] and on_path_mask[i]
    }
    if start_idx not in on_path or goal_idx not in on_path or len(on_path) < 2:
        return False

    # BFS over on_path cells using only real open edges; must reach goal and
    # touch every on_path cell exactly once (no disconnected extra blobs, no
    # branching — a simple path).
    visited = {start_idx}
    q = deque([start_idx])
    while q:
        i = q.popleft()
        r, c = divmod(i, grid_size)
        for d, (dr, dc) in _STEP.items():
            if int(token_conditions[i]) & d:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    j = nr * grid_size + nc
                    if j in on_path and j not in visited:
                        visited.add(j)
                        q.append(j)
    if visited != on_path or goal_idx not in visited:
        return False

    # Degree check: exactly 2 cells (start, goal) have degree 1, the rest degree 2.
    deg = {i: 0 for i in on_path}
    for i in on_path:
        r, c = divmod(i, grid_size)
        for d, (dr, dc) in _STEP.items():
            if int(token_conditions[i]) & d:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    j = nr * grid_size + nc
                    if j in on_path:
                        deg[i] += 1
    ones = sum(1 for v in deg.values() if v == 1)
    twos = sum(1 for v in deg.values() if v == 2)
    return ones == 2 and twos == len(on_path) - 2


@torch.no_grad()
def evaluate_mazes(
    images: torch.Tensor,             # (B, 3, H, W) float [0, 1] — generated
    conditions: torch.Tensor,         # (B, 3, H, W) float [0, 1] — spatial_conditions (ground truth start/goal)
    solution: torch.Tensor,           # (B, N) int64 — 0/1, IGNORE_LABEL_ID (-100) outside active region
    solution_mask: torch.Tensor,      # (B, N) bool — True = active/in-bounds cell
    token_conditions: torch.Tensor,   # (B, N) int64 — true per-cell openness bitmask
    grid_size: int,
    cell_size: int,
) -> dict:
    """Classify cells in *images* and score against ground truth + maze constraints.

    cell_acc              — per-cell on/off-path accuracy vs `solution`, over active cells.
    puzzle_acc             — fraction of mazes where the predicted on-path set
                             exactly equals the reference solution set.
    constraint_puzzle_acc — fraction of mazes whose predicted path is a valid
                             connected start->goal path per the maze's real
                             wall structure, regardless of whether it matches
                             the reference solution (mirrors Sudoku's
                             constraint_puzzle_acc: a model could draw an
                             unrelated-but-valid path and still score here).

    Returns dict with keys: cell_acc, puzzle_acc, constraint_puzzle_acc, preds,
    plus per-sample arrays (per_sample_cell_acc, per_sample_exact,
    per_sample_valid, per_sample_active_size) so a caller can bucket results
    by maze difficulty (active_size) instead of only seeing one pooled number
    across a mixed-difficulty batch — see MazeEvalCallback.
    """
    device = images.device
    B, N = solution.shape
    labels = classify_cells(images, grid_size, cell_size)  # (B, N)
    pred_on_path = (labels == PATH) | (labels == START) | (labels == GOAL)  # (B, N) bool

    gt_on_path = solution.to(device) == 1
    mask = solution_mask.to(device)

    correct = pred_on_path == gt_on_path
    per_sample_cell_acc = torch.where(
        mask.any(dim=1), (correct & mask).sum(dim=1).float() / mask.sum(dim=1).clamp(min=1).float(), torch.zeros(B, device=device)
    )
    cell_acc = correct[mask].float().mean().item() if mask.any() else 0.0
    per_sample_exact = (correct | ~mask).all(dim=1)
    puzzle_acc = per_sample_exact.float().mean().item()

    cond_labels = classify_cells(conditions, grid_size, cell_size).cpu().numpy()  # (B, N)
    token_np = token_conditions.cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = pred_on_path.cpu().numpy()

    valid = np.zeros(B, dtype=bool)
    for b in range(B):
        starts = np.where(cond_labels[b] == START)[0]
        goals = np.where(cond_labels[b] == GOAL)[0]
        if len(starts) != 1 or len(goals) != 1:
            continue  # condition image itself is malformed — shouldn't happen for real data
        valid[b] = _path_validity(
            pred_np[b], token_np[b], mask_np[b], int(starts[0]), int(goals[0]), grid_size
        )

    active_size = np.round(np.sqrt(mask_np.sum(axis=1))).astype(int)  # (B,)

    return {
        "cell_acc": cell_acc,
        "puzzle_acc": puzzle_acc,
        "constraint_puzzle_acc": float(valid.mean()),
        "preds": pred_on_path.cpu(),
        "per_sample_cell_acc": per_sample_cell_acc.cpu().numpy(),
        "per_sample_exact": per_sample_exact.cpu().numpy(),
        "per_sample_valid": valid,
        "per_sample_active_size": active_size,
    }


def make_maze_panel_image(
    condition: torch.Tensor,          # (3, H, W) float [0,1] — puzzle shown to model
    generated: torch.Tensor,          # (3, H, W) float [0,1] — model output
    token_conditions: np.ndarray,     # (N,) int — true openness bitmask
    solution: np.ndarray,             # (N,) int — 0/1/-100 ground truth
    solution_mask: np.ndarray,        # (N,) bool — active cells
    grid_size: int,
    cell_size: int,
) -> np.ndarray:
    """Build a horizontal (condition | generated | true solution) panel image."""
    active_size = int(round(float(np.sqrt(solution_mask.sum()))))
    open_mask = token_conditions.reshape(grid_size, grid_size)[:active_size, :active_size]
    sol_grid = solution.reshape(grid_size, grid_size)
    path_cells = {(r, c) for r in range(active_size) for c in range(active_size) if sol_grid[r, c] == 1}
    cond_labels = classify_cells(condition.unsqueeze(0), grid_size, cell_size)[0].numpy()
    starts = np.where(cond_labels == START)[0]
    goals = np.where(cond_labels == GOAL)[0]
    start = divmod(int(starts[0]), grid_size) if len(starts) else (0, 0)
    goal = divmod(int(goals[0]), grid_size) if len(goals) else (0, 0)

    true_img = render_maze(grid_size, cell_size, active_size, open_mask, start, goal, path_cells)

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        return (t.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    sep = np.full((condition.shape[-2], 4, 3), 200, dtype=np.uint8)
    true_uint8 = (np.transpose(true_img, (1, 2, 0)) * 255).astype(np.uint8)
    return np.concatenate([to_uint8(condition), sep, to_uint8(generated), sep, true_uint8], axis=1)
