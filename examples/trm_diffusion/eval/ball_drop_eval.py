"""
eval/ball_drop_eval.py – Evaluation for the Ball Drop dataset
(datasets/ball_drop_dataset.py).

Unlike Steiner Tree/Max-Area Polygon, this is a reachability/success task,
not an optimization one — there's no "shortest"/"largest" ground truth to
score a ratio against (see datasets/ball_drop_generation.py's docstring).
Evaluation instead re-simulates physics: extract the drawn solution line(s)
from the generated image (pixel-color based — see _extract_lines), add them
to the *exact* fixed scene from datasets/ball_drop_generation.py, drop the
ball from the instance's recorded start position, and check whether it
settles in the recorded target bucket. Both ball_start_x and target_bucket
must be supplied by the caller (BallDropEvalCallback, via
BallDropDataset.physics_spec_for(puzzle_id)) — unlike Steiner/Polygon's
embedding_conditions, this task has no per-slot point-set field to carry
them in the batch itself (there's nothing to pad — a single scalar start
position and a single scalar target bucket per instance), so callers look
them up directly rather than reading them off a DataSample field.

Calibration: round-tripping ground-truth rendered images (render -> extract
-> re-simulate) through this pipeline recovers the original target bucket
~75% of the time (checked against 200 of our own generated instances) — a
real noise ceiling, not a bug: this task's long, near-horizontal ramps make
the ball's final bucket a physically chaotic function of exact line
placement, so the small pixel-quantization/extraction error between a
line's exact stored coordinates and its rendered-then-extracted endpoints
occasionally cascades into a different final bucket (or a failure to
settle at all — ~80.5% of round-tripped instances settle). Read
constraint_puzzle_acc relative to this ~75% ceiling, not against 100%, same
spirit as Steiner Tree's ~78%/Max-Area Polygon's ~97.7% documented
reconstruction ceilings.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from datasets.ball_drop_dataset import COLOR_LINE
from datasets.ball_drop_generation import add_ball, add_lines, bucket_bounds_for, bucket_of, build_scene, simulate

# Line pixels are red-dominant (COLOR_LINE ~ (230, 30, 30)) vs. the other
# fixed palette colors (background black, gray structure, yellow target,
# green ball) — thresholding on R-vs-(G,B) dominance avoids needing a full
# nearest-color classification.
_LINE_R_MARGIN = 40
_LINE_R_MIN = 100
_MIN_COMPONENT_AREA = 6  # px; drops stray anti-aliasing specks.


def _extract_lines(img_uint8: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Extract solution line segments from a rendered (H, W, 3) uint8 image.

    Thresholds red-dominant pixels, then fits a line (via PCA on pixel
    coordinates) to each connected component — a component may in principle
    merge two crossing lines into one segment (a resolution-limit caveat,
    same spirit as Steiner Tree's documented vertex-detection limit), which
    conservatively degrades this sample's physics outcome rather than
    inflating it.

    Returns (x1, y1, x2, y2) normalized-[0,1] quadruples, in the same y-up
    convention as datasets/ball_drop_generation.py (y = 1 - row/(H-1)).
    """
    r, g, b = img_uint8[..., 0].astype(int), img_uint8[..., 1].astype(int), img_uint8[..., 2].astype(int)
    mask = (r > _LINE_R_MIN) & (r - g > _LINE_R_MARGIN) & (r - b > _LINE_R_MARGIN)
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)

    n_labels, labels = cv2.connectedComponents(mask_u8)
    H, W = img_uint8.shape[:2]
    lines = []
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if len(xs) < _MIN_COMPONENT_AREA:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        proj = centered @ direction
        p1_px = mean + direction * proj.min()
        p2_px = mean + direction * proj.max()
        x1, y1 = p1_px[0] / (W - 1), 1.0 - p1_px[1] / (H - 1)
        x2, y2 = p2_px[0] / (W - 1), 1.0 - p2_px[1] / (H - 1)
        lines.append((x1, y1, x2, y2))
    return lines


def _resimulate(lines: list, ball_start_x: float, target_bucket: int, image_size: int) -> tuple[bool, bool]:
    """Returns (settled, hit_target_bucket)."""
    space = build_scene(image_size)
    lines_px = [((x1 * image_size, y1 * image_size), (x2 * image_size, y2 * image_size)) for x1, y1, x2, y2 in lines]
    add_lines(space, lines_px, image_size)
    ball = add_ball(space, ball_start_x * image_size, image_size)
    _, final_pos, settled = simulate(space, ball, image_size)
    if not settled:
        return False, False
    bucket = bucket_of(final_pos.x, bucket_bounds_for(image_size))
    return True, bucket == target_bucket


@torch.no_grad()
def evaluate_ball_drop(
    images: torch.Tensor,       # (B, 3, H, W) float, generated, in [-1, 1]-ish
    ball_start_x: np.ndarray,   # (B,) float in [0,1]
    target_bucket: np.ndarray,  # (B,) int in [0,3]
    image_size: int,
) -> dict:
    """Extract solution line(s) from each generated image, re-simulate
    physics with the instance's recorded ball start, and check whether the
    ball settles in the recorded target bucket.

    Returns dict with keys:
      constraint_puzzle_acc — fraction of samples whose extracted lines
                              route the ball into the target bucket.
      per_sample_valid      — (B,) bool numpy array of that pass/fail.
      per_sample_settled    — (B,) bool numpy array; whether the ball
                              settled at all (independent of which bucket).
      per_sample_num_lines_extracted — (B,) int numpy array; diagnostic —
                              distinguishes "drew nothing legible" from
                              "drew lines but wrong physical outcome".
    """
    B = images.shape[0]
    imgs = images.clamp(-1, 1).cpu().numpy()
    imgs_u8 = ((imgs.transpose(0, 2, 3, 1) + 1.0) * 127.5).astype(np.uint8)  # (B, H, W, 3)

    valid = np.zeros(B, dtype=bool)
    settled_arr = np.zeros(B, dtype=bool)
    n_lines_extracted = np.zeros(B, dtype=np.int64)

    for b in range(B):
        lines = _extract_lines(imgs_u8[b])
        n_lines_extracted[b] = len(lines)
        settled, hit = _resimulate(lines, float(ball_start_x[b]), int(target_bucket[b]), image_size)
        settled_arr[b] = settled
        valid[b] = hit

    return {
        "constraint_puzzle_acc": float(valid.mean()),
        "per_sample_valid": valid,
        "per_sample_settled": settled_arr,
        "per_sample_num_lines_extracted": n_lines_extracted,
    }


def make_ball_drop_panel_image(
    condition: torch.Tensor,  # (3, H, W) float in [-1,1] — puzzle (no lines)
    generated: torch.Tensor,  # (3, H, W) float in [-1,1] — model output
    reference: torch.Tensor,  # (3, H, W) float in [-1,1] — ground-truth solved scene
) -> np.ndarray:
    """condition | generated | reference, each mapped from [-1,1] back to [0,255] RGB."""

    def to_uint8(t: torch.Tensor) -> np.ndarray:
        return ((t.clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)

    sep = np.full((condition.shape[-2], 4, 3), 200, dtype=np.uint8)
    return np.concatenate([to_uint8(condition), sep, to_uint8(generated), sep, to_uint8(reference)], axis=1)
