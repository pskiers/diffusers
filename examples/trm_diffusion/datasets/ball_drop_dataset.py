"""
BallDropDataset – loads pre-simulated Ball Drop instances (generated offline
by datasets/ball_drop_generation.py using pymunk) and renders condition/
target images on the fly, for the "draw 1-3 ramps that route the falling
ball into the highlighted target bucket" logical-constraint benchmark.

Loosely based on the "SketchVLM" paper's (arXiv 2604.22875) physics
ball-drop benchmark — unlike Steiner Tree/Max-Area Polygon (arXiv
2510.21697), SketchVLM publishes no generation code or diffusion
architecture (it's a training-free VLM framework, not a diffusion paper);
see ball_drop_generation.py's docstring for how the scene/line-placement
conventions were reverse-engineered from the real dataset's metadata_json
instead. Also unlike Steiner/Polygon, this is a reachability/success task,
not an optimization one — there's no "shortest"/"largest" ground truth to
compare against, only "did the ball reach the target bucket" (see
eval/ball_drop_eval.py).

Rendering is RGB (unlike Steiner/Polygon's single-channel 3-level scheme) —
there are more than 3 semantically distinct element *types* here (fixed
scene structure, the highlighted target bucket, the ball's start position,
the solution lines), so a small fixed color palette (matching Maze's
approach) reads more naturally than trying to overload a single channel's
value levels:
  background      — black
  structure        — light gray (floor + 3 dividers; fixed geometry, same
                     for every instance — see ball_drop_generation.DIVIDER_X_FRAC
                     etc., imported directly rather than duplicated here)
  target bucket    — yellow (the one floor segment between dividers bounding
                     the instance's target_bucket, drawn instead of gray)
  ball start       — green circle
  solution lines   — red (only in the target image, never the condition)

Each sample:
  images             – (3, H, W) float32 in [-1, 1]; full scene incl. the
                         1-3 solution ramps.
  spatial_conditions – same shape; scene + target-bucket highlight + ball
                         start marker, no ramps — the puzzle condition. Not
                         fed to the (unconditional) stage-1 painter; this is
                         the thinker's primary conditioning input in stage 2
                         (spatial CNN), matching Steiner/Polygon — the
                         target bucket and ball start both live in the same
                         pixel coordinate space as the ramps being generated.
  puzzle_id          – () int64.

No embedding_conditions/embedding_mask, solution/solution_mask/
token_conditions fields: unlike Steiner/Polygon's variable-size point sets,
there's nothing here that needs per-slot padded storage — target_bucket is
already legible directly from spatial_conditions' highlight (the model reads
it the same way it reads everything else about the puzzle), and eval needs
only a single (ball_start_x, target_bucket) pair per instance, exposed via
physics_spec_for(puzzle_id) (mirrors SteinerTreeDataset.optimal_length_for /
PolygonDataset.optimal_area_for's lookup-by-puzzle_id pattern) rather than a
new DataSample field.
"""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples
from datasets.ball_drop_generation import (
    BALL_START_Y_FRAC,
    BALL_RADIUS_FRAC,
    DIVIDER_HEIGHT_FRAC,
    DIVIDER_X_FRAC,
    FLOOR_Y_FRAC,
    bucket_bounds_for,
)

COLOR_BG = (0, 0, 0)
COLOR_STRUCTURE = (160, 160, 160)
COLOR_TARGET = (255, 220, 0)
COLOR_BALL = (0, 220, 0)
COLOR_LINE = (230, 30, 30)


def render_ball_drop(
    ball_start_x: float,
    lines: list,
    target_bucket: int,
    image_size: int = 128,
    draw_lines: bool = True,
    line_width: int = 2,
    structure_width: int = 2,
    ball_radius_px: Optional[int] = None,
) -> np.ndarray:
    """Render a single RGB image with the fixed floor/divider structure, the
    target bucket's floor segment highlighted, the ball's start marker, and
    (if draw_lines) the solution ramps.

    `lines` are [x1, y1, x2, y2] normalized-[0,1] quadruples, as stored by
    ball_drop_generation.py / BallDropDataset.__getitem__.
    """
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img[:] = COLOR_BG
    S = image_size - 1

    def px(x_frac, y_frac):
        row = int(round((1.0 - y_frac) * S))  # y-up physics -> row-major image (row 0 = top)
        col = int(round(x_frac * S))
        return col, row

    bounds = bucket_bounds_for(1.0)  # normalized bucket boundaries, e.g. [0, .25, .5, .75, 1.0]
    floor_y = FLOOR_Y_FRAC
    for b in range(4):
        color = COLOR_TARGET if b == target_bucket else COLOR_STRUCTURE
        p1 = px(bounds[b], floor_y)
        p2 = px(bounds[b + 1], floor_y)
        cv2.line(img, p1, p2, color, structure_width)

    div_top = floor_y + DIVIDER_HEIGHT_FRAC
    for xf in DIVIDER_X_FRAC:
        cv2.line(img, px(xf, floor_y), px(xf, div_top), COLOR_STRUCTURE, structure_width)

    ball_radius_px = ball_radius_px if ball_radius_px is not None else max(2, round(BALL_RADIUS_FRAC * image_size))
    cv2.circle(img, px(ball_start_x, BALL_START_Y_FRAC), ball_radius_px, COLOR_BALL, thickness=-1)

    if draw_lines:
        for x1, y1, x2, y2 in lines:
            cv2.line(img, px(x1, y1), px(x2, y2), COLOR_LINE, line_width)

    return ((img.astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1)  # (3, H, W) in [-1, 1]


class BallDropDataset(Dataset):
    """
    Args:
        ndjson_path: path to a file produced by datasets/ball_drop_generation.py.
            If None, downloads `hf_filename` from `hf_repo` instead — the
            default in configs/data/ball_drop.yaml, so training works on a
            fresh machine with no local data/ directory (generation is a
            one-time offline step, not something training re-runs).
        hf_repo, hf_filename: HuggingFace dataset repo/file to download from
            when ndjson_path is None.
        image_size, line_width, structure_width: rendering parameters.
    """

    def __init__(
        self,
        ndjson_path: Optional[str] = None,
        hf_repo: str = "pskiers/trm-diffusion-ball-drop",
        hf_filename: str = "train.ndjson",
        image_size: int = 128,
        line_width: int = 2,
        structure_width: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.line_width = line_width
        self.structure_width = structure_width

        if ndjson_path is None:
            from huggingface_hub import hf_hub_download

            ndjson_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_filename)

        self.instances: list[dict] = []
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))
        self._spec_by_id = {
            inst["instance_id"]: {"ball_start_x": inst["ball_start_x"], "target_bucket": inst["target_bucket"]}
            for inst in self.instances
        }

    def __len__(self) -> int:
        return len(self.instances)

    def physics_spec_for(self, puzzle_id: int) -> Optional[dict]:
        """{"ball_start_x": float in [0,1], "target_bucket": int in [0,3]}
        for this instance, as recorded at generation time — used by
        eval/ball_drop_eval.py to re-simulate the exact same drop with the
        model's extracted lines, without re-deriving it from pixels."""
        return self._spec_by_id.get(int(puzzle_id))

    def __getitem__(self, idx: int) -> DataSample:
        inst = self.instances[idx]
        ball_start_x = inst["ball_start_x"]
        target_bucket = inst["target_bucket"]
        lines = inst["lines"]

        cond_img = render_ball_drop(
            ball_start_x, lines, target_bucket, self.image_size, draw_lines=False,
            line_width=self.line_width, structure_width=self.structure_width,
        )
        full_img = render_ball_drop(
            ball_start_x, lines, target_bucket, self.image_size, draw_lines=True,
            line_width=self.line_width, structure_width=self.structure_width,
        )

        return DataSample(
            images=torch.from_numpy(full_img),
            spatial_conditions=torch.from_numpy(cond_img),
            puzzle_id=torch.tensor(inst["instance_id"], dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)
