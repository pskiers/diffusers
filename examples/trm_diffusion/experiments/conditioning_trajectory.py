"""
experiments/conditioning_trajectory.py — painter denoising trajectory with the
TRM's conditioning shown alongside it, split into solved vs failed puzzles.

Per puzzle, one real (CFG) sampling pass is captured; for each recorded denoising
step we save two things stacked into a filmstrip:
  row 1: the painter's x0 estimate      (TrajectoryRecorder.pred_original_sample)
  row 2: the conditioning energy(t)     (per-cell L2 of the post-bridge (11,12,12)
         map, NEAREST-overlaid on the maze)  -- omitted for models with no thinker
         (e.g. the DiT baseline, which then yields the plain x0 trajectory).

Works for both:
    TRM  : experiment=amaze_thinker_v2_controlnet +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt
    DiT  : experiment=amaze_dit_maze             +checkpoint=runs/dit_maze_final/checkpoint_final.pt

Usage:
    python experiments/conditioning_trajectory.py \
      experiment=amaze_thinker_v2_controlnet \
      +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +task=maze +trajectory_combo=square_n7 \
      +n_each=10 [+trajectory_num_steps=8] [+wandb_run_id=<id>]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from experiments.conditioning_lib import (
    TRM_ROOT,
    ConditioningCapture,
    energy_map,
    load_model,
    overlay_grid_on_maze,
    select_good_bad,
    to_hwc_uint8,
    wandb_attach,
)
from experiments.sample_amaze_metrics import _build_amaze_dataset, _require_test_parquet
from experiments.sample_amaze_trajectory import _resolve_combo, _select_steps
from models.sampling import TrajectoryRecorder

logger = get_logger(__name__, log_level="INFO")


def _stacked_filmstrip(rows: list[tuple[str, list[tuple[str, np.ndarray]]]], path: Path,
                       pad: int = 4, label_h: int = 14, row_label_w: int = 46) -> None:
    """Compose labelled rows of tiles into one image (top col labels + left row labels)."""
    n_cols = max(len(tiles) for _lbl, tiles in rows)
    tile_h = max(t.shape[0] for _lbl, tiles in rows for _l, t in tiles)
    tile_w = max(t.shape[1] for _lbl, tiles in rows for _l, t in tiles)
    W = row_label_w + n_cols * tile_w + (n_cols + 1) * pad
    H = label_h + len(rows) * tile_h + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci in range(max(len(tiles) for _l, tiles in rows)):
        # column header taken from the first row that has this column
        for _lbl, tiles in rows:
            if ci < len(tiles):
                x = row_label_w + pad + ci * (tile_w + pad)
                draw.text((x, 1), tiles[ci][0], fill=(0, 0, 0))
                break
    for ri, (row_label, tiles) in enumerate(rows):
        y = label_h + pad + ri * (tile_h + pad)
        draw.text((1, y + tile_h // 2), row_label, fill=(0, 0, 0))
        for ci, (_l, tile) in enumerate(tiles):
            x = row_label_w + pad + ci * (tile_w + pad)
            canvas.paste(Image.fromarray(tile).convert("RGB"), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def _capture(model, conditions, device, k_steps: int, seed: int):
    """Run one sampling pass; return recorded x0 per step and (if a thinker exists)
    the conditional-pass conditioning per denoising step."""
    pipeline = model.sampling_pipeline
    steps = _select_steps(pipeline.num_inference_steps, k_steps)
    recorder = TrajectoryRecorder(steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    with ConditioningCapture(model) as cap:
        final = pipeline.sample_one_batch(model, conditions, device, generator=gen, recorder=recorder)
        cond_all = list(cap.spatial)
        enabled = cap.enabled
    records = sorted(recorder.records, key=lambda r: r["step"])
    cond_per_step = []
    if enabled and cond_all:
        per_step = max(1, len(cond_all) // pipeline.num_inference_steps)
        cond_per_step = cond_all[::per_step]                      # one (B,11,12,12) per denoise step
    return records, cond_per_step, final


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", "maze")
    if checkpoint is None or task != "maze":
        print("ERROR: needs +checkpoint=<model.pt> +task=maze", file=sys.stderr)
        sys.exit(1)

    n_each = int(cfg.get("n_each", 10))
    k_steps = int(cfg.get("trajectory_num_steps", 8))
    seed = int(cfg.get("trajectory_seed", 0))

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    parquet, combo_label = _resolve_combo(task, data_root, cfg.get("trajectory_combo", None))
    _require_test_parquet(parquet, task)
    out_root = Path(cfg.get("trajectory_out", str(Path(checkpoint).parent / "conditioning" / "trajectory" / combo_label)))

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    model = load_model(cfg, checkpoint)
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()
    has_thinker = getattr(model, "thinker_painter_translator", None) is not None

    ds = _build_amaze_dataset(cfg, str(parquet))
    logger.info(f"[{combo_label}] selecting {n_each} solved + {n_each} failed of {len(ds)} puzzles"
                f" (thinker={'yes' if has_thinker else 'no (x0-only)'})...")
    good, bad = select_good_bad(model, ds, device, n_each, seed=seed)
    order = [("good", i) for i in good] + [("bad", i) for i in bad]
    if not order:
        logger.warning("No puzzles selected; nothing to do.")
        return
    puzzles = [ds[i] for _g, i in order]
    conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)

    records, cond_per_step, final = _capture(model, conditions, device, k_steps, seed)
    x0_by_step = {r["step"]: model.decode_for_eval(r["x0_pred"].to(device)).cpu() for r in records}
    step_ids = [r["step"] for r in records]
    step_ts = {r["step"]: r["t"] for r in records}
    final_imgs = model.decode_for_eval(final).cpu()

    rows_for_wandb = []
    for pi, (group, idx) in enumerate(order):
        p = puzzles[pi]
        maze = to_hwc_uint8(p.spatial_conditions)
        gt = to_hwc_uint8(p.images)

        x0_tiles = [(f"t={step_ts[s]}", to_hwc_uint8(x0_by_step[s][pi])) for s in step_ids]
        x0_tiles = [("cond", maze)] + x0_tiles + [("final", to_hwc_uint8(final_imgs[pi])), ("gt", gt)]
        film_rows = [("x0", x0_tiles)]

        if has_thinker and cond_per_step:
            emaps = [energy_map(cond_per_step[s][pi]) for s in step_ids]
            vmax = float(max(e.max() for e in emaps)) if emaps else 1.0
            cond_tiles = [(f"t={step_ts[s]}", overlay_grid_on_maze(e, maze, vmin=0.0, vmax=vmax))
                          for s, e in zip(step_ids, emaps)]
            cond_tiles = [("cond", maze)] + cond_tiles + [("final", maze), ("gt", gt)]
            film_rows.append(("TRM", cond_tiles))

        pdir = out_root / group / f"puzzle{pi:02d}_ds{idx}"
        _stacked_filmstrip(film_rows, pdir / "filmstrip.png")
        rows_for_wandb.append({"group": group, "idx": idx, "filmstrip": pdir / "filmstrip.png"})

    logger.info(f"Saved {len(order)} trajectory filmstrips -> {out_root}")

    if accelerator.is_main_process:
        run = wandb_attach(cfg, str(checkpoint), logger)
        if run is not None:
            import wandb
            try:
                run.log({
                    f"conditioning/trajectory/{combo_label}/filmstrips": [
                        wandb.Image(str(r["filmstrip"]), caption=f"{r['group']}#{r['idx']}")
                        for r in rows_for_wandb
                    ]
                })
                logger.info(f"wandb: logged trajectory filmstrips ({combo_label}).")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"wandb: logging failed ({e!r}); PNGs saved on disk.")
            finally:
                wandb.finish()


if __name__ == "__main__":
    main()
