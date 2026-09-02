"""
experiments/conditioning_heatmaps.py — visualise the TRM's conditioning on the
12x12 grid as heatmaps overlaid on the maze, for solved vs failed puzzles.

Two views per puzzle (see the module docstring of conditioning_lib.py):
  1. energy: per-cell L2 norm of the post-bridge (11,12,12) map (Tap2) — "where
     is the TRM injecting signal". Overlaid on the maze, NEAREST x12.
  2. channels: the 11 raw-logit channel maps (Tap1) as a montage, each on its own
     colour scale — shows which of the 11 channels are alive vs near-dead.

Captured TEACHER-FORCED at fixed noise level(s) t (the GT solution noised to t):
high t => x_noisy leaks no answer, so structure in the map is reasoned, not copied.

Usage (maze; attach to the training run for wandb):
    python experiments/conditioning_heatmaps.py \
      experiment=amaze_thinker_v2_controlnet \
      +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +task=maze +trajectory_combo=square_n7 \
      +n_each=10 [+heatmap_fracs=0.9,0.5,0.1] [+wandb_run_id=<id>]
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
from PIL import Image

from experiments.conditioning_lib import (
    TRM_ROOT,
    capture_teacher_forced,
    channel_montage,
    energy_map,
    load_model,
    overlay_grid_on_maze,
    select_good_bad,
    to_hwc_uint8,
    wandb_attach,
)
from experiments.sample_amaze_metrics import _build_amaze_dataset, _require_test_parquet
from experiments.sample_amaze_trajectory import _resolve_combo

logger = get_logger(__name__, log_level="INFO")


def _save(arr_uint8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8).convert("RGB").save(path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", "maze")
    if checkpoint is None or task != "maze":
        print("ERROR: needs +checkpoint=<thinker.pt> +task=maze", file=sys.stderr)
        sys.exit(1)

    hf = cfg.get("heatmap_fracs", "0.9,0.5,0.1")
    fracs = [float(x) for x in (hf.split(",") if isinstance(hf, str) else hf)]
    n_each = int(cfg.get("n_each", 10))
    seed = int(cfg.get("seed", 0))

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    parquet, combo_label = _resolve_combo(task, data_root, cfg.get("trajectory_combo", None))
    _require_test_parquet(parquet, task)
    out_root = Path(cfg.get("heatmap_out", str(Path(checkpoint).parent / "conditioning" / "heatmaps" / combo_label)))

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    model = load_model(cfg, checkpoint)
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()
    if getattr(model, "thinker_painter_translator", None) is None:
        print("ERROR: this model has no thinker_painter_translator (not a TRM painter-thinker).", file=sys.stderr)
        sys.exit(1)

    ds = _build_amaze_dataset(cfg, str(parquet))
    logger.info(f"[{combo_label}] selecting {n_each} solved + {n_each} failed of {len(ds)} puzzles...")
    good, bad = select_good_bad(model, ds, device, n_each, seed=seed)
    logger.info(f"[{combo_label}] got {len(good)} solved, {len(bad)} failed.")

    groups = [("good", good), ("bad", bad)]
    order = [(g, i) for g, idxs in groups for i in idxs]
    if not order:
        logger.warning("No puzzles selected; nothing to do.")
        return
    puzzles = [ds[i] for _g, i in order]
    conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)
    T = model.scheduler.config.num_train_timesteps

    # Capture the conditioning for the whole selection at each noise level.
    caps: dict[float, dict] = {}
    for frac in fracs:
        t = int(frac * (T - 1))
        caps[frac] = capture_teacher_forced(model, conditions, t, device, seed=seed)

    # Shared energy colour scale per frac so solved/failed are comparable.
    vmax = {frac: float(energy_map(caps[frac]["spatial"]).max()) for frac in fracs}

    rows = []
    for pi, (group, idx) in enumerate(order):
        p = puzzles[pi]
        maze = to_hwc_uint8(p.spatial_conditions)
        gt = to_hwc_uint8(p.images)
        pdir = out_root / group / f"puzzle{pi:02d}_ds{idx}"
        _save(maze, pdir / "maze.png")
        _save(gt, pdir / "gt.png")

        energy_imgs = {}
        for frac in fracs:
            t = int(frac * (T - 1))
            emap = energy_map(caps[frac]["spatial"][pi])          # (12,12)
            ov = overlay_grid_on_maze(emap, maze, vmin=0.0, vmax=vmax[frac])
            _save(ov, pdir / f"energy_t{t:04d}.png")
            energy_imgs[frac] = ov

        # heatmap-2: 11 raw-logit channels (Tap1) at the highest-noise frac.
        hi = max(fracs)
        logits = caps[hi]["logits"][pi]                            # (144, 11)
        spatial11 = logits.transpose(0, 1).reshape(11, 12, 12)     # (11,12,12)
        montage = channel_montage(spatial11)
        _save(montage, pdir / "channels_raw.png")

        rows.append({"group": group, "idx": idx, "maze": maze, "gt": gt,
                     "energy": energy_imgs, "channels": montage})
    logger.info(f"Saved heatmaps for {len(order)} puzzles -> {out_root}")

    if accelerator.is_main_process:
        _log_wandb(cfg, str(checkpoint), combo_label, fracs, T, rows)


def _log_wandb(cfg, checkpoint, combo_label, fracs, T, rows) -> None:
    run = wandb_attach(cfg, checkpoint, logger)
    if run is None:
        return
    import wandb

    try:
        frac_cols = [f"energy_t{int(f * (T - 1)):04d}" for f in fracs]
        columns = ["group", "ds_idx", "maze", *frac_cols, "channels_raw", "gt"]
        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r["group"], r["idx"], wandb.Image(r["maze"]),
                *[wandb.Image(r["energy"][f]) for f in fracs],
                wandb.Image(r["channels"]), wandb.Image(r["gt"]),
            )
        run.log({f"conditioning/heatmaps/{combo_label}/table": table})
        logger.info(f"wandb: logged conditioning heatmaps ({combo_label}).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb: logging failed ({e!r}); PNGs saved on disk.")
    finally:
        import wandb as _wb
        _wb.finish()


if __name__ == "__main__":
    main()
