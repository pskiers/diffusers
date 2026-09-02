"""
experiments/conditioning_swap.py — feed maze-B's TRM conditioning onto maze-A's
canvas and see whether the painter draws B's solution.

If A's painter, steered by B's conditioning, produces a blue path closer to B's
solution than to A's, the conditioning carries the *specific* solution (not a
generic "draw a path"). Pairs are matched at the SAME scale n, DIFFERENT shapes.

Method (static swap):
  1. Run B's own generation, capturing B's conditional conditioning per denoising
     step (the post-bridge (11,12,12) map).
  2. Run A's generation twice from identical initial noise: once normally (A->A),
     once with the translator's conditional output OVERRIDDEN by B's captured map
     at each step (Bcond->A). The only difference is the conditioning.
  3. Compare the blue path (pixel IoU) of Bcond->A against B's and A's GT paths.

Usage:
    python experiments/conditioning_swap.py \
      experiment=amaze_thinker_v2_controlnet \
      +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +task=maze +swap_n=7 +swap_shape_a=square +swap_shape_b=hexagon \
      +n_pairs=10 [+wandb_run_id=<id>]
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

from experiments.conditioning_lib import TRM_ROOT, ConditioningCapture, load_model, to_hwc_uint8, wandb_attach
from experiments.sample_amaze_metrics import _build_amaze_dataset, _require_test_parquet
from experiments.sample_amaze_trajectory import _resolve_combo

logger = get_logger(__name__, log_level="INFO")


class ConditioningInjector:
    """Override the translator's conditional-pass output with a pre-captured
    (P,11,12,12) map per denoising step; the CFG unconditional pass is left intact
    so guidance still contrasts injected-conditioning vs the painter's own null.
    """

    def __init__(self, model, cond_per_step: list[torch.Tensor], per_step: int):
        self.translator = model.thinker_painter_translator
        self.cond = cond_per_step
        self.per_step = per_step
        self._orig = None

    def __enter__(self):
        orig = self.translator._logits_to_spatial
        self._orig = orig
        cond, per_step, st = self.cond, self.per_step, {"c": 0}

        def _patched(logits):
            idx = st["c"]
            st["c"] += 1
            step, is_cond = idx // per_step, (idx % per_step == 0)
            if is_cond and step < len(cond):
                return cond[step].to(device=logits.device).float()
            return orig(logits)

        self.translator._logits_to_spatial = _patched
        return self

    def __exit__(self, *exc):
        try:
            del self.translator._logits_to_spatial
        except AttributeError:
            self.translator._logits_to_spatial = self._orig
        return False


def _blue_mask(img_uint8: np.ndarray) -> np.ndarray:
    r, g, b = img_uint8[..., 0].astype(int), img_uint8[..., 1].astype(int), img_uint8[..., 2].astype(int)
    return (b > 120) & (r < 110) & (g < 130)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 0.0


@torch.no_grad()
def _capture_B(model, conditions_B, device, seed):
    pipeline = model.sampling_pipeline
    gen = torch.Generator(device=device).manual_seed(seed)
    with ConditioningCapture(model) as cap:
        final = pipeline.sample_one_batch(model, conditions_B, device, generator=gen)
        spatial = list(cap.spatial)
    per_step = max(1, len(spatial) // pipeline.num_inference_steps)
    return spatial[::per_step], per_step, final


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", "maze")
    if checkpoint is None or task != "maze":
        print("ERROR: needs +checkpoint=<thinker.pt> +task=maze", file=sys.stderr)
        sys.exit(1)

    n = int(cfg.get("swap_n", 7))
    shape_a = str(cfg.get("swap_shape_a", "square"))
    shape_b = str(cfg.get("swap_shape_b", "hexagon"))
    n_pairs = int(cfg.get("n_pairs", 10))
    seed = int(cfg.get("seed", 0))

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    pq_a, label_a = _resolve_combo(task, data_root, f"{shape_a}_n{n}")
    pq_b, label_b = _resolve_combo(task, data_root, f"{shape_b}_n{n}")
    _require_test_parquet(pq_a, task)
    _require_test_parquet(pq_b, task)
    out_root = Path(cfg.get("swap_out", str(Path(checkpoint).parent / "conditioning" / "swap" / f"{label_a}__{label_b}")))

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    model = load_model(cfg, checkpoint)
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()
    if getattr(model, "thinker_painter_translator", None) is None:
        print("ERROR: swap needs a TRM painter-thinker (no translator found).", file=sys.stderr)
        sys.exit(1)

    ds_a = _build_amaze_dataset(cfg, str(pq_a))
    ds_b = _build_amaze_dataset(cfg, str(pq_b))
    P = min(n_pairs, len(ds_a), len(ds_b))
    puzzles_a = [ds_a[i] for i in range(P)]
    puzzles_b = [ds_b[i] for i in range(P)]
    conditions_a = model._batch_to_sample(ds_a.collate_fn(puzzles_a), device)
    conditions_b = model._batch_to_sample(ds_b.collate_fn(puzzles_b), device)

    logger.info(f"[swap {label_a} <- {label_b}] capturing B conditioning for {P} pairs...")
    cond_b, per_step, final_b = _capture_B(model, conditions_b, device, seed + 1)

    pipeline = model.sampling_pipeline
    gen_a = torch.Generator(device=device).manual_seed(seed)
    out_aa = pipeline.sample_one_batch(model, conditions_a, device, generator=gen_a)
    gen_a2 = torch.Generator(device=device).manual_seed(seed)  # identical noise
    with ConditioningInjector(model, cond_b, per_step):
        out_ba = pipeline.sample_one_batch(model, conditions_a, device, generator=gen_a2)

    dec_aa = model.decode_for_eval(out_aa).cpu()
    dec_ba = model.decode_for_eval(out_ba).cpu()
    dec_bb = model.decode_for_eval(final_b).cpu()

    rows = []
    iou_to_b, iou_to_a = [], []
    for pi in range(P):
        maze_a = to_hwc_uint8(puzzles_a[pi].spatial_conditions)
        maze_b = to_hwc_uint8(puzzles_b[pi].spatial_conditions)
        gt_a = to_hwc_uint8(puzzles_a[pi].images)
        gt_b = to_hwc_uint8(puzzles_b[pi].images)
        img_aa = to_hwc_uint8(dec_aa[pi])
        img_ba = to_hwc_uint8(dec_ba[pi])
        img_bb = to_hwc_uint8(dec_bb[pi])

        swap_blue = _blue_mask(img_ba)
        ib = _iou(swap_blue, _blue_mask(gt_b))
        ia = _iou(swap_blue, _blue_mask(gt_a))
        iou_to_b.append(ib)
        iou_to_a.append(ia)

        strip = _compose([("A maze", maze_a), ("A->A", img_aa), ("Bcond->A", img_ba),
                          ("B->B", img_bb), ("B maze", maze_b)],
                         caption=f"IoU(swap,B)={ib:.2f}  IoU(swap,A)={ia:.2f}")
        pdir = out_root / f"pair{pi:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strip).save(pdir / "swap.png")
        rows.append({"pair": pi, "iou_to_b": ib, "iou_to_a": ia, "img": pdir / "swap.png"})

    mib = float(np.mean(iou_to_b)) if iou_to_b else 0.0
    mia = float(np.mean(iou_to_a)) if iou_to_a else 0.0
    logger.info(f"[swap {label_a} <- {label_b}] mean blue-IoU(swap,B)={mib:.3f} vs (swap,A)={mia:.3f} "
                f"-> {'carries B (specific solution)' if mib > mia else 'closer to A / generic'}")

    if accelerator.is_main_process:
        run = wandb_attach(cfg, str(checkpoint), logger)
        if run is not None:
            import wandb
            try:
                table = wandb.Table(columns=["pair", "iou_to_B", "iou_to_A", "swap"])
                for r in rows:
                    table.add_data(r["pair"], r["iou_to_b"], r["iou_to_a"], wandb.Image(str(r["img"])))
                run.log({
                    f"conditioning/swap/{label_a}__{label_b}/table": table,
                    f"conditioning/swap/{label_a}__{label_b}/mean_iou_to_B": mib,
                    f"conditioning/swap/{label_a}__{label_b}/mean_iou_to_A": mia,
                })
                logger.info("wandb: logged swap results.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"wandb: logging failed ({e!r}); PNGs saved on disk.")
            finally:
                wandb.finish()


def _compose(tiles: list[tuple[str, np.ndarray]], caption: str = "", pad: int = 4, label_h: int = 14) -> np.ndarray:
    h = max(t.shape[0] for _l, t in tiles)
    w = sum(t.shape[1] for _l, t in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (w, h + label_h + pad + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, tile in tiles:
        draw.text((x, 1), label, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(tile).convert("RGB"), (x, label_h))
        x += tile.shape[1] + pad
    if caption:
        draw.text((pad, h + label_h + pad), caption, fill=(0, 0, 0))
    return np.asarray(canvas)


if __name__ == "__main__":
    main()
