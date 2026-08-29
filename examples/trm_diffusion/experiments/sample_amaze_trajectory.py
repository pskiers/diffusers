from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from eval.checkpoint_utils import load_checkpoint as _load_checkpoint
from factory import build_model
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.sampling import TrajectoryRecorder
from eval.amaze_eval import make_wandb_image as _make_wandb_image
from experiments.sample_amaze_metrics import (
    TRM_ROOT,
    _build_amaze_dataset,
    _require_test_parquet,
)

logger = get_logger(__name__, log_level="INFO")


def _resolve_combo(task: str, data_root: Path, combo: str | None) -> tuple[Path, str]:
    """Map a combo string to (test parquet, label).

    maze:   "<geometry>_n<scale>" (default "square_n9")
    queens: "n<scale>"            (default "n8")
    """
    if task == "maze":
        combo = combo or "square_n9"
        geometry, ntok = combo.split("_n")
        scale = int(ntok)
        parquet = data_root / "test_maze" / geometry / f"n{scale}_{geometry}_test.parquet"
        return parquet, f"{geometry}_n{scale}"
    combo = combo or "n8"
    scale = int(combo.lstrip("n"))
    parquet = data_root / "test_queens" / f"n{scale}_test.parquet"
    return parquet, f"n{scale}"


def _select_steps(num_inference_steps: int, k: int) -> set:
    """Pick k evenly-spaced inference-step indices (always incl. first + last)."""
    if k is None or k <= 0 or k >= num_inference_steps:
        return set(range(num_inference_steps))
    if k == 1:
        return {num_inference_steps - 1}
    return {round(i * (num_inference_steps - 1) / (k - 1)) for i in range(k)}


def _to_uint8_hwc(tensor):
    import numpy as np

    arr = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return (arr * 255).astype("uint8")


def _save_png(tensor, path: Path) -> None:
    from PIL import Image

    Image.fromarray(_to_uint8_hwc(tensor)).convert("RGB").save(path)


def _filmstrip(labeled_imgs: list[tuple[str, "torch.Tensor"]], path: Path,
               pad: int = 4, label_h: int = 14) -> None:
    """Horizontally concatenate labelled tiles into one strip for quick eyeballing."""
    from PIL import Image, ImageDraw

    tiles = [(label, Image.fromarray(_to_uint8_hwc(t)).convert("RGB")) for label, t in labeled_imgs]
    if not tiles:
        return
    h = max(im.height for _, im in tiles)
    w = sum(im.width for _, im in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (w, h + label_h + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, im in tiles:
        draw.text((x, 1), label, fill=(0, 0, 0))
        canvas.paste(im, (x, label_h))
        x += im.width + pad
    canvas.save(path)


@torch.no_grad()
def capture_trajectory(model, ds, task: str, device, cfg: DictConfig, out_root: Path, combo_label: str):
    """Run one denoising pass capturing intermediate x0 estimates, save per-step
    PNGs + a labelled filmstrip per puzzle, and return rows for wandb logging."""
    n_puzzles = min(int(cfg.get("trajectory_puzzles", 4)), len(ds))
    k = int(cfg.get("trajectory_num_steps", 8))
    seed = int(cfg.get("trajectory_seed", 0))
    capture_xt = str(cfg.get("trajectory_capture_xt", False)).lower() in ("1", "true", "yes")

    puzzles = [ds[i] for i in range(n_puzzles)]
    conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)

    pipeline = model.sampling_pipeline
    steps = _select_steps(pipeline.num_inference_steps, k)
    recorder = TrajectoryRecorder(steps=steps, capture_xt=capture_xt)
    generator = torch.Generator(device=device).manual_seed(seed)
    final = pipeline.sample_one_batch(model, conditions, device, generator=generator, recorder=recorder)

    records = sorted(recorder.records, key=lambda r: r["step"])
    decoded = []
    for r in records:
        entry = {"step": r["step"], "t": r["t"], "x0": model.decode_for_eval(r["x0_pred"].to(device)).cpu()}
        if capture_xt:
            entry["xt"] = model.decode_for_eval(r["x_t"].to(device)).cpu()
        decoded.append(entry)
    final_imgs = model.decode_for_eval(final).cpu()

    cond_imgs = [p.spatial_conditions for p in puzzles]
    gt_imgs = [p.images for p in puzzles]

    combo_dir = out_root / combo_label
    rows = []
    for pi in range(n_puzzles):
        pdir = combo_dir / f"puzzle{pi:02d}"
        pdir.mkdir(parents=True, exist_ok=True)

        labeled: list[tuple[str, torch.Tensor]] = []
        if cond_imgs[pi] is not None:
            _save_png(cond_imgs[pi], pdir / "condition.png")
            labeled.append(("cond", cond_imgs[pi]))
        for d in decoded:
            img = d["x0"][pi]
            _save_png(img, pdir / f"step{d['step']:02d}_t{d['t']:04d}.png")
            labeled.append((f"t={d['t']}", img))
            if capture_xt:
                _save_png(d["xt"][pi], pdir / f"step{d['step']:02d}_t{d['t']:04d}_xt.png")
        _save_png(final_imgs[pi], pdir / "final.png")
        labeled.append(("final", final_imgs[pi]))
        if gt_imgs[pi] is not None:
            _save_png(gt_imgs[pi], pdir / "gt.png")
            labeled.append(("gt", gt_imgs[pi]))
        _filmstrip(labeled, pdir / "filmstrip.png")

        rows.append({
            "puzzle": pi,
            "condition": cond_imgs[pi],
            "steps": [(d["t"], d["x0"][pi]) for d in decoded],
            "final": final_imgs[pi],
            "gt": gt_imgs[pi],
            "filmstrip": pdir / "filmstrip.png",
        })

    logger.info(f"Trajectory: saved {n_puzzles} puzzle(s) x {len(decoded)} step(s) -> {combo_dir}")
    return decoded, rows


def _log_wandb(cfg: DictConfig, checkpoint: str, task: str, combo_label: str, decoded: list, rows: list) -> None:
    """Attach to the training run (same id file as metrics) and log the trajectory
    table + filmstrips. Never blocks or crashes — the PNGs are already on disk."""
    project = cfg.run.get("wandb_project", None)
    explicit_id = cfg.get("wandb_run_id", None)
    id_file = Path(checkpoint).parent / "wandb_run_id.txt"
    run_id = str(explicit_id) if explicit_id else (id_file.read_text().strip() if id_file.exists() else None)
    if not project or not run_id:
        logger.info(f"wandb: skipping (need run.wandb_project and a run id via +wandb_run_id= or {id_file}).")
        return

    import wandb

    init_timeout = int(cfg.get("wandb_init_timeout", 60))
    try:
        run = wandb.init(project=project, id=run_id, resume="allow",
                         settings=wandb.Settings(init_timeout=init_timeout))
    except Exception as e:
        logger.warning(f"wandb: init failed/timed out after {init_timeout}s ({e!r}), skipping.")
        return

    try:
        prefix = f"trajectory/{task}/{combo_label}"
        step_cols = [f"t={d['t']}" for d in decoded]
        columns = ["puzzle", "condition", *step_cols, "final", "gt"]
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(
                row["puzzle"],
                _make_wandb_image(row["condition"]),
                *[_make_wandb_image(img) for _, img in row["steps"]],
                _make_wandb_image(row["final"]),
                _make_wandb_image(row["gt"]),
            )
        run.log({
            f"{prefix}/table": table,
            f"{prefix}/filmstrips": [
                wandb.Image(str(row["filmstrip"]), caption=f"puzzle{row['puzzle']}") for row in rows
            ],
        })
        logger.info(f"wandb: logged trajectory ({task}/{combo_label}) into run {run_id}.")
    except Exception as e:
        logger.warning(f"wandb: logging failed ({e!r}); PNGs already saved on disk.")
    finally:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", None)
    if checkpoint is None or task not in ("maze", "queens"):
        print(
            "ERROR: usage:\n"
            "  python experiments/sample_amaze_trajectory.py experiment=amaze_dit_maze \\\n"
            "    +checkpoint=<model.pt> +task=maze|queens [+trajectory_combo=square_n9] \\\n"
            "    [+trajectory_puzzles=4] [+trajectory_num_steps=8] [+trajectory_capture_xt=false]",
            file=sys.stderr,
        )
        sys.exit(1)

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    parquet, combo_label = _resolve_combo(task, data_root, cfg.get("trajectory_combo", None))
    _require_test_parquet(parquet, task)

    out_root = Path(cfg.get("trajectory_out", str(Path(checkpoint).parent / "trajectory" / task)))

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    ds = _build_amaze_dataset(cfg, str(parquet))
    logger.info(f"[{task}/{combo_label}] trajectory over {len(ds)} puzzles (using first "
                f"{int(cfg.get('trajectory_puzzles', 4))}).")

    decoded, rows = capture_trajectory(model, ds, task, device, cfg, out_root, combo_label)

    if accelerator.is_main_process:
        _log_wandb(cfg, str(checkpoint), task, combo_label, decoded, rows)


if __name__ == "__main__":
    main()
