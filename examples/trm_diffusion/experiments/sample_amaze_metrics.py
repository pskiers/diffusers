from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


# For load_checkpoint function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import (
    MAZE_GEOMETRIES,
    MAZE_OOD_SCALES,
    MAZE_SCALES,
    QUEEN_OOD_SCALES,
    QUEEN_SCALES,
    build_maze_result,
    build_queens_result,
    log_tables,
    maze_sample_key,
    queens_sample_key,
)
from eval.checkpoint_utils import load_checkpoint as _load_checkpoint
from factory import build_model
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


logger = get_logger(__name__, log_level="INFO")

TRM_ROOT = Path(__file__).resolve().parent.parent


def _require_test_parquet(parquet: Path, task: str) -> None:
    """Fail loudly if the canonical (flat) test parquet is missing. This script NEVER generates data — that's scripts/gen_amaze.py's job."""
    if not parquet.exists():
        raise FileNotFoundError(
            f"Canonical test set not found: {parquet}\n"
            f"Generate it first:  python scripts/gen_amaze.py test {task}"
        )


def _build_amaze_dataset(cfg: DictConfig, dataset_path: str) -> AmazeDataset:
    vd = cfg.data.val_dataset
    return AmazeDataset(
        dataset_path=dataset_path,
        split="test",
        image_size=int(vd.image_size),
        condition_field=vd.get("condition_field", "m_original_img"),
        target_field=vd.get("target_field", "sol_img"),
        num_channels=int(vd.get("num_channels", 3)),
        include_metadata=True,
    )


@torch.no_grad()
def sample_and_score(
    model, ds: AmazeDataset, task: str, device, samples_per_puzzle: int, base_seed: int, batch_size: int
) -> tuple[list[dict], dict | None]:
    """Sample `samples_per_puzzle` attempts per puzzle and score them with the AmazeMetrics evaluator.

    Returns ``(rows, sample_pair)`` where ``rows`` has one entry per puzzle:
    - first-attempt Violation/Coverage/MSE-In/MSE-Out, Pass@1 (exact solve on the first attempt)
    - Pass@K (exact solve within K attempts / best-of-K).
    ``sample_pair`` is one representative {'generated', 'condition'} image pair
    (first puzzle, first attempt) for the summary table, or None if empty.
    """
    from eval.amaze_eval import AmazeMetrics

    pipeline = model.sampling_pipeline
    scorer = AmazeMetrics(device=device, task=task)
    K = samples_per_puzzle
    pass_k_key = f"pass_at_{K}"
    puzzles_per_batch = max(1, batch_size // K)   # keep all K attempts of a puzzle in the same batch

    rows = []
    sample_pair = None
    for start in range(0, len(ds), puzzles_per_batch):
        puzzles = [ds[i] for i in range(start, min(start + puzzles_per_batch, len(ds)))]
        # Replicate each puzzle K times ([p0]*K + [p1]*K + ...): identical conditions,
        # independent initial noise -> K distinct attempts per puzzle in a single loop.
        flat = [p for p in puzzles for _ in range(K)]
        conditions = model._batch_to_sample(ds.collate_fn(flat), device)

        generator = torch.Generator(device=device).manual_seed(base_seed + start)
        generated = pipeline.sample_one_batch(model, conditions, device, generator=generator)
        decoded = model.decode_for_eval(generated).cpu()                 # (P*K, C, H, W) in [0, 1]

        P = len(puzzles)
        inputs = decoded.reshape(P, K, *decoded.shape[1:])               # (P, K, C, H, W)
        if sample_pair is None and P > 0:
            cond = puzzles[0].spatial_conditions
            sample_pair = {
                "generated": inputs[0, 0].clone(),
                "condition": cond.detach().cpu() if cond is not None else None,
            }
        metadata = [p.metadata if p.metadata is not None else {} for p in puzzles]

        # Pass@K is best-of-K: the evaluator sets pass_at_{K} = any(exact solve) over the K attempts (eval/amaze_eval.py).
        for rec in scorer.compute_and_accumulate_metrics(inputs, metadata):
            pass_at_k = rec[pass_k_key] if K > 1 else rec["pass"]
            rows.append({
                "violation": rec["background_violation"],
                "coverage": rec["gt_cell_coverage"],
                "mse_inside": rec["mse_inside"],
                "mse_outside": rec["mse_outside"],
                "pass1": rec["pass"],        # exact solve on the first attempt
                "pass5": pass_at_k,          # exact solve within K attempts (best-of-K)
            })
    return rows, sample_pair


def _print_metrics_row(label: str, agg: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Metrics — {label}")
    print("=" * 78)
    print(f"{'Violation':>10} {'Coverage':>10} {'MSE-In':>10} {'MSE-Out':>10} {'Pass@1':>10} {'Pass@5':>10}")
    print(f"{agg['violation']*100:>9.2f}% {agg['coverage']*100:>9.2f}% {agg['mse_inside']:>10.4f} "
          f"{agg['mse_outside']:>10.4f} {agg['pass1']*100:>9.2f}% {agg['pass5']*100:>9.2f}%")
    print("=" * 78)


def _print_metrics_table(label: str, named_aggs: list) -> None:
    """Print a multi-row metrics table (one row per (name, agg) pair)."""
    print("\n" + "=" * 90)
    print(f"Metrics — {label}")
    print("=" * 90)
    print(f"{'Group':>12} {'Violation':>10} {'Coverage':>10} {'MSE-In':>10} {'MSE-Out':>10} {'Pass@1':>10} {'Pass@5':>10}")
    for name, agg in named_aggs:
        print(f"{name:>12} {agg['violation']*100:>9.2f}% {agg['coverage']*100:>9.2f}% "
              f"{agg['mse_inside']:>10.4f} {agg['mse_outside']:>10.4f} "
              f"{agg['pass1']*100:>9.2f}% {agg['pass5']*100:>9.2f}%")
    print("=" * 90)


def _log_wandb(cfg: DictConfig, checkpoint: str, task: str, result: dict, samples: dict | None = None) -> None:
    """Attach to the training run's wandb run and log the metrics into the SAME panel/run as training.

    The run id comes from +wandb_run_id=<id> or from <checkpoint_dir>/wandb_run_id.txt (written by train_trm.py). Missing project
    or id -> skip with a hint (the JSON output is unaffected). The general/OOD/per-geometry/per-size tables
    themselves are built by the shared ``eval.amaze_eval.log_tables`` used by every AMAZE eval path.
    """
    project = cfg.run.get("wandb_project", None)
    explicit_id = cfg.get("wandb_run_id", None)
    id_file = Path(checkpoint).parent / "wandb_run_id.txt"
    run_id = str(explicit_id) if explicit_id else (id_file.read_text().strip() if id_file.exists() else None)
    if not project or not run_id:
        logger.info(
            f"wandb: skipping (need run.wandb_project and a run id via +wandb_run_id= or {id_file})."
        )
        return

    import wandb

    # Attaching to the training run can hang on Athena when resuming a poisoned/half-created run id
    # (see repo notes). Bound init and never let optional logging block or crash the eval — the JSON
    # results are already on disk before this runs.
    init_timeout = int(cfg.get("wandb_init_timeout", 60))
    try:
        run = wandb.init(
            project=project, id=run_id, resume="allow",
            settings=wandb.Settings(init_timeout=init_timeout),
        )
    except Exception as e:
        logger.warning(
            f"wandb: init failed/timed out after {init_timeout}s ({e!r}), skipping wandb logging "
            f"(results already saved to JSON). If the run id is poisoned, delete {id_file} and retry."
        )
        return

    try:
        log_tables(run, task, result, samples)
        logger.info(f"wandb: logged {task} metrics into run {run_id} (project {project}).")
    except Exception as e:
        logger.warning(f"wandb: logging failed ({e!r}); results already saved to JSON.")
    finally:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", None)
    if checkpoint is None or task not in ("maze", "queens"):
        print(
            "ERROR: usage:\n"
            "  python experiments/sample_amaze_metrics.py experiment=amaze_thinker_v1_controlnet \\\n"
            "    painter.checkpoint=<painter.pt> +checkpoint=<thinker.pt> +task=maze|queens",
            file=sys.stderr,
        )
        sys.exit(1)

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    samples_per_puzzle = int(cfg.get("samples_per_puzzle", 5))
    seed = int(cfg.get("sample_seed", 0))
    out_path = cfg.get("out_json", None) or str(Path(checkpoint).parent / f"amaze_metrics_{task}.json")

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    step = _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    # Pack whole puzzles + their K attempts into denoising batches of this size (defaults to the
    # sampling pipeline's own batch_size) instead of sampling one puzzle at a time.
    sample_batch_size = int(cfg.get("sample_batch_size", model.sampling_pipeline.batch_size))
    logger.info(f"Sampling with batch_size={sample_batch_size} (K={samples_per_puzzle} attempts/puzzle).")

    if task == "maze":
        def _score_maze(scales):
            per_combo, combo_samples = {}, {}
            for geometry in MAZE_GEOMETRIES:
                for scale in scales:
                    combo = data_root / "test_maze" / geometry / f"n{scale}_{geometry}_test.parquet"
                    _require_test_parquet(combo, "maze")
                    ds = _build_amaze_dataset(cfg, str(combo))
                    logger.info(f"[maze/{geometry}/{scale}] scoring {len(ds)} puzzles x{samples_per_puzzle} samples")
                    rows, sample_pair = sample_and_score(model, ds, "maze", device, samples_per_puzzle, seed, sample_batch_size)
                    per_combo[f"{geometry}_{scale}"] = rows
                    combo_samples[maze_sample_key(geometry, scale)] = sample_pair
            return per_combo, combo_samples

        per_combo, combo_samples = _score_maze(MAZE_SCALES)
        ood_combo, ood_samples = _score_maze(MAZE_OOD_SCALES)
        result = build_maze_result(per_combo, ood_combo)

        for g in MAZE_GEOMETRIES:
            _print_metrics_table(
                f"Maze — {g} (per scale)",
                [(f"{s}x{s}", result["per_shape"][g][str(s)]) for s in MAZE_SCALES],
            )
        _print_metrics_table("Maze — per geometry (general)", [(g, result["per_geometry"][g]) for g in MAZE_GEOMETRIES])
        _print_metrics_row("Maze — overall general (7 scales)", result["overall"])
        _print_metrics_table("Maze — per geometry OOD (3x3)", [(g, result["per_geometry_ood"][g]) for g in MAZE_GEOMETRIES])
        _print_metrics_row("Maze — overall OOD (3x3)", result["overall_ood"])

        samples = {**combo_samples, **ood_samples}

    else:
        def _score_queens(scales):
            per_scale_rows, scale_samples = {}, {}
            for scale in scales:
                combo = data_root / "test_queens" / f"n{scale}_test.parquet"
                _require_test_parquet(combo, "queens")
                ds = _build_amaze_dataset(cfg, str(combo))
                logger.info(f"[queens/{scale}] scoring {len(ds)} puzzles x{samples_per_puzzle} samples")
                rows, sample_pair = sample_and_score(model, ds, "queens", device, samples_per_puzzle, seed, sample_batch_size)
                per_scale_rows[str(scale)] = rows
                scale_samples[queens_sample_key(scale)] = sample_pair
            return per_scale_rows, scale_samples

        per_scale_rows, scale_samples = _score_queens(QUEEN_SCALES)
        ood_scale_rows, ood_samples = _score_queens(QUEEN_OOD_SCALES)
        result = build_queens_result(per_scale_rows, ood_scale_rows)

        _print_metrics_table("Queen — per scale", [(f"{s}x{s}", result["per_scale"][s]) for s in result["per_scale"]])
        _print_metrics_row("Queen — overall general (4..10)", result["overall"])
        _print_metrics_table("Queen — OOD per scale", [(f"{s}x{s}", result["per_scale_ood"][s]) for s in result["per_scale_ood"]])
        _print_metrics_row("Queen — overall OOD (12x12)", result["overall_ood"])

        samples = {**scale_samples, **ood_samples}

    result["checkpoint"] = str(checkpoint)
    result["step"] = step
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved -> {out_path}")

    if accelerator.is_main_process:
        _log_wandb(cfg, str(checkpoint), task, result, samples)


if __name__ == "__main__":
    main()
