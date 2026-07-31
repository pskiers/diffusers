"""
experiments/ablate_trm_loop_budget.py — Ablate the TRM thinker's reasoning
budget at inference time on a trained MNIST-Sudoku thinker+painter checkpoint.

Every denoising step currently calls ThinkerFrozenPainterBase.forward(),
which resets the TRM's recurrent carry (z_H, z_L) from scratch and runs the
full trained n_sup (x H_cycles x L_cycles) reasoning budget — even though the
puzzle condition never changes across denoising steps, only x_noisy does.
This script tests, purely at inference time (no retraining, the checkpoint
is untouched):

  static   — reset carry every denoising step (today's default), sweep n_sup.
  carry    — carry (z_H, z_L) across denoising steps instead of resetting
             every step. reset_every=1 reproduces "static"; reset_every=None
             persists for the whole trajectory; other K resets every K steps.
             NOTE: carrying is out of the training distribution (training
             only ever sees a fresh reset) — a regression here is a real,
             expected possible outcome, not just a bug.
  schedule — vary n_sup per denoising step (front-loaded / back-loaded, vs.
             flat) at matched total compute. Needs schedule_flat_n and
             schedule_reset_every set explicitly (informed by inspecting the
             static/carry results first) — run this axis in a second pass.

For every configuration, reports the same 4 accuracies as SudokuEvalCallback
(cell_acc, puzzle_acc, constraint_puzzle_acc, given_consistent_puzzle_acc)
plus a compute proxy (total_sup_calls = summed n_sup across the trajectory,
x2 under CFG since the unconditional branch reasons too) and wall-clock time.

All configs reuse the exact same cached validation batches and exact same
initial noise per sample (seeded once up front) — a paired comparison, so
differences are attributable to the reasoning-budget/carry setting alone.

Usage:
    python experiments/ablate_trm_loop_budget.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +ablation.num_samples=256

    # Second pass, once static/carry results suggest a good flat n_sup and
    # carry regime to build the schedule on top of:
    python experiments/ablate_trm_loop_budget.py \\
      experiment=mnist_thinker_v1_controlnet ... checkpoint=... \\
      +ablation.axes=[schedule] +ablation.schedule_flat_n=4 \\
      +ablation.schedule_reset_every=5

Config overrides work exactly like train_trm.py / eval.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from eval.mnist_eval import evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Duplicated from eval.py (not importable: the eval/ package in this
    same directory shadows the top-level eval.py module, so `from eval
    import ...` resolves to the package, not the script). Keep in sync with
    eval.py's version if the checkpoint format changes."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        step = ckpt.get("step", None)
        sd = strip_compiled_prefix(ckpt["model_state"])
        model.load_state_dict(sd, strict=False)

        if use_ema and ckpt.get("ema_state") is not None:
            ema_state = ckpt["ema_state"]
            if isinstance(ema_state, dict) and ema_state:
                ema_sd = strip_compiled_prefix(ema_state)
                model.load_state_dict(ema_sd, strict=False)
                logger.info(f"Loaded EMA weights on top of model_state (step={step})")
                return step
            logger.warning("EMA state is empty — using raw model_state")
        logger.info(f"Loaded model_state (step={step}, use_ema={use_ema})")
        return step

    sd = strip_compiled_prefix(ckpt)
    model.load_state_dict(sd, strict=False)
    logger.info("Loaded raw state_dict")
    return None


# ── Schedules ────────────────────────────────────────────────────────────────


def _matched_budget_schedule(direction: str, flat_n: int, num_steps: int, ratio: float = 3.0) -> list[int]:
    """Per-step n_sup ramp summing to flat_n * num_steps (matched total
    compute vs. the flat baseline), so a schedule vs. flat comparison isolates
    the effect of *reallocating* budget across time from the effect of
    changing the total amount of it.

    direction="front": high budget at step 0 (the first denoising step, at
    the highest noise level) tapering to low budget near the end.
    direction="back": the reverse ramp.
    """
    total = flat_n * num_steps
    lo = max(1, round(flat_n / ratio))
    hi = max(lo + 1, round(flat_n * ratio))
    raw = np.linspace(hi, lo, num_steps) if direction == "front" else np.linspace(lo, hi, num_steps)
    raw = raw * (total / raw.sum())
    sched = np.maximum(1, np.round(raw)).astype(int)

    drift = int(total - sched.sum())
    order = np.argsort(-sched)
    i = 0
    while drift != 0 and i < 10 * num_steps:
        j = order[i % len(order)]
        if drift > 0:
            sched[j] += 1
            drift -= 1
        elif sched[j] > 1:
            sched[j] -= 1
            drift += 1
        i += 1
    return sched.tolist()


def _make_n_sup_fn(n_sup_per_step: list[int]) -> Callable[[int, int, int], int]:
    return lambda step_idx, t, T: n_sup_per_step[step_idx]


def _make_reset_fn(reset_every: Optional[int]) -> Callable[[int], bool]:
    """reset_every=None: only step 0. Otherwise every `reset_every` steps."""
    if reset_every is None:
        return lambda step_idx: step_idx == 0
    return lambda step_idx: step_idx % reset_every == 0


# ── Sampling loop (mirrors SamplingPipeline.sample_one_batch, but threads a
#    separate TRM carry per branch across steps instead of treating each
#    denoising step as stateless) ─────────────────────────────────────────────


@torch.no_grad()
def _run_ablation_sampling(
    model,
    conditions,
    x_init: torch.Tensor,
    num_inference_steps: int,
    cfg_scale: float,
    n_sup_fn: Callable[[int, int, int], int],
    reset_fn: Callable[[int], bool],
) -> torch.Tensor:
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    T = model.scheduler.config.num_train_timesteps

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        n_sup = n_sup_fn(step_idx, int(t.item()), T)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c, n_sup=n_sup)
        noise_pred = pred_c.pred

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u, n_sup=n_sup)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return x


def _total_sup_calls(n_sup_fn: Callable[[int, int, int], int], num_steps: int, cfg_scale: float) -> int:
    total = sum(n_sup_fn(i, 0, 0) for i in range(num_steps))
    return total * (2 if cfg_scale != 1.0 else 1)


# ── Config runner ──────────────────────────────────────────────────────────


def _run_config(
    model,
    classifier,
    cell_size: int,
    cached_batches: list[dict],
    num_inference_steps: int,
    cfg_scale: float,
    n_sup_fn: Callable[[int, int, int], int],
    reset_fn: Callable[[int], bool],
) -> dict:
    all_cell, all_puzzle, all_constraint, all_given_consistent = [], [], [], []
    t0 = time.time()

    for cb in cached_batches:
        x = _run_ablation_sampling(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale, n_sup_fn, reset_fn
        )
        generated = model.decode_for_eval(x)
        acc = evaluate_grids(generated, cb["solutions"], classifier, cell_size, given_masks=cb["given_masks"])
        all_cell.append(acc["cell_acc"])
        all_puzzle.append(acc["puzzle_acc"])
        all_constraint.append(acc.get("constraint_puzzle_acc", 0.0))
        if acc.get("given_consistent_puzzle_acc") is not None:
            all_given_consistent.append(acc["given_consistent_puzzle_acc"])

    elapsed = time.time() - t0
    result = {
        "cell_acc": float(np.mean(all_cell)),
        "puzzle_acc": float(np.mean(all_puzzle)),
        "constraint_puzzle_acc": float(np.mean(all_constraint)),
        "total_sup_calls": _total_sup_calls(n_sup_fn, num_inference_steps, cfg_scale),
        "wall_time_sec": elapsed,
    }
    if all_given_consistent:
        result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent))
    return result


def _build_cached_batches(model, dataloader, device, num_samples: int, seed: int) -> list[dict]:
    """Cache a fixed set of (conditions, solutions, given_masks, x_init)
    once, reused identically across every ablation config — a paired
    comparison, so differences are attributable to the config alone."""
    torch.manual_seed(seed)
    cached = []
    n_done = 0
    for batch in dataloader:
        if n_done >= num_samples:
            break
        conditions = model._batch_to_sample(batch, device)
        solutions = batch["solution"]
        given_masks = batch.get("solution_mask")
        bsz = solutions.shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)
        cached.append({
            "conditions": conditions,
            "solutions": solutions,
            "given_masks": given_masks,
            "x_init": x_init,
        })
        n_done += bsz
    logger.info(f"Cached {n_done} samples across {len(cached)} batches for the sweep.")
    return cached


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/ablate_trm_loop_budget.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+ablation.xxx=...]"
        )

    ab = cfg.get("ablation", {})
    num_samples: int = ab.get("num_samples", 256)
    seed: int = ab.get("seed", 0)
    axes: list[str] = list(ab.get("axes", ["static", "carry"]))
    n_sup_values: list[int] = list(ab.get("n_sup_values", [1, 2, 4, 8, 16]))
    reset_every_values: list = list(ab.get("reset_every_values", [2, 5, 20]))
    schedule_flat_n = ab.get("schedule_flat_n", None)
    schedule_reset_every = ab.get("schedule_reset_every", None)
    schedule_directions: list[str] = list(ab.get("schedule_directions", ["front", "back"]))
    schedule_ratio: float = ab.get("schedule_ratio", 3.0)
    out_path: str = ab.get("out", str(Path(checkpoint).parent / "loop_budget_ablation.json"))

    if "schedule" in axes and schedule_flat_n is None:
        raise SystemExit(
            "ablation.axes includes 'schedule' but ablation.schedule_flat_n is not set. "
            "Run 'static'/'carry' first, inspect results, then rerun with "
            "+ablation.axes=[schedule] +ablation.schedule_flat_n=<N> "
            "+ablation.schedule_reset_every=<K or null>."
        )

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.eval.get("batch_size", cfg.train.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    cfg_scale: float = ab.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = ab.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  "
        f"trained n_sup={model.n_sup}  H_cycles={model.thinker.inner.config.H_cycles}  "
        f"L_cycles={model.thinker.inner.config.L_cycles}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict[str, dict] = {}

    if "static" in axes:
        for n in n_sup_values:
            key = f"static/n_sup={n}"
            logger.info(f"Running {key} ...")
            results[key] = _run_config(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                _make_n_sup_fn([n] * num_inference_steps), _make_reset_fn(1),
            )
            logger.info(f"  → {results[key]}")

    if "carry" in axes:
        for reset_every in reset_every_values:
            for n in n_sup_values:
                key = f"carry/reset_every={reset_every}/n_sup={n}"
                logger.info(f"Running {key} ...")
                results[key] = _run_config(
                    model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                    _make_n_sup_fn([n] * num_inference_steps), _make_reset_fn(reset_every),
                )
                logger.info(f"  → {results[key]}")

    if "schedule" in axes:
        reset_fn = _make_reset_fn(schedule_reset_every)
        flat_sched = [schedule_flat_n] * num_inference_steps
        for label, sched in [("flat", flat_sched)] + [
            (d, _matched_budget_schedule(d, schedule_flat_n, num_inference_steps, schedule_ratio))
            for d in schedule_directions
        ]:
            key = f"schedule/{label}/flat_n={schedule_flat_n}/reset_every={schedule_reset_every}"
            logger.info(f"Running {key} (per-step n_sup={sched}) ...")
            results[key] = _run_config(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                _make_n_sup_fn(sched), reset_fn,
            )
            results[key]["schedule"] = sched
            logger.info(f"  → {results[key]}")

    if accelerator.is_main_process:
        print("\n" + "=" * 100)
        print(f"{'config':<45}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}{'given_cons':>12}{'sup_calls':>10}")
        print("=" * 100)
        for key, r in results.items():
            print(
                f"{key:<45}{r['cell_acc']:>10.4f}{r['puzzle_acc']:>12.4f}{r['constraint_puzzle_acc']:>12.4f}"
                f"{r.get('given_consistent_puzzle_acc', float('nan')):>12.4f}{r['total_sup_calls']:>10d}"
            )
        print("=" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
