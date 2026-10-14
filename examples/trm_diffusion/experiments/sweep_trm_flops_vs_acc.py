"""experiments/sweep_trm_flops_vs_acc.py — puzzle accuracy vs. real FLOPs
for the TRM (painter+thinker) checkpoint, sweeping reasoning budget (n_sup).

Two of the paper's 6 compute/accuracy curves (see
paper_plots/plot_flops_vs_puzzle_acc.py; the other 4 come from
experiments/sweep_dm_flops_vs_acc.py, one per DM checkpoint):

  mode=flat — n_sup is the same at every denoising step, carry reset every
              step (reset_every=1, i.e. the model's normal/default inference
              policy — this reproduces the trained default n_sup=16 result).
              Reuses ablate_trm_loop_budget.py's `_run_ablation_sampling`/
              `_run_config` verbatim ("static" axis).

  mode=wise — total reasoning budget matched to a flat_n baseline
              (ablate_trm_loop_budget.py's `_matched_budget_schedule`, same
              "bump" shape used by experiments/export_z_scratchpad_animation.py:
              peak=0.1, width=0.2 — i.e. extra recursion concentrated near
              the highest-noise steps), latent state carried across ALL
              denoising steps (never reset after step 0), and steering
              frozen for the back half of the trajectory (step_idx >=
              round(0.5 * num_inference_steps): no thinker/encode call at
              all, just model.run_painter() reused on the last real logits —
              ported from export_z_scratchpad_animation.py's frozen branch).
              Reuses `_run_batched_schedule_scan` from that script (extended
              here with an optional H_cycles/L_cycles passthrough, default
              None = unchanged behavior for its original animation-export
              caller) instead of writing a new sampling loop.

Both modes also sweep a LOW-COMPUTE segment below n_sup=1 (which is the
floor — n_sup can't go to 0, see forward_with_carry) by reducing H_cycles
(the outer "big loop" — the thinker's trained config is H_cycles=3,
L_cycles=6) instead, at flat n_sup=1 / an all-ones schedule: for each value
in +sweep.h_cycles_values (default [1,2], i.e. below the trained default 3
— the trained-H_cycles segment is the normal n_sup/flat_n sweep above), one
point is run with that H_cycles override. This gives both curves a genuine
near-zero-compute prefix (rather than starting at n_sup=1's already-
substantial accuracy) for a fair comparison against the DM curves' floor.
F_thinker_step is re-measured (via flops_utils.measure_trm_flops's
H_cycles/L_cycles override) for each distinct H_cycles value used, since
FLOPs of one thinker.reasoning_step call scale with H_cycles*L_cycles.

FLOPs accounting (see flops_utils.py's docstring for why this is exact, not
a proxy): per non-frozen denoising step, FLOPs = (2 if cfg else 1) *
(F_encode + n_sup_i * F_thinker_step + F_painter); per frozen step, FLOPs =
(2 if cfg else 1) * F_painter (no encode, no thinker call). F_encode/
F_thinker_step/F_painter are each measured once up front via
flops_utils.measure_trm_flops.

IMPORTANT — diffusion.num_train_timesteps=4800: the frozen painter
(runs/mnist_unet_painter.pt) is the SAME checkpoint file used by the small-
DM curve, which was trained with diffusion.num_train_timesteps=4800 (see
sweep_dm_flops_vs_acc.py's docstring) — pass it explicitly here too so the
noise schedule matches what the painter's weights were actually trained
against.

Usage:
    # flat
    python experiments/sweep_trm_flops_vs_acc.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt +checkpoint=runs/sudoku-painter-thinker.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 +condition_encoder.enabled=false \\
      condition_encoder.inner.with_timestep_emb=false diffusion.num_train_timesteps=4800 \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +sweep.mode=flat +sweep.n_sup_values=[1,2,4,8,16,32] +sweep.num_samples=256 \\
      +sweep.out=runs/flops_sweep_trm_flat.json

    # wise (budget-matched bump schedule + carry + steering freeze)
    python experiments/sweep_trm_flops_vs_acc.py experiment=mnist_thinker_v1_controlnet \\
      ... (same overrides as above) \\
      +sweep.mode=wise +sweep.flat_n_values=[1,2,4,8,16,32] +sweep.num_samples=256 \\
      +sweep.out=runs/flops_sweep_trm_wise.json

Options (all under +sweep.*):
    mode              — "flat" or "wise" (required)
    num_samples       — puzzles evaluated per grid point (default 256)
    n_sup_values      — flat-mode n_sup grid (default [1,2,4,8,16,32])
    flat_n_values     — wise-mode budget grid (default [1,2,4,8,16,32]);
                        each value is the flat_n passed to
                        _matched_budget_schedule("bump", flat_n, ..., peak=
                        bump_peak, width=bump_width) — total reasoning
                        matches what a flat n_sup=flat_n baseline would use.
    h_cycles_values   — low-compute segment (both modes): H_cycles values
                        below the thinker's trained default to run at
                        n_sup=1 (flat) / an all-ones schedule (wise) —
                        default [1,2] (values >= the trained H_cycles are
                        skipped, since that's already the normal sweep's
                        first/densest region).
    bump_peak/bump_width — default 0.1 / 0.2 (matches
                        export_z_scratchpad_animation.py's defaults)
    freeze_frac       — fraction of the trajectory after which steering
                        freezes (default 0.5)
    seed              — default 0
    out               — output JSON path
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import (
    _build_cached_batches, _load_checkpoint, _make_n_sup_fn, _make_reset_fn, _matched_budget_schedule, _run_config,
)
from export_sampling_animation import _slice_sample
from export_z_scratchpad_animation import _run_batched_schedule_scan
from factory import build_datasets, build_model
from flops_utils import measure_trm_flops

logger = get_logger(__name__, log_level="INFO")

DEFAULT_N_SUP_VALUES = [1, 2, 4, 8, 16, 32]


def _flat_flops(num_steps: int, n_sup: int, cfg_scale: float, F_encode: float, F_thinker_step: float, F_painter: float) -> float:
    mult = 2 if cfg_scale != 1.0 else 1
    return num_steps * mult * (F_encode + n_sup * F_thinker_step + F_painter)


def _wise_flops(
    schedule: list, freeze_step: int, cfg_scale: float, F_encode: float, F_thinker_step: float, F_painter: float,
) -> float:
    mult = 2 if cfg_scale != 1.0 else 1
    total = 0.0
    for step_idx, n_sup_i in enumerate(schedule):
        if step_idx < freeze_step:
            total += mult * (F_encode + n_sup_i * F_thinker_step + F_painter)
        else:
            total += mult * F_painter
    return total


def _run_wise_point(
    model, classifier, cell_size: int, cached_batches: list, num_inference_steps: int, cfg_scale: float,
    schedule: list, freeze_step: int, reset_fn, H_cycles: int = None, L_cycles: int = None,
) -> dict:
    all_puzzle, weights = [], []
    t0 = time.time()
    for cb in cached_batches:
        preds_per_step = _run_batched_schedule_scan(
            model, classifier, cell_size, cb["conditions"], cb["x_init"],
            num_inference_steps, cfg_scale, schedule, freeze_step, reset_fn,
            H_cycles=H_cycles, L_cycles=L_cycles,
        )
        final_preds = preds_per_step[-1]  # (B, 81)
        solutions = cb["solutions"].cpu().numpy()
        puzzle_acc = (final_preds == solutions).all(axis=1).mean()
        all_puzzle.append(float(puzzle_acc))
        weights.append(solutions.shape[0])
    return {
        "puzzle_acc": float(np.average(all_puzzle, weights=weights)),
        "wall_time_sec": time.time() - t0,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/sweep_trm_flops_vs_acc.py experiment=mnist_thinker_v1_controlnet "
            "+checkpoint=<path/to/checkpoint.pt> diffusion.num_train_timesteps=4800 +sweep.mode=flat|wise [+sweep.xxx=...]"
        )

    sw = cfg.get("sweep", {})
    mode: str = sw.get("mode", None)
    if mode not in ("flat", "wise"):
        raise SystemExit("ERROR: +sweep.mode=flat or +sweep.mode=wise is required.")
    num_samples: int = sw.get("num_samples", 256)
    seed: int = sw.get("seed", 0)
    n_sup_values: list[int] = list(sw.get("n_sup_values", DEFAULT_N_SUP_VALUES))
    flat_n_values: list[int] = list(sw.get("flat_n_values", DEFAULT_N_SUP_VALUES))
    h_cycles_values: list[int] = list(sw.get("h_cycles_values", [1, 2]))
    l_cycles_values: list[int] = list(sw.get("l_cycles_values", h_cycles_values))
    bump_peak: float = sw.get("bump_peak", 0.1)
    bump_width: float = sw.get("bump_width", 0.2)
    freeze_frac: float = sw.get("freeze_frac", 0.5)
    out_path: str = sw.get("out", str(Path(checkpoint).parent / f"flops_sweep_trm_{mode}.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}  mode={mode}")

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds, batch_size=cfg.eval.get("batch_size", cfg.train.batch_size), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    if not hasattr(model, "forward_with_carry"):
        raise SystemExit("This model has no forward_with_carry — needs a TRM-based model.")
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    cfg_scale: float = sw.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = sw.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={model.n_sup}  "
        f"num_train_timesteps={model.scheduler.config.num_train_timesteps}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    import dataclasses
    flops_sample = _slice_sample(cached_batches[0]["conditions"], 0)
    flops_sample = dataclasses.replace(
        flops_sample, x_noisy=cached_batches[0]["x_init"][:1], timesteps=torch.zeros(1, device=device),
    )
    f = measure_trm_flops(model, flops_sample)
    F_encode, F_thinker_step, F_painter = f["F_encode"], f["F_thinker_step"], f["F_painter"]
    logger.info(f"F_encode={F_encode:,}  F_thinker_step={F_thinker_step:,}  F_painter={F_painter:,}")

    trained_H_cycles = model.thinker.inner.config.H_cycles
    trained_L_cycles = model.thinker.inner.config.L_cycles
    reduced_hl_pairs = [
        (h, l) for h, l in zip(h_cycles_values, l_cycles_values) if h < trained_H_cycles
    ]
    f_thinker_step_by_hl = {
        (h, l): measure_trm_flops(model, flops_sample, H_cycles=h, L_cycles=l)["F_thinker_step"]
        for h, l in reduced_hl_pairs
    }
    for (h, l), f_hl in f_thinker_step_by_hl.items():
        logger.info(
            f"F_thinker_step(H_cycles={h}, L_cycles={l}) = {f_hl:,}  "
            f"(trained H_cycles={trained_H_cycles}, L_cycles={trained_L_cycles})"
        )

    points = []
    if mode == "flat":
        for h, l in reduced_hl_pairs:
            logger.info(f"Running flat/H_cycles={h}/L_cycles={l}/n_sup=1 (low-compute segment) ...")
            r = _run_config(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                _make_n_sup_fn([1] * num_inference_steps), _make_reset_fn(1), H_cycles=h, L_cycles=l,
            )
            flops = _flat_flops(num_inference_steps, 1, cfg_scale, F_encode, f_thinker_step_by_hl[(h, l)], F_painter)
            point = {
                "n_sup": 1, "H_cycles": h, "L_cycles": l, "flops": flops, "puzzle_acc": r["puzzle_acc"],
                "wall_time_sec": r["wall_time_sec"],
            }
            points.append(point)
            logger.info(f"  → {point}")
        for n in n_sup_values:
            logger.info(f"Running flat/n_sup={n} ...")
            r = _run_config(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                _make_n_sup_fn([n] * num_inference_steps), _make_reset_fn(1),
            )
            flops = _flat_flops(num_inference_steps, n, cfg_scale, F_encode, F_thinker_step, F_painter)
            point = {
                "n_sup": n, "H_cycles": trained_H_cycles, "L_cycles": trained_L_cycles, "flops": flops,
                "puzzle_acc": r["puzzle_acc"], "wall_time_sec": r["wall_time_sec"],
            }
            points.append(point)
            logger.info(f"  → {point}")
    else:
        freeze_step = int(round(freeze_frac * num_inference_steps))
        reset_fn = _make_reset_fn(None)
        ones_schedule = [1] * num_inference_steps
        for h, l in reduced_hl_pairs:
            logger.info(f"Running wise/H_cycles={h}/L_cycles={l}/flat_n=1 (low-compute segment) freeze_step={freeze_step} ...")
            r = _run_wise_point(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                ones_schedule, freeze_step, reset_fn, H_cycles=h, L_cycles=l,
            )
            flops = _wise_flops(ones_schedule, freeze_step, cfg_scale, F_encode, f_thinker_step_by_hl[(h, l)], F_painter)
            point = {
                "flat_n": 1, "H_cycles": h, "L_cycles": l, "schedule": ones_schedule, "freeze_step": freeze_step,
                "flops": flops, **r,
            }
            points.append(point)
            logger.info(f"  → puzzle_acc={r['puzzle_acc']:.4f}  flops={flops:,.0f}")
        for flat_n in flat_n_values:
            schedule = _matched_budget_schedule("bump", flat_n, num_inference_steps, peak=bump_peak, width=bump_width)
            logger.info(f"Running wise/flat_n={flat_n} schedule={schedule} freeze_step={freeze_step} ...")
            r = _run_wise_point(
                model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale,
                schedule, freeze_step, reset_fn,
            )
            flops = _wise_flops(schedule, freeze_step, cfg_scale, F_encode, F_thinker_step, F_painter)
            point = {
                "flat_n": flat_n, "H_cycles": trained_H_cycles, "L_cycles": trained_L_cycles, "schedule": schedule,
                "freeze_step": freeze_step, "flops": flops, **r,
            }
            points.append(point)
            logger.info(f"  → puzzle_acc={r['puzzle_acc']:.4f}  flops={flops:,.0f}")

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        key = "n_sup" if mode == "flat" else "flat_n"
        print(f"{key:>10}{'flops':>18}{'puzzle_acc':>12}")
        print("=" * 70)
        for p in points:
            print(f"{p[key]:>10}{p['flops']:>18,.0f}{p['puzzle_acc']:>12.4f}")
        print("=" * 70)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({
                "checkpoint": str(checkpoint), "mode": mode, "num_samples": num_samples, "cfg_scale": cfg_scale,
                "num_inference_steps": num_inference_steps, "F_encode": F_encode, "F_thinker_step": F_thinker_step,
                "F_painter": F_painter, "points": points,
            }, fh, indent=2)
        logger.info(f"Results saved → {out_path}")

    return points


if __name__ == "__main__":
    main()
