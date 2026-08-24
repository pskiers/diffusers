"""
experiments/probe_reasoning_dynamics.py — D2 (candidate trajectory) and D3
(repair vs. regression), rebuilt on a trained digit probe instead of
render-then-classify (see experiments/train_digit_probe.py for why: the
CNN-on-rendered-pixels decode used by candidate_trajectory_probe.py /
repair_transition_probe.py is dominated by ordinary diffusion x0-blur at
early denoising steps, swamping whatever the TRM has actually reasoned
out; a probe on z_H itself sidesteps that).

Two changes from the original D2/D3 scripts, beyond the decode mechanism:

1. H-CYCLE-LEVEL GRANULARITY, not just n_sup-level. Every real
   reasoning_step call internally runs H_cycles sub-cycles, and z_H only
   actually changes once per H-cycle (it's held fixed while the inner
   L_cycles loop updates z_L) — so H-cycle level is the finest granularity
   at which probing z_H again would show new information. This is done
   with NO changes to the model and NO extra compute: reasoning_step
   already accepts an H_cycles override, and one real n_sup iteration
   (H_cycles=3 in one call) is mathematically identical to three
   sequential H_cycles=1 calls with z_H/z_L threaded through (verified by
   reading TinyRecursiveReasoningModel_ACTV1_Inner.forward — an H_cycles=1
   call always runs exactly "the final graded cycle", and chaining three
   of those performs the identical sequence of L_level operations as one
   H_cycles=3 call; gradient bookkeeping differs, forward values don't,
   and this all runs under no_grad anyway). Concretely: instead of calling
   forward_with_carry(n_sup=n_sup, H_cycles=<default>) once per denoising
   step, call forward_with_carry(n_sup=n_sup*H_cycles, H_cycles=1) once
   per denoising step — same total L_level calls, same final prediction
   (so x still advances identically to real inference), but zH_out now
   captures every H-cycle-level update instead of one point per n_sup
   iteration.

2. RESET-POLICY ABLATION. Real deployed inference resets the carry
   (z_H, z_L) fresh at every denoising step (forward_with_carry always
   called with z_H=z_L=None from model.forward()). This script sweeps
   probe.reset_every to also test carrying the (now H-cycle-granular)
   state across denoising steps, exactly like ablate_trm_loop_budget.py's
   D1 axis (reset_every=1 reproduces real inference; reset_every=None
   never resets after the first step; other K resets every K steps) — out
   of the training distribution for K!=1, a real possible regression, not
   a bug if seen.

Ground truth, not the probe's own training label: unlike training (which
deliberately avoids ground truth to sidestep circularity — see
train_digit_probe.py), D2/D3 use the frozen, already-trained probe purely
as a READOUT instrument and compare its output to the actual puzzle
solution. This is not circular: the probe was never fit against ground
truth, so asking "does this readout match the true answer" is a genuine,
independent accuracy measurement.

D2 output: per (denoising_step, fine-grained iteration) cell/puzzle/
constraint accuracy against ground truth — the full grid is saved;
representative rows/cols are printed, bucketed into early/mid/late
denoising steps as before.

D3 output: repair/regression transition counts between CONSECUTIVE fine
iterations *within* the same denoising step only (not across denoising
step boundaries, which would conflate reasoning's own effect with the
noise level changing) — pooled per denoising-step bucket.

ACT halt-head ablation is intentionally NOT included here yet — neither
real checkpoint currently has a trained halt head; that's a separate,
sequenced follow-up (train one via experiments/train_halt_head.py first).

Usage:
    python experiments/probe_reasoning_dynamics.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +checkpoint=runs/sudoku-painter-thinker.pt \\
      +probe_checkpoint=runs/digit_probe_sudoku_painter_thinker.pt \\
      +use_ema=true +dynamics.num_samples=64 +dynamics.reset_every_values=[1,5,null]

    # Options (all under +dynamics.*):
    #   num_samples        — default 64
    #   seed                — default 0
    #   reset_every_values  — default [1, 5, null] (null = never reset after step 0)
    #   out                 — json path (default: alongside probe_checkpoint)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
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

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from eval.mnist_eval import _check_sudoku_constraints
from factory import build_datasets, build_model
from models.digit_probe import DigitProbe

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _run_fine_grained_trajectory(
    model, probe: DigitProbe, puzzle_emb_len: int, conditions, x_init: torch.Tensor,
    num_inference_steps: int, cfg_scale: float, reset_fn,
) -> list:
    """Runs the real CFG trajectory (x advances identically to real
    inference — the fine-grained decomposition below only reorganizes how
    the SAME total compute is split into calls, see module docstring).

    Returns fine_grid: list of length num_inference_steps, each entry a
    list of length (n_sup * H_cycles) of (B, 81) long tensors — the
    probe's own predicted digit at every H-cycle-level sub-step of that
    denoising step, read off the CONDITIONAL branch only (the CFG null
    branch reasons about a zeroed condition and isn't meaningful here).
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    reasoner = getattr(model, "thinker", model)
    n_sup = model.n_sup
    H_cycles_real = reasoner.inner.config.H_cycles
    fine_n_sup = n_sup * H_cycles_real

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None
    fine_grid: list = []

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        step_zH: list = []
        pred_c, z_H_c, z_L_c = model.forward_with_carry(
            step_sample, z_H_c, z_L_c, n_sup=fine_n_sup, H_cycles=1, zH_out=step_zH
        )
        noise_pred = pred_c.pred

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, z_H_u, z_L_u = model.forward_with_carry(
                null_sample, z_H_u, z_L_u, n_sup=fine_n_sup, H_cycles=1,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        step_preds = [probe(z_H[:, puzzle_emb_len:]).argmax(-1) for z_H in step_zH]
        fine_grid.append(step_preds)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return fine_grid


def _d2_accumulate(acc_correct, acc_total, constraint_correct, fine_grid, solutions, blank):
    """acc_correct/acc_total: (num_inference_steps, fine_n_sup) running
    sums, accumulated in place. constraint_correct: same shape, counts
    constraint-satisfying grids (denominator is the batch size, not blank
    cell count)."""
    B = solutions.shape[0]
    for step_idx, step_preds in enumerate(fine_grid):
        for fine_idx, preds in enumerate(step_preds):
            correct = (preds == solutions) & blank
            acc_correct[step_idx, fine_idx] += int(correct.sum().item())
            acc_total[step_idx, fine_idx] += int(blank.sum().item())
            constraint_correct[step_idx, fine_idx] += int(_check_sudoku_constraints(preds).sum().item())


def _d3_accumulate(repairs, regressions, wrong_counts, right_counts, fine_grid, solutions, blank):
    """Within-denoising-step-only consecutive-pair transition counts,
    pooled per denoising step index (dims: num_inference_steps,
    fine_n_sup - 1)."""
    for step_idx, step_preds in enumerate(fine_grid):
        for k in range(len(step_preds) - 1):
            right_k = (step_preds[k] == solutions) & blank
            right_k1 = (step_preds[k + 1] == solutions) & blank
            wrong_k = (~right_k) & blank
            repairs[step_idx, k] += int((wrong_k & right_k1).sum().item())
            regressions[step_idx, k] += int((right_k & ~right_k1).sum().item())
            wrong_counts[step_idx, k] += int(wrong_k.sum().item())
            right_counts[step_idx, k] += int(right_k.sum().item())


@torch.no_grad()
def _run_halt_trajectory(
    model, probe: DigitProbe, puzzle_emb_len: int, conditions, x_init: torch.Tensor,
    num_inference_steps: int, cfg_scale: float, halt_threshold: float,
) -> list:
    """Runs the real CFG trajectory using STANDARD (non-fine-decomposed)
    n_sup granularity with use_halt_head=True — the granularity the halt
    head was actually trained at (see experiments/train_halt_head.py;
    applying it to the fine H_cycles=1 decomposition used by
    _run_fine_grained_trajectory would ask it to make decisions at a
    granularity it never saw in training). Always resets fresh at every
    denoising step (matching real deployed behavior — this ablation is
    about halting, not carry; see reset_every for that axis separately).

    Returns grid: list of length num_inference_steps, each entry a list of
    length n_sup of (B, 81) long tensors (the probe's prediction at every
    iteration). If every sample halts before n_sup iterations, the list is
    padded by repeating the last captured prediction — a real, exact
    behavior (once everyone has halted the frozen state doesn't change),
    not an approximation — so downstream accumulators always see a
    uniform-length (num_inference_steps, n_sup) grid.
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    n_sup = model.n_sup
    grid: list = []

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        step_zH: list = []
        pred_c, _, _ = model.forward_with_carry(
            step_sample, use_halt_head=True, halt_threshold=halt_threshold, zH_out=step_zH
        )
        noise_pred = pred_c.pred

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, _, _ = model.forward_with_carry(
                null_sample, use_halt_head=True, halt_threshold=halt_threshold,
            )
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        step_preds = [probe(z_H[:, puzzle_emb_len:]).argmax(-1) for z_H in step_zH]
        while len(step_preds) < n_sup:
            step_preds.append(step_preds[-1])
        grid.append(step_preds)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return grid


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    probe_checkpoint = cfg.get("probe_checkpoint", None)
    if checkpoint is None or probe_checkpoint is None:
        raise SystemExit(
            "ERROR: need both checkpoint= and +probe_checkpoint=.\n"
            "  Usage: python experiments/probe_reasoning_dynamics.py experiment=<name> "
            "checkpoint=<main.pt> +probe_checkpoint=<digit_probe.pt> [+dynamics.xxx=...]"
        )

    dyn = cfg.get("dynamics", {})
    num_samples: int = dyn.get("num_samples", 64)
    seed: int = dyn.get("seed", 0)
    reset_every_values: list = list(dyn.get("reset_every_values", [1, 5, None]))
    out_path: str = dyn.get("out", str(Path(probe_checkpoint).parent / "probe_reasoning_dynamics.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}  Probe: {probe_checkpoint}")

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

    reasoner = getattr(model, "thinker", model)
    puzzle_emb_len = reasoner.inner.puzzle_emb_len
    H_cycles_real = reasoner.inner.config.H_cycles
    n_sup = model.n_sup
    fine_n_sup = n_sup * H_cycles_real

    probe_ckpt = torch.load(str(probe_checkpoint), map_location=device, weights_only=True)
    probe = DigitProbe(probe_ckpt["hidden_size"]).to(device)
    probe.load_state_dict(probe_ckpt["probe_state"])
    probe.eval()
    assert probe_ckpt["puzzle_emb_len"] == puzzle_emb_len, "probe/model puzzle_emb_len mismatch"

    pipeline = model.sampling_pipeline
    cfg_scale: float = dyn.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = dyn.get("num_inference_steps", pipeline.num_inference_steps)

    logger.info(
        f"num_inference_steps={num_inference_steps}  n_sup={n_sup}  H_cycles={H_cycles_real}  "
        f"fine_n_sup={fine_n_sup}  reset_every_values={reset_every_values}"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    def _compute_dynamics(grid_fn, grid_width: int) -> dict:
        """grid_fn(cb) -> grid (list of length num_inference_steps, each a
        list of length grid_width of (B,81) preds). Shared accumulation
        logic for both the reset-ablation (fine_n_sup-wide grid) and the
        halt-ablation (n_sup-wide grid) axes."""
        acc_correct = np.zeros((num_inference_steps, grid_width), dtype=np.int64)
        acc_total = np.zeros((num_inference_steps, grid_width), dtype=np.int64)
        constraint_correct = np.zeros((num_inference_steps, grid_width), dtype=np.int64)
        repairs = np.zeros((num_inference_steps, grid_width - 1), dtype=np.int64)
        regressions = np.zeros((num_inference_steps, grid_width - 1), dtype=np.int64)
        wrong_counts = np.zeros((num_inference_steps, grid_width - 1), dtype=np.int64)
        right_counts = np.zeros((num_inference_steps, grid_width - 1), dtype=np.int64)
        total_samples = 0

        for cb in cached_batches:
            device_ = cb["x_init"].device
            solutions = cb["solutions"].to(device_)
            given_mask = cb["given_masks"].to(device_) if cb["given_masks"] is not None else None
            blank = (~given_mask) if given_mask is not None else torch.ones_like(solutions, dtype=torch.bool)
            total_samples += solutions.shape[0]

            grid = grid_fn(cb)
            _d2_accumulate(acc_correct, acc_total, constraint_correct, grid, solutions, blank)
            _d3_accumulate(repairs, regressions, wrong_counts, right_counts, grid, solutions, blank)

        return {
            "pooled_acc": float(acc_correct.sum() / max(acc_total.sum(), 1)),
            "acc_grid": (acc_correct / np.maximum(acc_total, 1)).tolist(),
            "constraint_grid": (constraint_correct / max(total_samples, 1)).tolist(),
            "repair_rate_grid": (repairs / np.maximum(wrong_counts, 1)).tolist(),
            "regression_rate_grid": (regressions / np.maximum(right_counts, 1)).tolist(),
        }

    reset_results: dict = {}
    for reset_every in reset_every_values:
        reset_fn = _make_reset_fn(reset_every)
        r = _compute_dynamics(
            lambda cb: _run_fine_grained_trajectory(
                model, probe, puzzle_emb_len, cb["conditions"], cb["x_init"],
                num_inference_steps, cfg_scale, reset_fn,
            ),
            fine_n_sup,
        )
        r["reset_every"] = reset_every
        reset_results[str(reset_every)] = r
        logger.info(f"reset_every={reset_every}  pooled_acc={r['pooled_acc']:.4f}")

    halt_threshold_values: list = list(dyn.get("halt_threshold_values", []))
    halt_results: dict = {}
    if halt_threshold_values:
        if not getattr(reasoner, "with_halt_head", False):
            raise SystemExit(
                "dynamics.halt_threshold_values is set but the model was built without a halt "
                "head — add thinker.with_halt_head=true and point checkpoint= at a checkpoint "
                "produced by experiments/train_halt_head.py."
            )
        for halt_threshold in halt_threshold_values:
            r = _compute_dynamics(
                lambda cb: _run_halt_trajectory(
                    model, probe, puzzle_emb_len, cb["conditions"], cb["x_init"],
                    num_inference_steps, cfg_scale, halt_threshold,
                ),
                n_sup,
            )
            r["halt_threshold"] = halt_threshold
            halt_results[str(halt_threshold)] = r
            logger.info(f"halt_threshold={halt_threshold}  pooled_acc={r['pooled_acc']:.4f}")

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint), "probe_checkpoint": str(probe_checkpoint),
                    "num_samples": num_samples, "num_inference_steps": num_inference_steps,
                    "n_sup": n_sup, "H_cycles": H_cycles_real, "fine_n_sup": fine_n_sup,
                    "reset_results": reset_results,
                    "halt_results": halt_results,
                },
                f, indent=2,
            )
        logger.info(f"Saved → {out_path}")

    return {"reset_results": reset_results, "halt_results": halt_results}


if __name__ == "__main__":
    main()
