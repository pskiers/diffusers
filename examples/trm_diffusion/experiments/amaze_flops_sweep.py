"""experiments/amaze_flops_sweep.py — the sudoku FLOPs-vs-accuracy analysis, for AMAZE maze / queens.

Ported from the sudoku implementation (branch pskiers/trm-diffusion):
    flops_utils.py, experiments/sweep_dm_flops_vs_acc.py, experiments/sweep_trm_flops_vs_acc.py
and the reasoning-allocation ablation (ablate_trm_loop_budget.py's "schedule" axis, bump heatmap).
Sampling loops / schedules / FLOP accounting are reused or mirrored verbatim; ONLY the scoring
differs: sudoku used a digit classifier (evaluate_grids), here every generated image is scored
with eval.amaze_eval.AmazeMetrics, exactly as experiments/amaze_generate_and_calculate_metrics.py
does (K=1 attempt per puzzle). Headline accuracy = `exact` (strict set equality, the AMAZE
analogue of sudoku puzzle_acc); `pass` (the paper's continuous Pass) is stored alongside.

Modes (+sweep.mode=...):
  dm        plain DiT baseline (experiment=amaze_dit_<task>), sweep DDPM inference-step count.
            FLOPs = steps * (2 if cfg else 1) * F_model        (flops_utils.measure_dm_flops)
  trm_flat  PaTh, n_sup the same at every denoising step, carry reset every step (the default
            inference policy), plus the sudoku script's low-compute prefix (reduced H/L cycles
            at n_sup=1). FLOPs = steps * mult * (F_encode + n_sup*F_thinker_step + F_painter)
  trm_wise  PaTh with a budget-matched Gaussian-bump schedule (peak 0.1 / width 0.2), carry kept
            across ALL denoising steps, steering frozen after freeze_frac of the trajectory
            (no encode / thinker call: run_painter reused on the last real logits).
  alloc     WHERE reasoning is needed: matched-total-budget reallocation over denoising steps
            (flat vs bump/dip over a peak x width grid, front, back), for each flat_n. Delta vs
            flat isolates *where* the budget sits from *how much* there is.

Phase-1 modes (the "where to cut" ablations; all inference-only, on the TUNING boards, trained thinker):
  profile   per denoising step: relative z_H change at every reasoning iteration (-> where z_H has converged) and the
            same-board cosine of the final z_H between every pair of denoising steps (-> regime boundary / carry map);
            also checks that the new sampler reproduces _run_ablation_sampling.
  cut_at_k  reason on the first k% of denoising steps (fresh state each step, trained n_sup), then reuse the last
            logits through the painter (SteeringReusePredictor-style freeze); accuracy + FLOPs per k.
  converge  exit INSIDE the reasoning loop per sample once the relative z_H change < eps (eps_values), with fresh state
            or carry-within-regime; FLOPs are analytic from the measured per-sample iteration counts.
  carry     n_sup x {fresh each step, carry everywhere, carry within a regime (reset at regime_boundary_step)}.
  +sweep.offset picks WHICH boards: 0 = tuning boards, e.g. 16 = held-out report boards.

Every point stores accuracy overall AND per board size and per shape (the difficulty axis), so "which puzzles
need more reasoning" falls out of the same runs. Results -> JSON (+sweep.out) and wandb.

Usage (see slurm_scripts/sample_amaze/flops_analysis.sh):
    python experiments/amaze_flops_sweep.py experiment=amaze_thinker_v2_controlnet \\
      painter.checkpoint=<painter.pt> +checkpoint=<thinker.pt> eval_callbacks=amaze_queens \\
      +sweep.task=queens +sweep.mode=trm_flat +sweep.out=runs/flops/queens_trm_flat.json
    python experiments/amaze_flops_sweep.py experiment=amaze_dit_queens +checkpoint=<dit.pt> \\
      +sweep.task=queens +sweep.mode=dm +sweep.out=runs/flops/queens_dm.json

Options (+sweep.*): task (maze|queens, required), mode (required), puzzles_per_combo (default 16:
puzzles per (geometry, size) for maze / per size for queens), batch_size (64), seed (0), out,
n_sup_values / flat_n_values ([1,2,4,8,16,32]), h_cycles_values / l_cycles_values ([1,2] / same
as h), bump_peak / bump_width (0.1/0.2), freeze_frac (0.5), steps (dm grid), alloc_flat_n_values
([2,4,8]), alloc_peaks ([0.1,0.3,0.5,0.7]), alloc_widths ([0.08,0.2,0.4]), alloc_ratio (3.0),
alloc_reset_every (1), wandb_project (amaze_flops), wandb_group, wandb (true).
"""

from __future__ import annotations

import dataclasses
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

from ablate_trm_loop_budget import (
    _make_n_sup_fn, _make_reset_fn, _matched_budget_schedule, _run_ablation_sampling,
)
from amaze_generate_and_calculate_metrics import _build_amaze_dataset, _require_test_parquet
from eval.amaze_eval import AmazeMetrics, MAZE_GEOMETRIES, MAZE_SCALES, QUEEN_SCALES
from eval.checkpoint_utils import load_checkpoint as _load_checkpoint
from factory import build_model
from flops_utils import measure_dm_flops, measure_trm_flops

logger = get_logger(__name__, log_level="INFO")

DEFAULT_N_SUP_VALUES = [1, 2, 4, 8, 16, 32]
DEFAULT_DM_STEPS = [1, 2, 5, 10, 20, 50, 100]


# ── Evaluation set + scoring ────────────────────────────────────────────────


def _build_cached_batches(model, cfg, task: str, data_root: Path, per_combo: int, batch_size: int, device, seed: int,
                          offset: int = 0, geometries=None, scales_only=None):
    """Fixed set of puzzles (boards [offset, offset+per_combo) of every (geometry, size) test parquet) with fixed
    initial noise, reused identically by every config -> paired comparison. `offset` is what separates the TUNING
    boards (offset 0) from the held-out REPORT boards. geometries / scales_only restrict the combos (smoke tests).
    Returns (batches, n_puzzles). Each batch: conditions, metadata, scales, shapes, x_init."""
    if task == "maze":
        geoms = list(geometries) if geometries else MAZE_GEOMETRIES
        sizes = [int(x) for x in scales_only] if scales_only else MAZE_SCALES
        combos = [(f"{g}_{s}", g, s, data_root / "maze" / g / f"n{s}_test.parquet") for g in geoms for s in sizes]
    else:
        sizes = [int(x) for x in scales_only] if scales_only else QUEEN_SCALES
        combos = [(f"n{s}", "queens", s, data_root / "queens" / f"n{s}_test.parquet") for s in sizes]

    puzzles, scales, shapes, collate = [], [], [], None
    for name, shape, scale, parquet in combos:
        _require_test_parquet(parquet, task)
        ds = _build_amaze_dataset(cfg, str(parquet))
        collate = ds.collate_fn
        idx = range(min(offset, len(ds)), min(offset + per_combo, len(ds)))
        puzzles += [ds[i] for i in idx]
        scales += [scale] * len(idx)
        shapes += [shape] * len(idx)
        logger.info(f"[{task}/{name}] using boards {idx.start}..{idx.stop - 1} of {len(ds)}")

    torch.manual_seed(seed)
    batches = []
    for start in range(0, len(puzzles), batch_size):
        chunk = puzzles[start:start + batch_size]
        conditions = model._batch_to_sample(collate(chunk), device)
        x_init = torch.randn(len(chunk), *model.noise_shape, device=device)
        batches.append({
            "conditions": conditions,
            "metadata": [p.metadata if p.metadata is not None else {} for p in chunk],
            "scales": scales[start:start + batch_size],
            "shapes": shapes[start:start + batch_size],
            "x_init": x_init,
        })
    logger.info(f"Cached {len(puzzles)} puzzles in {len(batches)} batches.")
    return batches, len(puzzles)


def _score_batches(model, scorer: AmazeMetrics, cached_batches: list, run_batch) -> dict:
    """run_batch(cb) -> final latent x (B, ...). Scores every puzzle with AmazeMetrics (K=1)."""
    scorer.reset()
    recs, scales, shapes = [], [], []
    t0 = time.time()
    for bi, cb in enumerate(cached_batches):
        # DDPMScheduler.step injects fresh noise at every step: reseed per batch so every config sees the SAME
        # injected noise on the same boards (x_init is already cached) -> a genuinely paired comparison.
        torch.manual_seed(10_000 + bi)
        x = run_batch(cb)
        decoded = model.decode_for_eval(x).float().cpu()
        recs += scorer.compute_and_accumulate_metrics(decoded, cb["metadata"])
        scales += cb["scales"]
        shapes += cb["shapes"]

    def _mean(key, idx=None, shape=None):
        vals = [r[key] for i, r in enumerate(recs)
                if (idx is None or scales[i] == idx) and (shape is None or shapes[i] == shape)]
        return float(np.mean(vals)) if vals else float("nan")

    per_scale = {
        str(s): {"exact": _mean("exact", s), "pass": _mean("pass", s), "n": int(sum(1 for x in scales if x == s))}
        for s in sorted(set(scales))
    }
    per_shape = {
        sh: {"exact": _mean("exact", shape=sh), "pass": _mean("pass", shape=sh),
             "n": int(sum(1 for x in shapes if x == sh))}
        for sh in sorted(set(shapes))
    }
    return {
        "exact": _mean("exact"), "pass": _mean("pass"), "per_shape": per_shape,
        "violation": _mean("background_violation"), "coverage": _mean("gt_cell_coverage"),
        "n": len(recs), "per_scale": per_scale, "wall_time_sec": time.time() - t0,
    }


# ── Sampling loops ──────────────────────────────────────────────────────────


@torch.no_grad()
def _run_dm_sampling(model, cb, num_steps: int, cfg_scale: float) -> torch.Tensor:
    """Plain denoiser loop (mirrors SamplingPipeline.sample_one_batch with the model's predictor)."""
    device = cb["x_init"].device
    model.scheduler.set_timesteps(num_inference_steps=num_steps, device=device)
    x = cb["x_init"].clone()
    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(cb["conditions"], x_noisy=x, timesteps=t_batch)
        pred = model(step_sample).pred
        if cfg_scale != 1.0:
            pred_u = model(model.null_condition_sample(step_sample)).pred
            pred = pred_u + cfg_scale * (pred - pred_u)
        x = model.scheduler.step(pred, t, x).prev_sample
    return x


@torch.no_grad()
def _run_wise_sampling(
    model, cb, num_steps: int, cfg_scale: float, schedule: list, freeze_step: int, H_cycles=None, L_cycles=None,
) -> torch.Tensor:
    """Carry (z_H, z_L) across ALL denoising steps (reset only at step 0); steering frozen from
    `freeze_step` on: no _encode_condition / thinker call, run_painter reused on each CFG
    branch's last real logits (the same call forward_with_carry makes after its n_sup loop)."""
    device = cb["x_init"].device
    model.scheduler.set_timesteps(num_steps, device=device)
    x = cb["x_init"].clone()
    z_H_c = z_L_c = z_H_u = z_L_u = None
    logits_c = logits_u = None
    use_cfg = cfg_scale != 1.0
    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(cb["conditions"], x_noisy=x, timesteps=t_batch)
        null_sample = model.null_condition_sample(step_sample) if use_cfg else None
        if step_idx < freeze_step:
            n_sup = schedule[step_idx]
            pred_c, z_H_c, z_L_c = model.forward_with_carry(
                step_sample, z_H_c, z_L_c, n_sup=n_sup, H_cycles=H_cycles, L_cycles=L_cycles)
            noise_pred, logits_c = pred_c.pred, pred_c.logits
            if use_cfg:
                pred_u, z_H_u, z_L_u = model.forward_with_carry(
                    null_sample, z_H_u, z_L_u, n_sup=n_sup, H_cycles=H_cycles, L_cycles=L_cycles)
                logits_u = pred_u.logits
                noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)
        else:
            noise_pred = model.run_painter(step_sample, logits_c)
            if use_cfg:
                pred_u = model.run_painter(null_sample, logits_u)
                noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return x


# ── Phase-1 policy sampler: cut along timesteps, exit inside the reasoning loop, carry maps ──────────────────


def _bcast(mask, ref):
    return mask.view(-1, *([1] * (ref.dim() - 1)))


@torch.no_grad()
def _reason(model, sample, z_H, z_L, n_max: int, eps=None, H_cycles=None, L_cycles=None, trace=None):
    """The thinker loop of ThinkerFrozenPainterBase.forward_with_carry (no-halt branch), re-written so each sample can
    leave it early. eps=None -> exactly n_max iterations (identical to forward_with_carry; checked in `profile`).
    eps=float -> a sample is DONE after the first iteration whose relative z_H change
    ||z_H' - z_H|| / ||z_H|| falls below eps; its state/logits are then frozen at that iteration's result.
    NOTE the batch still computes finished samples (no re-batching), so the FLOPs of an early-exit run are accounted
    ANALYTICALLY from the per-sample iteration counts returned here (as if finished samples had been removed).
    trace: optional list; the (B,) relative change of every iteration is appended (profile mode).
    Returns (logits, z_H, z_L, iters (B,) float)."""
    bsz = sample.x_noisy.shape[0]
    device = sample.x_noisy.device
    enc_emb = model._encode_condition(sample)
    if z_H is None or z_L is None:
        z_H, z_L = model.get_initial_states(bsz)
        z_H, z_L = z_H.to(device), z_L.to(device)
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    iters = torch.zeros(bsz, device=device)
    logits = None
    for _ in range(n_max):
        new_logits, new_zH, new_zL = model.thinker.reasoning_step(
            enc_emb, z_H, z_L, sample.puzzle_id, timesteps=sample.timesteps, H_cycles=H_cycles, L_cycles=L_cycles)
        rel = (new_zH - z_H).flatten(1).float().norm(dim=1) / z_H.flatten(1).float().norm(dim=1).clamp_min(1e-8)
        if trace is not None:
            trace.append(rel.detach())
        upd = ~done
        iters += upd.float()
        z_H = torch.where(_bcast(upd, z_H), new_zH, z_H)
        z_L = torch.where(_bcast(upd, z_L), new_zL, z_L)
        logits = new_logits if logits is None else torch.where(_bcast(upd, logits), new_logits, logits)
        if eps is not None:
            done = done | (rel < eps)
            if bool(done.all()):
                break
    return logits, z_H, z_L, iters


def _new_rec(num_steps: int) -> dict:
    return {"iters": [{"c": 0.0, "u": 0.0} for _ in range(num_steps)], "n": 0, "trace": [[] for _ in range(num_steps)],
            "zH": [None] * num_steps, "cos": None, "cos_n": 0}


@torch.no_grad()
def _run_policy(
    model, cb, num_steps: int, cfg_scale: float, n_sup_sched: list, freeze_step: int, reset_steps, eps=None,
    H_cycles=None, L_cycles=None, rec=None, record_trace: bool = False, record_z: bool = False,
) -> torch.Tensor:
    """One sampler for every phase-1 variant. Per denoising step, per CFG branch:
      step < freeze_step : reason n_sup_sched[step] iterations (early exit if eps) then run_painter on the logits
      step >= freeze_step: thinker NOT called, run_painter on that branch's last logits  (= cut_at_k)
    The carry (z_H, z_L) is reset to the thinker's initial state at the steps in `reset_steps` and carried otherwise:
      all steps -> fresh every step (the trained policy);  {0} -> carry everywhere;  {0, b} -> carry within a regime.
    rec (from _new_rec) accumulates per-step iteration counts (-> analytic FLOPs) and optional profile data."""
    device = cb["x_init"].device
    model.scheduler.set_timesteps(num_steps, device=device)
    x = cb["x_init"].clone()
    bsz = x.shape[0]
    use_cfg = cfg_scale != 1.0
    z_c = (None, None)
    z_u = (None, None)
    logits_c = logits_u = None
    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(bsz).to(device)
        step_sample = dataclasses.replace(cb["conditions"], x_noisy=x, timesteps=t_batch)
        null_sample = model.null_condition_sample(step_sample) if use_cfg else None
        if step_idx in reset_steps:
            z_c, z_u = (None, None), (None, None)
        if step_idx < freeze_step:
            trace = [] if (record_trace and rec is not None) else None
            logits_c, zH, zL, it = _reason(model, step_sample, *z_c, n_sup_sched[step_idx], eps, H_cycles, L_cycles, trace)
            z_c = (zH, zL)
            if rec is not None:
                rec["iters"][step_idx]["c"] += float(it.sum())
                if trace is not None:
                    rec["trace"][step_idx].append(torch.stack(trace, 1).float().cpu())  # (B, iters)
                if record_z:
                    rec["zH"][step_idx] = zH.detach().float()
            noise_pred = model.run_painter(step_sample, logits_c)
            if use_cfg:
                logits_u, zHu, zLu, itu = _reason(model, null_sample, *z_u, n_sup_sched[step_idx], eps, H_cycles, L_cycles)
                z_u = (zHu, zLu)
                if rec is not None:
                    rec["iters"][step_idx]["u"] += float(itu.sum())
                pred_u = model.run_painter(null_sample, logits_u)
                noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        else:
            noise_pred = model.run_painter(step_sample, logits_c)
            if use_cfg:
                pred_u = model.run_painter(null_sample, logits_u)
                noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    if rec is not None:
        rec["n"] += bsz
        if record_z:
            _accumulate_cos(rec, num_steps)
    return x


def _accumulate_cos(rec: dict, num_steps: int):
    """Same-board cosine of the final z_H between every pair of denoising steps (first panel of the interpretability
    heatmap). 'centered' removes the per-step mean over the boards of this batch first (the board-specific part); 'raw'
    does not. Accumulated as sums over boards; rec['cos'] = {'raw': (S,S), 'centered': (S,S)}."""
    zs = rec["zH"]
    if any(z is None for z in zs):
        return
    flat = [z.flatten(1) for z in zs]
    cen = [f - f.mean(0, keepdim=True) for f in flat]
    if rec["cos"] is None:
        rec["cos"] = {"raw": torch.zeros(num_steps, num_steps), "centered": torch.zeros(num_steps, num_steps)}
    for name, vs in (("raw", flat), ("centered", cen)):
        v = [torch.nn.functional.normalize(a, dim=1) for a in vs]
        for i in range(num_steps):
            for j in range(num_steps):
                rec["cos"][name][i, j] += float((v[i] * v[j]).sum(1).sum().cpu())
    rec["cos_n"] += flat[0].shape[0]
    rec["zH"] = [None] * num_steps


def _policy_flops(rec: dict, num_steps: int, freeze_step: int, cfg_scale: float, F_encode, F_thinker_step, F_painter):
    """Analytic FLOPs per sample from the measured mean iterations per step and branch (same per-call numbers as the
    other modes). Frozen steps cost F_painter per branch only."""
    n = max(rec["n"], 1)
    branches = ("c", "u") if cfg_scale != 1.0 else ("c",)
    total = 0.0
    for step in range(num_steps):
        for b in branches:
            if step < freeze_step:
                total += F_encode + (rec["iters"][step][b] / n) * F_thinker_step + F_painter
            else:
                total += F_painter
    return total


def _mean_iters_per_step(rec: dict, num_steps: int, freeze_step: int) -> list:
    n = max(rec["n"], 1)
    return [rec["iters"][s]["c"] / n if s < freeze_step else 0.0 for s in range(num_steps)]


def _profile_summary(rec: dict, num_steps: int) -> dict:
    """mean/median relative z_H change per (denoising step, iteration) + the step-by-step cosine matrices."""
    rel_mean, rel_med = [], []
    for s in range(num_steps):
        tr = torch.cat(rec["trace"][s], 0) if rec["trace"][s] else None
        rel_mean.append(tr.mean(0).tolist() if tr is not None else [])
        rel_med.append(tr.median(0).values.tolist() if tr is not None else [])
    out = {"rel_change_mean": rel_mean, "rel_change_median": rel_med}
    if rec["cos"] is not None:
        out["cos_raw"] = (rec["cos"]["raw"] / rec["cos_n"]).tolist()
        out["cos_centered"] = (rec["cos"]["centered"] / rec["cos_n"]).tolist()
    return out


def _load_profile(path: str) -> dict:
    """The `profile` block of a maze_profile.json (relative z_H change per iteration + step-by-step cosine matrices)."""
    return json.load(open(path))["profile"]


def _carry_maps(profile: dict, cos_thr: float, kind: str = "cos_centered"):
    """Reset maps derived from the same-board cosine of the final z_H (profile heatmap).
    nb[i] = cosine between denoising step i and i+1 (how well the state of step i still describes step i+1).
      regime boundary : the ONE step with the lowest neighbouring cosine (+1 = first step of the new regime)
      adaptive map    : reset at every step i+1 whose neighbouring cosine is below cos_thr (plus step 0)
    Returns (boundary_step, adaptive_reset_set, nb)."""
    c = profile[kind]
    nb = [c[i][i + 1] for i in range(len(c) - 1)]
    boundary = min(range(len(nb)), key=lambda i: nb[i]) + 1
    adaptive = {0} | {i + 1 for i, v in enumerate(nb) if v < cos_thr}
    return boundary, adaptive, nb


def _iters_needed(profile: dict, eps: float) -> list:
    """Per denoising step: number of reasoning iterations after which the mean relative z_H change is < eps
    (all iterations if it never gets there)."""
    out = []
    for rel in profile["rel_change_mean"]:
        k = next((i + 1 for i, v in enumerate(rel) if v < eps), len(rel))
        out.append(k)
    return out


def _rescale_schedule(needed: list, total: int) -> list:
    """Integer per-step budget proportional to `needed`, >= 1 everywhere, summing exactly to `total`."""
    scale = total / float(sum(needed))
    sched = [max(1, int(round(n * scale))) for n in needed]
    drift = total - sum(sched)
    order = sorted(range(len(sched)), key=lambda i: -sched[i])
    i = 0
    while drift != 0 and i < 10 * len(sched):
        j = order[i % len(order)]
        if drift > 0:
            sched[j] += 1
            drift -= 1
        elif sched[j] > 1:
            sched[j] -= 1
            drift += 1
        i += 1
    return sched


# ── FLOPs accounting (identical formulas to sweep_trm_flops_vs_acc.py) ──────


def _flat_flops(num_steps, n_sup, cfg_scale, F_encode, F_thinker_step, F_painter) -> float:
    mult = 2 if cfg_scale != 1.0 else 1
    return num_steps * mult * (F_encode + n_sup * F_thinker_step + F_painter)


def _wise_flops(schedule, freeze_step, cfg_scale, F_encode, F_thinker_step, F_painter) -> float:
    mult = 2 if cfg_scale != 1.0 else 1
    total = 0.0
    for step_idx, n_sup_i in enumerate(schedule):
        if step_idx < freeze_step:
            total += mult * (F_encode + n_sup_i * F_thinker_step + F_painter)
        else:
            total += mult * F_painter
    return total


def _slice0(sample):
    kw = {f.name: (getattr(sample, f.name)[:1] if isinstance(getattr(sample, f.name), torch.Tensor)
                   else getattr(sample, f.name)) for f in dataclasses.fields(type(sample))}
    return type(sample)(**kw)


def _flops_sample(model, cb, device):
    """B=1 real sample (FLOPs depend on shapes only, not values — see flops_utils.py)."""
    T = model.scheduler.config.num_train_timesteps
    return dataclasses.replace(
        _slice0(cb["conditions"]), x_noisy=cb["x_init"][:1], timesteps=torch.full((1,), T - 1, device=device),
    )


# ── wandb ───────────────────────────────────────────────────────────────────


def _wandb_log(sw, out: dict):
    """One readable wandb run per sweep (experiments/amaze_flops_relog.log_points): a labelled
    "results" table (one row per setting), exact / pass vs TFLOPs per maze, exact per board
    size. Never lets logging break the sweep (the JSON written before this is the source of truth)."""
    if not sw.get("wandb", True):
        return
    try:
        import wandb
        from amaze_flops_relog import log_points, split_of
        mode = sw["mode"]
        log_points(wandb, out, split_of(out), project=sw.get("wandb_project", "amaze_flops"),
                   group=sw.get("wandb_group", None), name=sw.get("wandb_name", None) or f"flops_{sw['task']}_{mode}",
                   tags=[sw["task"], mode, "flops"] + list(sw.get("wandb_tags", None) or []),
                   job_type=sw.get("wandb_job_type", None))
    except Exception as e:
        logger.warning(f"wandb logging failed ({e!r}); results are in the JSON.")


# ── Main ────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    sw = cfg.get("sweep", {})
    task, mode = sw.get("task", None), sw.get("mode", None)
    if checkpoint is None or task not in ("maze", "queens") or mode not in ("dm", "trm_flat", "trm_wise", "alloc", "profile", "cut_at_k", "converge", "carry"):
        raise SystemExit(
            "ERROR: need +checkpoint=<pt> +sweep.task=maze|queens +sweep.mode=dm|trm_flat|trm_wise|alloc|profile|cut_at_k|converge|carry "
            "(see this file's docstring)."
        )
    per_combo = int(sw.get("puzzles_per_combo", 16))
    batch_size = int(sw.get("batch_size", 64))
    seed = int(sw.get("seed", 0))
    offset = int(sw.get("offset", 0))
    geometries = list(sw["geometries"]) if "geometries" in sw else None
    scales_only = list(sw["scales"]) if "scales" in sw else None
    data_root = Path(cfg.get("data_root", str(Path(__file__).resolve().parent.parent / "data" / "amaze")))
    out_path = sw.get("out", str(Path(checkpoint).parent / f"flops_{task}_{mode}.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    step = _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    # same prepare/unwrap dance as amaze_generate_and_calculate_metrics.py so precision matches
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    is_trm = hasattr(model, "forward_with_carry")
    if mode == "dm" and is_trm:
        raise SystemExit("mode=dm needs a plain painter (e.g. experiment=amaze_dit_<task>), got a thinker model.")
    if mode != "dm" and not is_trm:
        raise SystemExit(f"mode={mode} needs a thinker model (experiment=amaze_thinker_v2_controlnet).")

    pipeline = model.sampling_pipeline
    cfg_scale = float(sw.get("cfg_scale", pipeline.cfg_scale))
    num_steps = int(sw.get("num_inference_steps", pipeline.num_inference_steps))
    logger.info(f"task={task} mode={mode} ckpt={checkpoint} (step={step}) cfg_scale={cfg_scale} "
                f"num_inference_steps={num_steps} T_train={scheduler.config.num_train_timesteps}")
    logger.info(OmegaConf.to_yaml(cfg.get("sweep", {})))

    cached, n_puzzles = _build_cached_batches(model, cfg, task, data_root, per_combo, batch_size, device, seed,
                                              offset=offset, geometries=geometries, scales_only=scales_only)
    scorer = AmazeMetrics(device=device, task=task)
    fs = _flops_sample(model, cached[0], device)
    points: list[dict] = []
    meta: dict = {}

    def record(point: dict, label: str):
        points.append(point)
        logger.info(f"  [{label}] flops={point.get('flops', float('nan')):,.0f} exact={point['exact']:.4f} "
                    f"pass={point['pass']:.4f} viol={point['violation']:.4f} cov={point['coverage']:.4f}")

    with accelerator.autocast():
        if mode == "dm":
            F_model = measure_dm_flops(model, fs)
            meta["F_model"] = F_model
            logger.info(f"F_model={F_model:,}")
            T = scheduler.config.num_train_timesteps
            grid = sorted({int(s) for s in sw.get("steps", DEFAULT_DM_STEPS) if int(s) <= T} | {num_steps})
            mult = 2 if cfg_scale != 1.0 else 1
            for n in grid:
                r = _score_batches(model, scorer, cached, lambda cb, n=n: _run_dm_sampling(model, cb, n, cfg_scale))
                record({"num_steps": n, "flops": n * mult * F_model, **r}, f"dm steps={n}")
            x_key, label_key = "flops", "num_steps"

        elif mode in ("trm_flat", "trm_wise"):
            f = measure_trm_flops(model, fs)
            F_encode, F_thinker_step, F_painter = f["F_encode"], f["F_thinker_step"], f["F_painter"]
            meta.update(f)
            logger.info(f"F_encode={F_encode:,} F_thinker_step={F_thinker_step:,} F_painter={F_painter:,}")
            trained_H = model.thinker.inner.config.H_cycles
            trained_L = model.thinker.inner.config.L_cycles
            meta.update(trained_n_sup=model.n_sup, trained_H_cycles=trained_H, trained_L_cycles=trained_L)
            h_vals = list(sw.get("h_cycles_values", [1, 2]))
            l_vals = list(sw.get("l_cycles_values", h_vals))
            # same selection rule as sweep_trm_flops_vs_acc.py: pairs with h < trained H_cycles
            reduced = [(h, l) for h, l in zip(h_vals, l_vals) if h < trained_H]
            f_step_hl = {(h, l): measure_trm_flops(model, fs, H_cycles=h, L_cycles=l)["F_thinker_step"]
                         for h, l in reduced}
            n_vals = list(sw.get("n_sup_values", DEFAULT_N_SUP_VALUES))
            flat_vals = list(sw.get("flat_n_values", DEFAULT_N_SUP_VALUES))

            if mode == "trm_flat":
                for h, l in reduced:
                    r = _score_batches(model, scorer, cached, lambda cb, h=h, l=l: _run_ablation_sampling(
                        model, cb["conditions"], cb["x_init"], num_steps, cfg_scale,
                        _make_n_sup_fn([1] * num_steps), _make_reset_fn(1), H_cycles=h, L_cycles=l))
                    record({"n_sup": 1, "H_cycles": h, "L_cycles": l,
                            "flops": _flat_flops(num_steps, 1, cfg_scale, F_encode, f_step_hl[(h, l)], F_painter), **r},
                           f"flat H={h} L={l} n_sup=1")
                for n in n_vals:
                    r = _score_batches(model, scorer, cached, lambda cb, n=n: _run_ablation_sampling(
                        model, cb["conditions"], cb["x_init"], num_steps, cfg_scale,
                        _make_n_sup_fn([n] * num_steps), _make_reset_fn(1)))
                    record({"n_sup": n, "H_cycles": trained_H, "L_cycles": trained_L,
                            "flops": _flat_flops(num_steps, n, cfg_scale, F_encode, F_thinker_step, F_painter), **r},
                           f"flat n_sup={n}")
                label_key = "n_sup"
            else:
                bump_peak, bump_width = float(sw.get("bump_peak", 0.1)), float(sw.get("bump_width", 0.2))
                freeze_step = int(round(float(sw.get("freeze_frac", 0.5)) * num_steps))
                meta.update(bump_peak=bump_peak, bump_width=bump_width, freeze_step=freeze_step)
                for h, l in reduced:
                    sched = [1] * num_steps
                    r = _score_batches(model, scorer, cached, lambda cb, h=h, l=l, s=sched: _run_wise_sampling(
                        model, cb, num_steps, cfg_scale, s, freeze_step, H_cycles=h, L_cycles=l))
                    record({"flat_n": 1, "H_cycles": h, "L_cycles": l, "schedule": sched, "freeze_step": freeze_step,
                            "flops": _wise_flops(sched, freeze_step, cfg_scale, F_encode, f_step_hl[(h, l)], F_painter),
                            **r}, f"wise H={h} L={l} flat_n=1")
                for fn in flat_vals:
                    sched = _matched_budget_schedule("bump", fn, num_steps, peak=bump_peak, width=bump_width)
                    r = _score_batches(model, scorer, cached, lambda cb, s=sched: _run_wise_sampling(
                        model, cb, num_steps, cfg_scale, s, freeze_step))
                    record({"flat_n": fn, "H_cycles": trained_H, "L_cycles": trained_L, "schedule": sched,
                            "freeze_step": freeze_step,
                            "flops": _wise_flops(sched, freeze_step, cfg_scale, F_encode, F_thinker_step, F_painter),
                            **r}, f"wise flat_n={fn}")
                label_key = "flat_n"
            x_key = "flops"

        elif mode == "alloc":
            f = measure_trm_flops(model, fs)
            F_encode, F_thinker_step, F_painter = f["F_encode"], f["F_thinker_step"], f["F_painter"]
            meta.update(f)
            reset_every = sw.get("alloc_reset_every", 1)
            ratio = float(sw.get("alloc_ratio", 3.0))
            peaks = list(sw.get("alloc_peaks", [0.1, 0.3, 0.5, 0.7]))
            widths = list(sw.get("alloc_widths", [0.08, 0.2, 0.4]))
            flat_vals = list(sw.get("alloc_flat_n_values", [2, 4, 8]))
            directions = list(sw.get("alloc_directions", ["bump", "dip", "front", "back"]))
            meta.update(alloc_reset_every=reset_every, alloc_ratio=ratio, alloc_peaks=peaks, alloc_widths=widths,
                        alloc_directions=directions)
            needed = None
            if sw.get("profile_json", None):
                prof_eps = float(sw.get("alloc_profile_eps", 0.01))
                needed = _iters_needed(_load_profile(str(sw["profile_json"])), prof_eps)
                meta.update(alloc_profile_eps=prof_eps, alloc_profile_needed=needed, profile_json=str(sw["profile_json"]))
                logger.info(f"profile-derived need per step (eps={prof_eps}): {needed}")
            reset_fn = _make_reset_fn(reset_every)
            for fn in flat_vals:
                variants = [("flat", None, None, [fn] * num_steps)]
                for d in directions:
                    if d in ("bump", "dip"):
                        for pk in peaks:
                            for w in widths:
                                variants.append((d, pk, w, _matched_budget_schedule(d, fn, num_steps, ratio, pk, w)))
                    else:
                        variants.append((d, None, None, _matched_budget_schedule(d, fn, num_steps, ratio)))
                if needed is not None:
                    variants.append(("profile", None, None, _rescale_schedule(needed, fn * num_steps)))
                for d, pk, w, sched in variants:
                    r = _score_batches(model, scorer, cached, lambda cb, s=sched: _run_ablation_sampling(
                        model, cb["conditions"], cb["x_init"], num_steps, cfg_scale,
                        _make_n_sup_fn(s), reset_fn))
                    record({"direction": d, "peak": pk, "width": w, "flat_n": fn, "schedule": sched,
                            "total_sup_calls": int(sum(sched)) * (2 if cfg_scale != 1.0 else 1),
                            "flops": sum((2 if cfg_scale != 1.0 else 1) * (F_encode + n * F_thinker_step + F_painter)
                                         for n in sched), **r},
                           f"alloc flat_n={fn} {d} peak={pk} width={w}")
            x_key, label_key = "flat_n", "direction"

        else:  # phase 1: profile | cut_at_k | converge | carry   (all on the thinker with fresh-state training policy)
            f = measure_trm_flops(model, fs)
            F_encode, F_thinker_step, F_painter = f["F_encode"], f["F_thinker_step"], f["F_painter"]
            meta.update(f)
            n_tr = int(model.n_sup)
            meta["trained_n_sup"] = n_tr
            all_steps = set(range(num_steps))
            boundary = int(sw.get("regime_boundary_step", 13))   # first step of the late regime (t=30 of 20 steps 95..0)
            meta["regime_boundary_step"] = boundary
            reset_sets = {"fresh": all_steps, "carry_all": {0}, "carry_regime": {0, boundary}}
            for N in (1, 2, 5, 20):                       # "reset every N denoising steps" (every1 = fresh, every20 = carry_all)
                reset_sets[f"every{N}"] = set(range(0, num_steps, N))
            carry_default = ["fresh", "carry_all", "carry_regime"]
            if sw.get("profile_json", None):
                cos_thr = float(sw.get("cos_threshold", 0.93))
                boundary, adaptive, nb = _carry_maps(_load_profile(str(sw["profile_json"])), cos_thr)
                reset_sets["carry_regime"] = {0, boundary}
                reset_sets["regime"] = {0, boundary}
                reset_sets["adaptive"] = adaptive
                carry_default = ["every1", "every2", "every5", "every20", "regime", "adaptive"]
                meta.update(profile_json=str(sw["profile_json"]), cos_threshold=cos_thr, measured_boundary_step=boundary,
                            adaptive_reset_steps=sorted(adaptive), neighbour_cos=nb, regime_boundary_step=boundary)
                logger.info(f"cosine-derived maps: boundary step={boundary} adaptive resets={sorted(adaptive)} (thr={cos_thr})")

            def policy_point(label: dict, n_sup: int, freeze: int, resets, eps=None, **kw):
                rec = _new_rec(num_steps)
                sched = [n_sup] * num_steps
                r = _score_batches(model, scorer, cached, lambda cb: _run_policy(
                    model, cb, num_steps, cfg_scale, sched, freeze, resets, eps=eps, rec=rec, **kw))
                fl = _policy_flops(rec, num_steps, freeze, cfg_scale, F_encode, F_thinker_step, F_painter)
                return {**label, "flops": fl, "iters_per_step": _mean_iters_per_step(rec, num_steps, freeze), **r}, rec

            if mode == "profile":
                # (1) consistency check: the new sampler with the trained policy must reproduce _run_ablation_sampling
                cb0 = cached[0]
                torch.manual_seed(7)   # DDPM step() draws noise: identical seed for both runs
                ref = _run_ablation_sampling(model, cb0["conditions"], cb0["x_init"], num_steps, cfg_scale,
                                             _make_n_sup_fn([n_tr] * num_steps), _make_reset_fn(1))
                torch.manual_seed(7)
                new = _run_policy(model, cb0, num_steps, cfg_scale, [n_tr] * num_steps, num_steps, all_steps)
                meta["consistency_max_abs_diff"] = float((ref.float() - new.float()).abs().max())
                logger.info(f"consistency check (_run_policy vs _run_ablation_sampling): max|diff|="
                            f"{meta['consistency_max_abs_diff']:.3e}")
                # (2) per-iteration relative z_H change per denoising step + step-by-step cosine matrices
                rec = _new_rec(num_steps)
                r = _score_batches(model, scorer, cached, lambda cb: _run_policy(
                    model, cb, num_steps, cfg_scale, [n_tr] * num_steps, num_steps, all_steps, rec=rec,
                    record_trace=True, record_z=True))
                meta["profile"] = _profile_summary(rec, num_steps)
                meta["timesteps"] = [int(t) for t in model.scheduler.timesteps]
                record({"flops": _policy_flops(rec, num_steps, num_steps, cfg_scale, F_encode, F_thinker_step, F_painter),
                        **r}, "profile (trained policy)")

            elif mode == "cut_at_k":
                for frac in list(sw.get("cut_fracs", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 1.0])):
                    freeze = max(1, round(float(frac) * num_steps))
                    pt, _ = policy_point({"reason_frac": float(frac), "freeze_step": freeze}, n_tr, freeze, all_steps)
                    record(pt, f"cut_at_k frac={frac} (reason on {freeze}/{num_steps} steps)")

            elif mode == "converge":
                for eps in list(sw.get("eps_values", [0.1, 0.03, 0.01])):
                    for variant in list(sw.get("converge_variants", ["fresh", "carry_regime"])):
                        pt, _ = policy_point({"eps": float(eps), "variant": variant}, n_tr, num_steps,
                                             reset_sets[variant], eps=float(eps))
                        record(pt, f"converge eps={eps} {variant} mean_iters={np.mean(pt['iters_per_step']):.2f}")

            else:  # carry
                for n in list(sw.get("carry_n_values", [2, 4, 8, 16])):
                    for variant in list(sw.get("carry_variants", carry_default)):
                        pt, _ = policy_point({"n_sup": int(n), "variant": variant}, int(n), num_steps, reset_sets[variant])
                        record(pt, f"carry n_sup={n} {variant}")
            x_key, label_key = "flops", {"cut_at_k": "reason_frac", "converge": "eps", "carry": "variant"}.get(mode, "flops")

    if accelerator.is_main_process:
        print("\n" + "=" * 78)
        print(f"{task} / {mode}  (n_puzzles={n_puzzles})")
        for p in points:
            lab = {k: p[k] for k in ("num_steps", "n_sup", "flat_n", "direction", "peak", "width", "H_cycles", "L_cycles") if k in p and p[k] is not None}
            print(f"{str(lab):<70} flops={p['flops']:>16,.0f} exact={p['exact']:.4f} pass={p['pass']:.4f}")
        print("=" * 78)

        out = {
            "task": task, "mode": mode, "checkpoint": str(checkpoint), "ckpt_step": step, "n_puzzles": n_puzzles,
            "puzzles_per_combo": per_combo, "offset": offset, "seed": seed, "cfg_scale": cfg_scale, "num_inference_steps": num_steps,
            "num_train_timesteps": scheduler.config.num_train_timesteps, **meta, "points": points,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        logger.info(f"Results saved -> {out_path}")

        if not points:
            return points
        _wandb_log(sw, out)
    return points


if __name__ == "__main__":
    main()
