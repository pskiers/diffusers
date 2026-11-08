"""flops_utils.py — real per-forward-pass FLOP measurement for the
FLOPs-vs-puzzle-accuracy paper plot (paper_plots/plot_flops_vs_puzzle_acc.py
and its two sweep scripts, experiments/sweep_dm_flops_vs_acc.py and
experiments/sweep_trm_flops_vs_acc.py).

Uses torch.utils.flop_counter.FlopCounterMode (built into torch >= 2.0, no
extra dependency) rather than fvcore/thop: it counts FLOPs via
__torch_dispatch__ on a normal eager forward pass, so it works directly on
this codebase's dataclass (DataSample) inputs and branchy model code —
fvcore's jit-trace-based analysis would need every traced module to accept
tensor-only args and would likely choke on that.

Each model call here is profiled ONCE per checkpoint on a single (B=1) real
cached sample — FLOPs of conv/linear/attention ops depend only on tensor
shapes, not values, so which puzzle is used doesn't matter. The sweep
scripts then compute per-config FLOPs analytically from these per-call
numbers (see forward_with_carry's call structure: _encode_condition runs
once per denoising step, thinker.reasoning_step runs n_sup times, and
run_painter runs exactly once, AFTER the n_sup loop, regardless of n_sup —
verified by reading models/painter_thinkers.py's forward_with_carry in
full) instead of re-profiling every sweep point.
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode


@torch.no_grad()
def _measure(fn) -> int:
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


@torch.no_grad()
def measure_dm_flops(model, sample) -> int:
    """FLOPs of one plain-DM denoiser forward pass: model(sample).pred.
    Works on any PainterBase model (see denoising_budget_sweep.py's own
    docstring for the same claim) — used for the 4 pure-DM checkpoints."""
    return _measure(lambda: model(sample))


@torch.no_grad()
def measure_trm_flops(model, sample, H_cycles: int = None, L_cycles: int = None) -> dict:
    """FLOPs of the three sub-calls forward_with_carry is built from, each
    measured once on a real (B=1) sample:

      F_encode       — model._encode_condition(sample), runs once per
                        forward_with_carry call (i.e. once per denoising
                        step), independent of n_sup.
      F_thinker_step — one model.thinker.reasoning_step(...) call (already
                        includes that call's H_cycles*L_cycles internal
                        work — H_cycles/L_cycles default to the thinker's
                        trained config, or pass explicit overrides to
                        profile a reduced-recursion-depth config, e.g. for
                        experiments/sweep_trm_flops_vs_acc.py's low-compute
                        curve segment) — runs n_sup times per
                        forward_with_carry call.
      F_painter      — model.run_painter(sample, logits), runs exactly once
                        per forward_with_carry call regardless of n_sup
                        (translator + frozen painter UNet).

    A sweep config's total FLOPs = num_steps-worth of
    (F_encode + n_sup_i * F_thinker_step + F_painter), doubled under CFG —
    see the two sweep scripts for the exact per-config formulas (flat vs.
    the steering-freeze/carry "wise" schedule).
    """
    bsz = sample.x_noisy.shape[0]
    z_H, z_L = model.get_initial_states(bsz)
    z_H, z_L = z_H.to(sample.x_noisy.device), z_L.to(sample.x_noisy.device)

    f_encode = _measure(lambda: model._encode_condition(sample))
    enc_emb = model._encode_condition(sample)
    puzzle_ids = sample.puzzle_id

    f_thinker_step = _measure(
        lambda: model.thinker.reasoning_step(
            enc_emb, z_H, z_L, puzzle_ids, timesteps=sample.timesteps, H_cycles=H_cycles, L_cycles=L_cycles,
        )
    )
    logits, _, _ = model.thinker.reasoning_step(
        enc_emb, z_H, z_L, puzzle_ids, timesteps=sample.timesteps, H_cycles=H_cycles, L_cycles=L_cycles,
    )

    f_painter = _measure(lambda: model.run_painter(sample, logits))

    return {"F_encode": f_encode, "F_thinker_step": f_thinker_step, "F_painter": f_painter}
