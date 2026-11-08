"""
experiments/z_scratchpad_probe.py — is the TRM's fast recurrent state z_L a
committed partial solution, or a disposable scratchpad the model is free to
"go crazy" in because only the slow state z_H (which alone is ever decoded
into the answer) has to end up valid?

Mechanism
---------
Per reasoning iteration (one call to SpatialTRM.reasoning_step), the vendored
loop (models/trm/recursive_reasoning/trm.py:204-217) does, using the SAME
module `inner.L_level` for both updates:

    for h in range(H_cycles):
        for l in range(L_cycles):
            z_L = L_level(z_L, z_H + input_embeddings)   # fast update
        z_H = L_level(z_H, z_L)                          # slow update

Only z_H is ever decoded (via lm_head) into logits; z_L has no decode path
of its own. Neither H_cycles sub-steps nor L_cycles sub-steps escape
reasoning_step — it returns only the final z_H/z_L of the whole n_sup
iteration. To read them without altering the real trajectory (no
reimplementing the loop, which would risk silently drifting from the real
detach/dtype/no_grad behavior), this script registers a forward hook on
`inner.L_level` and lets the real forward_with_carry call run untouched. A
hook sees (module, args, output) for BOTH the fast and slow update calls;
since the two are interleaved in an exactly known pattern (L_cycles fast
calls, then 1 slow call, repeated H_cycles times, repeated n_sup times), the
call list can be sliced back into structured (k, h, l) positions with no
ambiguity — see `_structure_calls`.

This gives us, per (k, h) reasoning position:
  - z_H_before — the slow state that was held fixed while L_cycles fast
    updates ran (captured as the slow-update call's own input).
  - z_L snapshots, one per l in [0, L_cycles) — the fast state's value
    after each individual fast update.
  - z_H_after — the slow state's real value once this H_cycle's slow
    update actually ran (free bonus: finer than the existing zH_out/
    logits_out hooks, which only expose z_H once per n_sup iteration, not
    once per H_cycle).

Decoding:
  - z_H_after decodes directly (no counterfactual needed — it's the model's
    own real slow state): logits = lm_head(z_H_after). This is the "y_fine"
    curve — n_sup * H_cycles points, strictly finer than the y curve in
    candidate_trajectory_probe.py (which only sees the last of these
    per n_sup iteration).
  - A z_L snapshot has no decode path of its own. Per the user's own
    recipe: run ONE hypothetical slow update using this z_L snapshot and
    the CURRENT (not-yet-updated) z_H, then decode that: logits_hyp =
    lm_head(L_level(z_H_before, z_L_snapshot)). This is the "z" curve —
    n_sup * H_cycles * L_cycles points. Cheap because the slow update is a
    pure function of (z_H, z_L) plus a constant positional table — no
    re-encoding of the sample, no re-running any earlier part of the
    network.

Both curves are folded through run_painter (+ CFG combine on images, same
as candidate_trajectory_probe.py) and scored with evaluate_grids, plus two
extra metrics:
  violations — graded 0-27 sudoku constraint-violation count (from
    violation_trajectory_probe.py's _count_violating_units), finer signal
    than evaluate_grids' binary constraint_puzzle_acc.
  entropy    — mean per-cell softmax entropy of the TRM's own logits
    (pre-painter, vocab_size classes), a token-level confidence measure
    decoupled from the painter/classifier.

Repair/regression transition rates (repair_transition_probe.py's
_transition_counts, reused verbatim) are computed over the z sequence and
the y_fine sequence. Within z, transitions are further split into
  within_h  — consecutive l -> l+1 pairs, same (k, h): z_H is NOT changing
              between these two points, so any repair/regression here is
              pure z-side churn under a fixed answer.
  h_boundary / k_boundary — the last l of one (k, h) group to the first l
              of the next: z_H just changed (h_boundary) or a full n_sup
              iteration just completed (k_boundary).
If the scratchpad hypothesis holds, within_h churn (both repair AND
regression) should be high relative to y_fine's transition rates, and
given_consistent_puzzle_acc for z should be far below y_fine's — clue cells
are only ever supervised through y_H, never through z_L directly.

No corruption involved — this is the plain generative trajectory, same as
violation_trajectory_probe.py.

Usage:
    python experiments/z_scratchpad_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +probe.num_samples=64

    # TRM-alone (TRMDiffusionBackbone) works identically:
    python experiments/z_scratchpad_probe.py \\
      experiment=mnist_trm_diffusion_backbone \\
      checkpoint=runs/mnist_trm_diffusion_backbone/checkpoint_final.pt \\
      +probe.num_samples=64

Cost warning: this decodes+paints+classifies n_sup*H_cycles*(L_cycles+1)
candidates per instrumented denoising step (e.g. 16*3*7=336 with the
default H_cycles=3/L_cycles=6/n_sup=16) — far more than
candidate_trajectory_probe.py's n_sup per step. Defaults to a smaller
num_samples and a single instrumented denoise step; widen
probe.denoise_fractions deliberately.

Options (all under +probe.*):
    num_samples        — default 64
    seed               — default 0
    reset_every        — default 1 (see ablate_trm_loop_budget._make_reset_fn)
    denoise_fractions  — default [0.5] (single mid-trajectory step)
    cfg_scale / num_inference_steps — default from model.sampling_pipeline
    out                — json path
"""

from __future__ import annotations

import json
import logging
import os
import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from candidate_trajectory_probe import _bucket_labels
from repair_transition_probe import _accumulate, _rates_from_counts, _transition_counts
from violation_trajectory_probe import _count_violating_units
from eval.mnist_eval import evaluate_grids, _check_sudoku_constraints
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


class _LLevelCapture:
    """Forward hook on `inner.L_level` — records (old_state, new_state) for
    EVERY call (both fast/z_L updates and slow/z_H updates), in call order,
    without altering anything the real forward pass computes or returns."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._handle = None

    def __enter__(self):
        def hook(module, args, output):
            self.calls.append((args[0].detach(), output.detach()))

        self._handle = self.inner.L_level.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


def _structure_calls(calls: list, n_sup: int, H_cycles: int, L_cycles: int) -> list:
    """calls: flat list of (old_state, new_state), length
    n_sup * H_cycles * (L_cycles + 1), in the exact order trm.py's forward
    issues them (L_cycles fast calls then 1 slow call, repeated H_cycles
    times, repeated n_sup times). Returns structured[k][h] = dict(
    z_H_before, z_L: list[L_cycles], z_H_after)."""
    per_group = L_cycles + 1
    expected = n_sup * H_cycles * per_group
    if len(calls) != expected:
        raise RuntimeError(f"expected {expected} L_level calls, got {len(calls)} — H_cycles/L_cycles/n_sup mismatch?")

    structured = []
    idx = 0
    for _k in range(n_sup):
        groups = []
        for _h in range(H_cycles):
            z_l_snapshots = []
            for _l in range(L_cycles):
                _, z_l_out = calls[idx]
                idx += 1
                z_l_snapshots.append(z_l_out)
            z_h_before, z_h_after = calls[idx]
            idx += 1
            groups.append({"z_H_before": z_h_before, "z_L": z_l_snapshots, "z_H_after": z_h_after})
        structured.append(groups)
    return structured


def _lm_head_decode(inner, z_H: torch.Tensor) -> torch.Tensor:
    return inner.lm_head(z_H)[:, inner.puzzle_emb_len:]


def _hypothetical_decode(inner, z_H_before: torch.Tensor, z_L_snapshot: torch.Tensor) -> torch.Tensor:
    """The user's own recipe: one hypothetical slow update using this z_L
    snapshot and the CURRENT (not-yet-updated) z_H, then decode normally."""
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    z_H_hyp = inner.L_level(z_H_before, z_L_snapshot, cos_sin=cos_sin)
    return _lm_head_decode(inner, z_H_hyp)


def _entropy(logits: torch.Tensor) -> float:
    """logits: (B, seq_len, vocab_size) -> scalar mean per-cell softmax entropy."""
    p = logits.float().softmax(dim=-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(dim=-1)
    return ent.mean().item()


def _per_sample_metrics(preds: torch.Tensor, solutions: torch.Tensor, given_masks) -> dict:
    """preds/solutions: (B, 81). Per-sample (not batch-averaged) versions of
    evaluate_grids' metrics, computed directly from an already-decoded preds
    tensor (no extra classifier/painter calls) — reimplemented rather than
    reusing evaluate_grids because we need every sample's own value, not a
    pre-reduced batch mean, to ask "is THIS puzzle's worst intermediate state
    worse than THIS puzzle's own previous y". Formulas match
    eval/mnist_eval.py's evaluate_grids exactly."""
    sol = solutions.to(preds.device)
    correct = preds == sol
    violations = _count_violating_units(preds)
    valid_grid = _check_sudoku_constraints(preds)
    if given_masks is not None:
        gm = given_masks.to(preds.device)
        blank = ~gm
        n_blank = blank.sum(dim=1).clamp(min=1)
        cell_acc = (correct & blank).sum(dim=1).float() / n_blank.float()
        given_ok = (correct | blank).all(dim=1)
        given_cons = (valid_grid & given_ok).float()
    else:
        cell_acc = correct.float().mean(dim=1)
        given_cons = None
    return {"violations": violations, "cell_acc": cell_acc, "given_cons": given_cons}


def _group_dip_counts(y_prev_preds: torch.Tensor, group_preds: list, solutions: torch.Tensor, given_masks) -> dict:
    """y_prev_preds: (B, 81) — the y this group of z's was decoded relative
    to (the one it started from, BEFORE this group's H_cycle worth of
    reasoning ran). group_preds: list of L_cycles (B, 81) tensors — every
    z decoded within that group.

    Per sample, compares the group's WORST point (max violations / min
    cell_acc / min given-consistency across all L_cycles z's) against the
    starting y — not the group's average or its ending y. If the model
    only ever refines/interpolates toward the next y, the worst z within a
    group should never be worse than where it started; a nonzero dip rate
    here is evidence it actually passes through states worse than its own
    starting point — real search, not monotonic smoothing.

    given_cons dip only counts samples where the previous y WAS
    given-consistent (nothing to regress from otherwise)."""
    prev = _per_sample_metrics(y_prev_preds, solutions, given_masks)
    per_l = [_per_sample_metrics(p, solutions, given_masks) for p in group_preds]

    max_viol = torch.stack([m["violations"] for m in per_l], dim=0).max(dim=0).values
    min_cell_acc = torch.stack([m["cell_acc"] for m in per_l], dim=0).min(dim=0).values

    dip_viol = max_viol > prev["violations"]
    dip_cell_acc = min_cell_acc < prev["cell_acc"]

    out = {
        "n": prev["violations"].shape[0],
        "dip_viol_count": int(dip_viol.sum().item()),
        "dip_viol_magnitude_sum": float((max_viol - prev["violations"]).clamp(min=0).float().sum().item()),
        "dip_cell_acc_count": int(dip_cell_acc.sum().item()),
        "dip_cell_acc_magnitude_sum": float((prev["cell_acc"] - min_cell_acc).clamp(min=0).sum().item()),
    }
    if prev["given_cons"] is not None:
        min_given_cons = torch.stack([m["given_cons"] for m in per_l], dim=0).min(dim=0).values
        eligible = prev["given_cons"] > 0.5
        dip_given_cons = eligible & (min_given_cons < 0.5)
        out["dip_given_cons_count"] = int(dip_given_cons.sum().item())
        out["given_cons_eligible"] = int(eligible.sum().item())
    return out


def _dip_rates_from_counts(c: dict) -> dict:
    n = c.get("n", 0)
    out = {
        "n": n,
        "viol_dip_rate": c["dip_viol_count"] / n if n else float("nan"),
        "viol_dip_magnitude_mean": c["dip_viol_magnitude_sum"] / n if n else float("nan"),
        # "_given_dip": conditional on a dip actually happening (divides by
        # dip_count, not n) — decouples "how often" (rate) from "how bad
        # when it happens" (this), which the unconditional _mean above
        # blends together (it equals rate × this value). nan when
        # dip_count==0 (no dips happened here): "0 dips -> 0 magnitude" is
        # arithmetically defensible but a misleading VALUE — it reads as
        # "the dip that didn't happen would have been zero-sized", which
        # isn't a meaningful statement, and visually whipsaws a plotted
        # curve between 0 and whatever nearby groups' real (nonzero)
        # conditional magnitude is. Downstream consumers (e.g.
        # paper_plots/plot_z_scratchpad_search.py) should treat this as
        # missing and interpolate across it, not substitute 0 — rate above
        # already carries the correct, well-defined "how often" signal for
        # a dip_count==0 group.
        "viol_dip_magnitude_given_dip": (
            c["dip_viol_magnitude_sum"] / c["dip_viol_count"] if c["dip_viol_count"] else float("nan")
        ),
        "cell_acc_dip_rate": c["dip_cell_acc_count"] / n if n else float("nan"),
        "cell_acc_dip_magnitude_mean": c["dip_cell_acc_magnitude_sum"] / n if n else float("nan"),
        "cell_acc_dip_magnitude_given_dip": (
            c["dip_cell_acc_magnitude_sum"] / c["dip_cell_acc_count"] if c["dip_cell_acc_count"] else float("nan")
        ),
    }
    if "dip_given_cons_count" in c:
        elig = c.get("given_cons_eligible", 0)
        out["given_cons_dip_rate"] = c["dip_given_cons_count"] / elig if elig else float("nan")
        out["given_cons_eligible_frac"] = elig / n if n else float("nan")
    return out


def _mean_dicts(dicts: list) -> dict:
    """Mean of a list of same-keyed scalar dicts — used to roll per-(k, h)
    group dip rates up to per-k (mean over h) or an overall headline (mean
    over all groups). Every group at a given denoise step sees the exact
    same pooled sample count, so this is equivalent to re-pooling raw counts
    — done as a rate-mean instead purely to avoid re-summing "n" across
    groups that all refer to the SAME samples (which would silently inflate
    denominators if counts were re-accumulated instead). Uses nanmean since
    given_cons_dip_rate is nan whenever a group had zero eligible samples
    (its own starting y was already given-inconsistent for everyone) — a
    plain mean would let that one group's nan blank out the whole rollup."""
    keys = dicts[0].keys()
    return {k: float(np.nanmean([d[k] for d in dicts])) for k in keys}


@torch.no_grad()
def _decode_and_score(
    model, classifier, cell_size, step_sample, null_sample, cfg_scale, logits_c, logits_u, solutions, given_masks,
    return_image: bool = False,
) -> dict:
    """return_image: also stash the decoded (B, 1, H, W) image tensor under
    m["image"] — off by default (every caller in the quantitative sweeps
    calls this many times per sample and never needs the image, so keeping
    it would only bloat memory). Callers that DO need to display exactly
    what was classified (not a separately re-rendered approximation of it —
    see qualitative_dip_example.py, where a from-scratch single-sample
    re-render was found to diverge from the batched scan's own classifier
    read after enough denoising steps) should pass True and use m["image"]
    directly instead of re-deriving it."""
    cand = model.run_painter(step_sample, logits_c)
    if cfg_scale != 1.0:
        cand_u = model.run_painter(null_sample, logits_u)
        cand = cand_u + cfg_scale * (cand - cand_u)
    cand = model.decode_for_eval(cand)
    m = evaluate_grids(cand, solutions, classifier, cell_size, given_masks=given_masks)
    m["entropy"] = _entropy(logits_c)
    m["violations"] = _count_violating_units(m["preds"]).float().mean().item()
    if return_image:
        m["image"] = cand
    return m


@torch.no_grad()
def _run_and_capture(
    model, classifier, cell_size, conditions, solutions, given_masks, x_init,
    num_inference_steps, cfg_scale, reset_fn, instrumented_idx: set,
) -> tuple[torch.Tensor, dict]:
    """Real trajectory — x always advances using the real final CFG-combined
    prediction, exactly like real sampling. At instrumented denoising steps,
    additionally hooks inner.L_level to recover and decode every (k, h, l)
    position. Returns (final_image, curves) where curves[step_idx] =
    {"z": [...], "y_fine": [...]} — flat lists of evaluate_grids()+extra
    dicts, in (k, h, l) / (k, h) order respectively."""
    device = x_init.device
    reasoner = getattr(model, "thinker", model)
    inner = reasoner.inner
    H_cycles = inner.config.H_cycles
    L_cycles = inner.config.L_cycles
    n_sup = model.n_sup

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None
    curves: dict = {}

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        instrument = step_idx in instrumented_idx

        if instrument:
            with _LLevelCapture(inner) as cap_c:
                pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c)
            cond_calls = cap_c.calls
        else:
            pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c)
        noise_pred = pred_c.pred

        null_sample = None
        uncond_calls = None
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            if instrument:
                with _LLevelCapture(inner) as cap_u:
                    pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u)
                uncond_calls = cap_u.calls
            else:
                pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        if instrument:
            groups_c = _structure_calls(cond_calls, n_sup, H_cycles, L_cycles)
            groups_u = _structure_calls(uncond_calls, n_sup, H_cycles, L_cycles) if uncond_calls is not None else None

            # y_initial: the z_H carried INTO this denoising step (before any
            # reasoning this step) — decoded directly (real, not
            # counterfactual) so group 0's dip check has a "previous y" to
            # compare against. groups_c[0][0]["z_H_before"] already equals
            # this value exactly, via the hook, regardless of whether it came
            # from a fresh reset or a carried-over previous-step state.
            logits_y0_c = _lm_head_decode(inner, groups_c[0][0]["z_H_before"])
            logits_y0_u = _lm_head_decode(inner, groups_u[0][0]["z_H_before"]) if groups_u is not None else None
            y_initial_point = _decode_and_score(
                model, classifier, cell_size, step_sample, null_sample, cfg_scale,
                logits_y0_c, logits_y0_u, solutions, given_masks,
            )

            z_points, y_fine_points = [], []
            for k in range(n_sup):
                for h in range(H_cycles):
                    g_c = groups_c[k][h]
                    g_u = groups_u[k][h] if groups_u is not None else None

                    logits_y_c = _lm_head_decode(inner, g_c["z_H_after"])
                    logits_y_u = _lm_head_decode(inner, g_u["z_H_after"]) if g_u is not None else None
                    y_fine_points.append(_decode_and_score(
                        model, classifier, cell_size, step_sample, null_sample, cfg_scale,
                        logits_y_c, logits_y_u, solutions, given_masks,
                    ))

                    for l in range(L_cycles):
                        logits_z_c = _hypothetical_decode(inner, g_c["z_H_before"], g_c["z_L"][l])
                        logits_z_u = (
                            _hypothetical_decode(inner, g_u["z_H_before"], g_u["z_L"][l]) if g_u is not None else None
                        )
                        z_points.append(_decode_and_score(
                            model, classifier, cell_size, step_sample, null_sample, cfg_scale,
                            logits_z_c, logits_z_u, solutions, given_masks,
                        ))

            # dip_by_group[g] (g = k*H_cycles + h): compares group g's own
            # worst-of-L_cycles z's against the y it started from — NOT its
            # ending y. y_prev_points[g] is that starting y: group 0 starts
            # from y_initial_point; every later group starts from the
            # previous group's own y_fine result (its z_H_after).
            y_prev_points = [y_initial_point] + y_fine_points[:-1]
            dip_by_group = []
            for g in range(n_sup * H_cycles):
                group_preds = [z_points[g * L_cycles + l]["preds"] for l in range(L_cycles)]
                dip_by_group.append(
                    _group_dip_counts(y_prev_points[g]["preds"], group_preds, solutions, given_masks)
                )

            curves[step_idx] = {
                "z": z_points, "y_fine": y_fine_points, "dip_by_group": dip_by_group,
                "H_cycles": H_cycles, "L_cycles": L_cycles, "n_sup": n_sup,
            }

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return x, curves


def _average_positionwise(all_curves: list, sample_counts: list) -> list:
    """all_curves[i] is a length-N list of per-position dicts for cached
    batch i; averages across batches (sample-count weighted) into a
    length-N list of dicts. Mirrors candidate_trajectory_probe._average_curves."""
    n_positions = len(all_curves[0])
    out = []
    for pos in range(n_positions):
        metrics: dict[str, list] = {}
        for curves, n in zip(all_curves, sample_counts):
            for key, val in curves[pos].items():
                if key == "preds" or val is None:
                    continue
                metrics.setdefault(key, []).append((val, n))
        out.append({
            key: float(np.average([v for v, _ in pairs], weights=[n for _, n in pairs]))
            for key, pairs in metrics.items()
        })
    return out


def _pool_transitions(pooled: dict, points: list, solutions, given_masks, key_fn) -> None:
    """Accumulates _transition_counts for every consecutive pair in
    `points`, bucketed by key_fn(i) (i = index of the left element of the
    pair) into pooled[key]."""
    for i in range(len(points) - 1):
        key = key_fn(i)
        delta = _transition_counts(points[i]["preds"], points[i + 1]["preds"], solutions, given_masks)
        pooled.setdefault(key, {})
        _accumulate(pooled[key], delta)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/z_scratchpad_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 64)
    seed: int = pb.get("seed", 0)
    reset_every = pb.get("reset_every", 1)
    denoise_fractions: list[float] = list(pb.get("denoise_fractions", [0.5]))
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "z_scratchpad.json"))

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
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    reasoner = getattr(model, "thinker", model)
    H_cycles = reasoner.inner.config.H_cycles
    L_cycles = reasoner.inner.config.L_cycles

    labels = _bucket_labels(denoise_fractions)
    idx_to_label = {round(f * (num_inference_steps - 1)): label for f, label in zip(denoise_fractions, labels)}
    instrumented_idx = sorted(idx_to_label.keys())

    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  trained n_sup={model.n_sup}  "
        f"H_cycles={H_cycles}  L_cycles={L_cycles}  "
        f"instrumented denoise steps={instrumented_idx} ({[idx_to_label[i] for i in instrumented_idx]})  "
        f"reset_every={reset_every}  "
        f"({model.n_sup * H_cycles * (L_cycles + 1)} decodes/step — this is expensive)"
    )

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)
    reset_fn = _make_reset_fn(reset_every)

    all_z_curves: dict = {i: [] for i in instrumented_idx}
    all_y_curves: dict = {i: [] for i in instrumented_idx}
    sample_counts: list[int] = []
    pooled_z_transitions: dict = {i: {} for i in instrumented_idx}
    pooled_y_transitions: dict = {i: {} for i in instrumented_idx}
    # *_by_pos: same _transition_counts, but keyed by the literal transition
    # index instead of collapsed into the 3 within_h/h_boundary/k_boundary
    # buckets — gives a full repair_rate/regression_rate curve over every
    # position, for plotting against the combined progression axis.
    pooled_z_transitions_by_pos: dict = {i: {} for i in instrumented_idx}
    pooled_y_transitions_by_pos: dict = {i: {} for i in instrumented_idx}
    pooled_dip: dict = {i: {} for i in instrumented_idx}  # step_idx -> group index g -> pooled dip counts

    for bi, cb in enumerate(cached_batches):
        _, curves = _run_and_capture(
            model, classifier, cell_size, cb["conditions"], cb["solutions"], cb["given_masks"], cb["x_init"],
            num_inference_steps, cfg_scale, reset_fn, set(instrumented_idx),
        )
        n = cb["solutions"].shape[0]
        sample_counts.append(n)
        for step_idx in instrumented_idx:
            z_points = curves[step_idx]["z"]
            y_points = curves[step_idx]["y_fine"]
            all_z_curves[step_idx].append(z_points)
            all_y_curves[step_idx].append(y_points)

            def z_bucket(i: int, _L=L_cycles) -> str:
                # i is the index of the LEFT element of a (i, i+1) pair,
                # flattened over (k, h, l). within_h: still inside the same
                # (k, h) group (l -> l+1). h_boundary: last l of one (k, h)
                # group to the next h's l=0 (z_H just changed). k_boundary:
                # additionally crosses an n_sup iteration boundary.
                if (i + 1) % _L != 0:
                    return "within_h"
                group_idx = (i + 1) // _L
                return "k_boundary" if group_idx % H_cycles == 0 else "h_boundary"

            _pool_transitions(pooled_z_transitions[step_idx], z_points, cb["solutions"], cb["given_masks"], z_bucket)
            _pool_transitions(
                pooled_y_transitions[step_idx], y_points, cb["solutions"], cb["given_masks"], lambda i: "all"
            )
            _pool_transitions(
                pooled_z_transitions_by_pos[step_idx], z_points, cb["solutions"], cb["given_masks"], lambda i: i
            )
            _pool_transitions(
                pooled_y_transitions_by_pos[step_idx], y_points, cb["solutions"], cb["given_masks"], lambda i: i
            )

            for g, d in enumerate(curves[step_idx]["dip_by_group"]):
                pooled_dip[step_idx].setdefault(g, {})
                _accumulate(pooled_dip[step_idx][g], d)
        logger.info(f"  batch {bi + 1}/{len(cached_batches)} done ({sum(sample_counts)} samples)")

    results: dict = {}
    for step_idx in instrumented_idx:
        label = idx_to_label[step_idx]
        z_curve = _average_positionwise(all_z_curves[step_idx], sample_counts)
        y_curve = _average_positionwise(all_y_curves[step_idx], sample_counts)

        def _mean_across_positions(curve: list) -> dict:
            keys = curve[0].keys()
            return {k: float(np.mean([p[k] for p in curve])) for k in keys}

        n_sup = model.n_sup
        dip_by_group_rates = [_dip_rates_from_counts(pooled_dip[step_idx][g]) for g in range(n_sup * H_cycles)]
        dip_by_k = [_mean_dicts(dip_by_group_rates[k * H_cycles:(k + 1) * H_cycles]) for k in range(n_sup)]
        dip_headline = _mean_dicts(dip_by_group_rates)

        results[label] = {
            "z_curve": z_curve,
            "y_fine_curve": y_curve,
            "z_headline": _mean_across_positions(z_curve),
            "y_fine_headline": _mean_across_positions(y_curve),
            "z_transitions": {k: _rates_from_counts(v) for k, v in pooled_z_transitions[step_idx].items()},
            "y_fine_transitions": {k: _rates_from_counts(v) for k, v in pooled_y_transitions[step_idx].items()},
            "z_transitions_by_position": [
                _rates_from_counts(pooled_z_transitions_by_pos[step_idx][i])
                for i in range(len(pooled_z_transitions_by_pos[step_idx]))
            ],
            "y_fine_transitions_by_position": [
                _rates_from_counts(pooled_y_transitions_by_pos[step_idx][i])
                for i in range(len(pooled_y_transitions_by_pos[step_idx]))
            ],
            "dip_by_group": dip_by_group_rates,
            "dip_by_k": dip_by_k,
            "dip_headline": dip_headline,
        }

    if accelerator.is_main_process:
        for label, r in results.items():
            print("\n" + "=" * 90)
            print(f"bucket={label}  (z: n_sup*H_cycles*L_cycles={len(r['z_curve'])} pts, "
                  f"y_fine: n_sup*H_cycles={len(r['y_fine_curve'])} pts)")
            print("=" * 90)
            print(f"{'':>10}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}{'given_cons':>12}{'violations':>12}{'entropy':>10}")
            for name, h in (("z", r["z_headline"]), ("y_fine", r["y_fine_headline"])):
                print(
                    f"{name:>10}{h['cell_acc']:>10.4f}{h['puzzle_acc']:>12.4f}{h['constraint_puzzle_acc']:>12.4f}"
                    f"{h.get('given_consistent_puzzle_acc', float('nan')):>12.4f}{h['violations']:>12.4f}{h['entropy']:>10.4f}"
                )
            print(f"\n  z transitions (by pair type):")
            for key, tr in r["z_transitions"].items():
                print(f"    {key:>12}  repair_rate={tr['repair_rate']:.4f}  regression_rate={tr['regression_rate']:.4f}  net_rate={tr['net_rate']:.4f}")
            print(f"  y_fine transitions:")
            for key, tr in r["y_fine_transitions"].items():
                print(f"    {key:>12}  repair_rate={tr['repair_rate']:.4f}  regression_rate={tr['regression_rate']:.4f}  net_rate={tr['net_rate']:.4f}")

            dh = r["dip_headline"]
            print(f"\n  dip-vs-previous-y (worst of each group's L_cycles z's vs the y the group STARTED from, not its ending y):")
            line = (
                f"    headline  viol_dip_rate={dh['viol_dip_rate']:.4f}  viol_dip_magnitude_mean={dh['viol_dip_magnitude_mean']:.4f}  "
                f"cell_acc_dip_rate={dh['cell_acc_dip_rate']:.4f}  cell_acc_dip_magnitude_mean={dh['cell_acc_dip_magnitude_mean']:.4f}"
            )
            if "given_cons_dip_rate" in dh:
                line += f"  given_cons_dip_rate={dh['given_cons_dip_rate']:.4f} (of {dh['given_cons_eligible_frac']:.2f} eligible)"
            print(line)
            print(f"  dip_by_k (n_sup iteration k=1..{len(r['dip_by_k'])}, averaged over the {H_cycles} H_cycles within each k):")
            for k, dk in enumerate(r["dip_by_k"]):
                line = f"    k={k + 1:>2}  viol_dip_rate={dk['viol_dip_rate']:.4f}  cell_acc_dip_rate={dk['cell_acc_dip_rate']:.4f}"
                if "given_cons_dip_rate" in dk:
                    line += f"  given_cons_dip_rate={dk['given_cons_dip_rate']:.4f}"
                print(line)
        print("=" * 90)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": sum(sample_counts), "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()