"""
experiments/export_sampling_animation.py — export per-frame data for an
interactive "how does the model sample/reason" animation, for several
puzzles per checkpoint.

Design (v2, revised after viewing v1):

  - The painter is only decoded ONCE per denoising step, using the same
    final logits the real trajectory uses to advance x — not once per
    reasoning iteration. v1 decoded every H_cycles-th iteration to show
    "progressive" candidates, but that made the image appear to change
    faster than the timestep itself, which is confusing: the real model
    only ever produces one committed image per denoising step. The z_H
    probe panel is still read every fine iteration (that's free — a linear
    probe, no painter call), so the reasoning dynamics are still visible;
    only the rendered image is now tied strictly to timestep.

  - Only the ARGMAX predicted digit is stored per cell (uint8), not a full
    9-way softmax — the viewer no longer draws probability bars ("logits"),
    only correct/incorrect/given cell marking, which only needs the digit.
    This is roughly a 9x reduction in the dominant per-frame payload and is
    what makes exporting several puzzles per checkpoint fit the same size
    budget v1 used for one.

  - "halt" granularity trajectories now stop for real once the conditional
    branch halts, instead of padding to n_sup with repeated frames (v1's
    bug — it copied probe_reasoning_dynamics.py's padding convention,
    which exists there only so aggregate accumulator arrays have a uniform
    shape; a single-sample visual export has no such constraint, and
    padding was hiding the real, variable, per-timestep halting behavior
    the halt-threshold ablation exists to show).

  - Puzzle selection (v3) scans the pool with the model's own native
    forward_with_carry(use_halt_head=True) machinery, carry NEVER reset
    across denoising steps, judged on the REAL painted image (not the
    digit probe, which turned out to be unreliable, especially on
    "hard"). This is cheap enough — most samples exit well before n_sup
    reasoning iterations under halting — to scan a large pool (1000+)
    directly, batched, with no separate coarse pre-filter needed. Among
    puzzles that end up solved (valid, given-consistent final grid — see
    _derive_ground_truth), it prefers ones wrong for MANY denoising steps,
    not just a late single one (a real "struggle then resolve" story, not
    a late noise flip). See _rank_candidates / _run_batched_halt_carry_scan.

  - CONFIGS now includes "halt_carry": halting WITH the carry never reset
    (previously the halt path always reset every step, the only granularity
    the halt head was actually trained at — this combination is an
    additional out-of-training-distribution ablation, same spirit as
    "never reset" already is for the non-halting case).

Painter decode (run_painter) is the expensive part, so it is now called
exactly num_inference_steps times per trajectory regardless of granularity
— see _thinker_step (reasoning only, no paint) vs _paint_logits (paint
only, called once per step per CFG branch).

Usage:
    python experiments/export_sampling_animation.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      thinker.with_halt_head=true \\
      +checkpoint=runs/sudoku_painter_thinker_with_halt_head.pt \\
      +probe_checkpoint=runs/digit_probe_sudoku_painter_thinker.pt \\
      +use_ema=false \\
      +viz.tag=easy +viz.num_puzzles=3 +viz.out=runs/sampling_animation_easy.json
"""

from __future__ import annotations

import base64
import dataclasses
import io
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
from PIL import Image
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint, _make_reset_fn
from eval.mnist_eval import _check_sudoku_constraints
from factory import build_datasets, build_model
from models.digit_probe import DigitProbe

logger = get_logger(__name__, log_level="INFO")


# ── Reasoning-only / paint-only steps (kept separate so per-iteration probe
#    reads don't pay for a painter forward pass they don't need) ───────────

def _thinker_step(model, sample, z_H, z_L, H_cycles=None):
    """One reasoning_step, no painting. Returns (logits, z_H_next, z_L_next)."""
    enc_emb = model._encode_condition(sample)
    return model.thinker.reasoning_step(
        enc_emb, z_H, z_L, sample.puzzle_id, timesteps=sample.timesteps, H_cycles=H_cycles,
    )


def _paint_logits(model, sample, logits):
    scale_fn = getattr(model.loss_fn, "scale_logits_for_painter", None)
    logits_for_tpt = scale_fn(logits) if scale_fn is not None else logits
    return model.run_painter(sample, logits_for_tpt)


@torch.no_grad()
def _run_visual_trajectory(
    model, probe: DigitProbe, classifier, cell_size: int, puzzle_emb_len: int,
    conditions, x_init: torch.Tensor, num_inference_steps: int, cfg_scale: float,
    granularity: str, reset_fn, halt_threshold: float = 0.0,
) -> dict:
    """Runs one real single-sample (B=1) generation trajectory. x always
    advances using the same final-iteration CFG-combined prediction real
    inference uses — nothing here changes what gets generated, it only
    records intermediates forward_with_carry's own hooks don't expose for
    the halted case (see module docstring) or at fine granularity.

    granularity: "fine" (n_sup*H_cycles reasoning iterations/denoising
    step, H_cycles=1 — no halting) or "halt" (up to n_sup iterations/
    denoising step, per-sample early exit via the trained halt head).
    reset_fn controls carry across denoising steps for EITHER granularity
    — reset_fn always-true reproduces the halt head's actual training
    distribution (fresh state every step); reset_fn=never lets the carry
    persist into a halting call too ("halt_carry"), which is out of the
    halt head's training distribution the same way "never reset" already
    is for the non-halting case, and is fine as an ablation. The
    conditional and unconditional CFG branches halt independently, matching
    real batched behaviour; only the conditional branch's z_H is probed/
    recorded (the null branch reasons about a zeroed condition and isn't
    meaningful to read).

    Returns a dict with:
      x_noisy_frames, decoded_frames: list of length num_inference_steps,
        (H, W) uint8 numpy arrays — the image is only ever decoded once
        per denoising step, using the SAME final logits used to advance x.
      classifier_pred: (num_inference_steps, 81) uint8 — classifier's
        argmax digit (0-8) on each decoded_frames entry.
      probe_pred: (n_fine_frames, 81) uint8 — digit probe's argmax digit
        at every reasoning iteration (fine-grained for "fine"; real,
        un-padded per-sample halting length for "halt").
      frame_denoise_step, frame_within_step: (n_fine_frames,) int.
      steps_used_per_denoise_step: (num_inference_steps,) int — how many
        conditional-branch reasoning iterations actually ran this step
        (== n_sup*H_cycles always for "fine"; <= n_sup, real halting
        point, for "halt").
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    n_sup = model.n_sup
    H_cycles_real = model.thinker.inner.config.H_cycles
    do_cfg = cfg_scale != 1.0
    max_iters = (n_sup * H_cycles_real) if granularity == "fine" else n_sup
    h_override = 1 if granularity == "fine" else None

    z_H_c = z_L_c = None
    z_H_u = z_L_u = None

    x_noisy_frames, decoded_frames = [], []
    classifier_pred_list, probe_pred_list = [], []
    frame_denoise_step, frame_within_step = [], []
    steps_used_per_denoise_step = []

    for step_idx, t in enumerate(model.scheduler.timesteps):
        t_batch = t.expand(1).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        null_sample = model.null_condition_sample(step_sample) if do_cfg else None

        if reset_fn(step_idx):
            z_H_c = z_L_c = None
            z_H_u = z_L_u = None

        if z_H_c is None:
            z_H_c, z_L_c = model.get_initial_states(1)
            z_H_c, z_L_c = z_H_c.to(device), z_L_c.to(device)
        if do_cfg and z_H_u is None:
            z_H_u, z_L_u = model.get_initial_states(1)
            z_H_u, z_L_u = z_H_u.to(device), z_L_u.to(device)

        # decode_for_eval, not a raw clamp — x lives in the painter's own
        # pixel convention (e.g. [-1,1] for some experiments), and a raw
        # clamp(0,1) silently discards its entire negative half, which is
        # why the "noisy canvas" panel looked wrong (worse, not better, as
        # denoising progressed): decode_for_eval applies the same
        # pixel_range rescale used for every other rendered image here.
        x_noisy_frames.append(_to_uint8_image(model.decode_for_eval(x)[0]))

        # conditional branch: drives the recorded probe frames, halts on its own
        last_logits_c = None
        k_used = 0
        for k in range(max_iters):
            last_logits_c, z_H_c, z_L_c = _thinker_step(model, step_sample, z_H_c, z_L_c, H_cycles=h_override)
            k_used = k + 1
            probe_logits = probe(z_H_c[:, puzzle_emb_len:])
            probe_pred_list.append(probe_logits.argmax(-1)[0].byte().cpu().numpy())
            frame_denoise_step.append(step_idx)
            frame_within_step.append(k)
            if granularity == "halt" and (model.thinker.predict_halt_value(z_H_c) <= halt_threshold).item():
                break
        steps_used_per_denoise_step.append(k_used)

        # unconditional branch: independent, only needed for the CFG-combined
        # image at the end of this step — not recorded frame-by-frame.
        last_logits_u = None
        if do_cfg:
            for k in range(max_iters):
                last_logits_u, z_H_u, z_L_u = _thinker_step(model, null_sample, z_H_u, z_L_u, H_cycles=h_override)
                if granularity == "halt" and (model.thinker.predict_halt_value(z_H_u) <= halt_threshold).item():
                    break

        noise_pred_c = _paint_logits(model, step_sample, last_logits_c)
        if do_cfg:
            noise_pred_u = _paint_logits(model, null_sample, last_logits_u)
            noise_pred = noise_pred_u + cfg_scale * (noise_pred_c - noise_pred_u)
        else:
            noise_pred = noise_pred_c

        cand = model.decode_for_eval(noise_pred)
        decoded_frames.append(_to_uint8_image(cand[0]))
        classifier_pred_list.append(_classify_cell_pred(cand, classifier, cell_size))

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return {
        "x_noisy_frames": x_noisy_frames,
        "decoded_frames": decoded_frames,
        "classifier_pred": np.stack(classifier_pred_list, axis=0),
        "probe_pred": np.stack(probe_pred_list, axis=0),
        "frame_denoise_step": frame_denoise_step,
        "frame_within_step": frame_within_step,
        "steps_used_per_denoise_step": steps_used_per_denoise_step,
        "n_sup": n_sup, "H_cycles": H_cycles_real,
    }


@torch.no_grad()
def _classify_cell_pred(images: torch.Tensor, classifier, cell_size: int) -> np.ndarray:
    """images: (1, 1, H, W) float [0,1] -> (81,) uint8 argmax digit (0-8)."""
    device = next(classifier.parameters()).device
    images = images.to(device)
    cells = images.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(81, 1, cell_size, cell_size)
    return classifier(cells).argmax(dim=-1).byte().cpu().numpy()


def _to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """img: (1, H, W) or (H, W) float, roughly [0,1] -> (H, W) uint8."""
    arr = img.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return (arr * 255.0).round().astype(np.uint8)


def _derive_ground_truth(final_pred: np.ndarray, solution: np.ndarray, given_mask) -> tuple:
    """This dataset's puzzles can have more than one valid completion, so
    exact-matching only the one recorded `solution` understates a model
    that legitimately converged on a different valid one (see the
    targeted_corruption_probe.py "adapt" finding). If the model's own final
    generated grid is a full valid sudoku AND agrees with the solution on
    every given cell (given cells are factual hints, never ambiguous), use
    IT as ground truth instead. Returns (ground_truth, was_derived)."""
    given = given_mask if given_mask is not None else np.zeros(81, dtype=bool)
    given_ok = bool(np.all(final_pred[given] == solution[given])) if given.any() else True
    if given_ok:
        valid = bool(_check_sudoku_constraints(torch.from_numpy(final_pred).long().unsqueeze(0))[0].item())
        if valid:
            return final_pred.copy(), True
    return solution, False


def _slice_sample(conditions, idx: int):
    kwargs = {}
    for f in dataclasses.fields(type(conditions)):
        val = getattr(conditions, f.name)
        kwargs[f.name] = val[idx:idx + 1] if isinstance(val, torch.Tensor) else val
    return type(conditions)(**kwargs)


# ── Puzzle selection ─────────────────────────────────────────────────────
#
# v4: dropped halting from the SELECTION scan entirely. v3 used
# forward_with_carry(use_halt_head=True) batched over the whole scan pool,
# which is cheap (most samples exit early) but wrong: halting is a hard
# per-step threshold decision (predict_halt_value <= threshold), and the
# batched scan (16-32 samples/call) computes slightly different floating-
# point values than the single-sample re-run the final export does for
# that same puzzle — enough, near the threshold, to flip a halt decision
# and send the two trajectories down completely different paths. Verified
# directly: the scan claimed a "hard" pick was wrong through step 11, but
# its own exported halt_carry trajectory actually resolved by step 3.
#
# The fix: scan with NO halting at all (plain forward_with_carry, carry
# never reset — i.e. what "reset_none" represents), so there is no discrete
# decision to be batch-size-sensitive about; batching only ever introduces
# tiny float noise that can't flip which digit a cell reads as. Halting
# configs (halt_0/halt_0.0002/halt_carry) stay in the export as things to
# WATCH, just not what puzzles get SELECTED on.
#
# Solvability: a puzzle counts as solved if its final real image is a
# valid, given-consistent sudoku — possibly a different completion than
# the dataset's one recorded solution (see _derive_ground_truth). Among
# solved puzzles, rank by how LATE the last wrong denoising step is,
# expressed as a fraction of the trajectory (last_wrong_step+1)/T — not
# raw wrong-step COUNT, which rewards puzzles that are wrong for a burst of
# early steps (where the model uses few reasoning iterations anyway) just
# as much as ones genuinely still wrong deep into the trajectory. Total
# wrong-step count is kept only as a tie-break.

@torch.no_grad()
def _run_batched_carry_scan(
    model, classifier, cell_size: int, conditions, x_init: torch.Tensor,
    num_inference_steps: int, cfg_scale: float,
) -> np.ndarray:
    """Batched real-image scan: plain forward_with_carry (no halting), carry
    never reset across denoising steps — the "reset_none" config's own
    behaviour, at coarse (default H_cycles) granularity for speed; the
    resulting x/image trajectory is identical to the fine H_cycles=1
    decomposition (same total compute, just organized into fewer Python-
    level calls — see probe_reasoning_dynamics.py's verified equivalence).
    Returns classifier_pred_per_step: (T, B, 81) uint8 — the classifier's
    read of the real decoded image at every step, for the WHOLE batch."""
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    B = x_init.shape[0]
    do_cfg = cfg_scale != 1.0
    z_H_c = z_L_c = None
    z_H_u = z_L_u = None

    per_step = []
    for t in model.scheduler.timesteps:
        t_batch = t.expand(B).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c)
        noise_pred = pred_c.pred
        if do_cfg:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        cand = model.decode_for_eval(noise_pred)
        cells = cand.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
        cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)
        preds = classifier(cells).argmax(dim=-1).reshape(B, 81).byte().cpu().numpy()
        per_step.append(preds)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return np.stack(per_step, axis=0)


def _score_real_trajectory(classifier_pred_per_step: np.ndarray, solutions: np.ndarray, given_mask) -> list:
    """Per-sample scoring from a (T, B, 81) real classifier-read trajectory.
    Returns a list of dicts (length B): solved, derived, last_wrong_step,
    num_wrong_steps, late_ratio = (last_wrong_step+1)/T."""
    T, B, _ = classifier_pred_per_step.shape
    final_pred = classifier_pred_per_step[-1]
    results = []
    for b in range(B):
        given_b = given_mask[b] if given_mask is not None else None
        gt, derived = _derive_ground_truth(final_pred[b], solutions[b], given_b)
        blank = ~given_b if given_b is not None else np.ones(81, dtype=bool)
        wrong_per_step = np.array([
            np.any(classifier_pred_per_step[t, b][blank] != gt[blank]) for t in range(T)
        ])
        last_wrong_step = int(np.max(np.where(wrong_per_step)[0])) if wrong_per_step.any() else -1
        results.append({
            "solved": not wrong_per_step[-1], "derived": derived,
            "last_wrong_step": last_wrong_step, "num_wrong_steps": int(wrong_per_step.sum()),
            "late_ratio": (last_wrong_step + 1) / T,
        })
    return results


@torch.no_grad()
def _rank_candidates(model, classifier, cell_size, cached: list, num_inference_steps, cfg_scale, n_pick) -> list:
    """cached: list of small batches (see _build_cached_batches) — scanned
    ALL of them, not just the first, so the scan pool size is
    len(cached)*batch_size."""
    scored_all = []
    total = 0
    for bi, cb in enumerate(cached):
        solutions = cb["solutions"].cpu().numpy()
        given_mask = cb["given_masks"].cpu().numpy().astype(bool) if cb["given_masks"] is not None else None
        B = solutions.shape[0]

        classifier_pred_per_step = _run_batched_carry_scan(
            model, classifier, cell_size, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale,
        )
        scores = _score_real_trajectory(classifier_pred_per_step, solutions, given_mask)
        for si, sc in enumerate(scores):
            scored_all.append({"batch_idx": bi, "idx": si, "global_idx": total + si, **sc})
        total += B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    solved = [s for s in scored_all if s["solved"]]
    logger.info(f"Rank: {len(solved)}/{total} scanned puzzles solved (valid final grid, possibly a different completion)")
    pool = solved if solved else scored_all  # graceful fallback — shouldn't normally trigger

    pool.sort(key=lambda s: (s["late_ratio"], s["num_wrong_steps"]), reverse=True)
    picks = pool[:n_pick]
    for p in picks:
        logger.info(
            f"  picked global_idx={p['global_idx']} (batch {p['batch_idx']}, sample {p['idx']})  "
            f"late_ratio={p['late_ratio']:.2f}  last_wrong_step={p['last_wrong_step']}  "
            f"num_wrong_steps={p['num_wrong_steps']}/{num_inference_steps}  derived_gt={p['derived']}"
        )
    return picks


# ── Encoding helpers ────────────────────────────────────────────────────────

def _sprite_sheet(frames: list, max_cols: int = 20) -> tuple:
    """Pack a list of (H, W) uint8 arrays into one grid PNG.
    Returns (base64_png_str, cols, rows, frame_h, frame_w)."""
    n = len(frames)
    h, w = frames[0].shape
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    buf = io.BytesIO()
    Image.fromarray(sheet, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), cols, rows, h, w


def _png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bytes_b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _resize_frames(frames: list, size: int) -> list:
    if frames[0].shape[0] == size:
        return frames
    return [np.array(Image.fromarray(f, mode="L").resize((size, size), Image.BILINEAR)) for f in frames]


XNOISY_DISPLAY_SIZE = 64  # random noise compresses far worse than the real
# candidate images (it's incompressible by nature) despite carrying much
# less perceptible detail — this panel is supporting/secondary, so it's
# stored much smaller than the candidate/puzzle images regardless of
# display_size (this alone was ~45% of a trajectory's exported size).


def _package_trajectory(traj: dict, display_size: int, solution: np.ndarray, given_mask) -> dict:
    decoded = _resize_frames(traj["decoded_frames"], display_size)
    xnoisy = _resize_frames(traj["x_noisy_frames"], min(display_size, XNOISY_DISPLAY_SIZE))
    img_b64, img_cols, img_rows, fh, fw = _sprite_sheet(decoded)
    xn_b64, xn_cols, xn_rows, xn_fh, _ = _sprite_sheet(xnoisy)

    # Ground truth is derived PER CONFIG, from THIS config's own final
    # answer — "reset every step", "never reset", and the two halt
    # thresholds are four independently-sampled trajectories and routinely
    # converge to four different (but each potentially valid) completions;
    # judging all of them against one fixed reference (e.g. reset_1's own
    # answer) would count a config's own genuinely-valid different solution
    # as wrong. See _derive_ground_truth.
    ground_truth, derived = _derive_ground_truth(traj["classifier_pred"][-1], solution, given_mask)

    return {
        "n_sup": traj["n_sup"], "H_cycles": traj["H_cycles"],
        "frame_denoise_step": traj["frame_denoise_step"],
        "frame_within_step": traj["frame_within_step"],
        "steps_used_per_denoise_step": traj["steps_used_per_denoise_step"],
        "probe_pred_b64": _bytes_b64(traj["probe_pred"]),
        "classifier_pred_b64": _bytes_b64(traj["classifier_pred"]),
        "images_sprite_png_b64": img_b64, "images_cols": img_cols, "images_rows": img_rows,
        "frame_size": fh,
        "xnoisy_sprite_png_b64": xn_b64, "xnoisy_cols": xn_cols, "xnoisy_rows": xn_rows,
        "xnoisy_frame_size": xn_fh,
        "n_fine_frames": len(traj["frame_denoise_step"]),
        "solution": ground_truth.tolist(), "solution_derived_from_model": derived,
    }


def _log_size(key: str, packaged: dict):
    approx_kb = (len(packaged["images_sprite_png_b64"]) + len(packaged["probe_pred_b64"]) +
                 len(packaged["classifier_pred_b64"]) + len(packaged["xnoisy_sprite_png_b64"])) / 1024
    logger.info(f"    {key}: {packaged['n_fine_frames']} fine frames, ~{approx_kb:.0f} KB")


CONFIGS = [
    {"key": "reset_1", "granularity": "fine", "reset_every": 1},
    {"key": "reset_none", "granularity": "fine", "reset_every": None},
    {"key": "halt_0", "granularity": "halt", "reset_every": 1, "halt_threshold": 0.0},
    {"key": "halt_0.0002", "granularity": "halt", "reset_every": 1, "halt_threshold": 0.0002},
    {"key": "halt_carry", "granularity": "halt", "reset_every": None, "halt_threshold": 0.0002},
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    probe_checkpoint = cfg.get("probe_checkpoint", None)
    if checkpoint is None or probe_checkpoint is None:
        raise SystemExit(
            "ERROR: need both checkpoint= and +probe_checkpoint=.\n"
            "  Usage: python experiments/export_sampling_animation.py experiment=<name> "
            "checkpoint=<main.pt> +probe_checkpoint=<digit_probe.pt> [+viz.xxx=...]"
        )

    vz = cfg.get("viz", {})
    tag: str = vz.get("tag", "model")
    num_scan: int = vz.get("num_scan", 1024)
    num_puzzles: int = vz.get("num_puzzles", 3)
    seed: int = vz.get("seed", 0)
    display_size = vz.get("display_size", None)
    configs_to_run: list = list(vz.get("configs", [c["key"] for c in CONFIGS]))
    rank_halt_threshold: float = vz.get("rank_halt_threshold", 0.0002)
    out_path: str = vz.get("out", str(Path(probe_checkpoint).parent / f"sampling_animation_{tag}.json"))

    torch.set_float32_matmul_precision("high")
    # Puzzle selection depends on absolute per-puzzle accuracy, not just
    # relative ranking — cuDNN's default nondeterministic kernel selection
    # was observed to swing an identical-seed scan from "several puzzles
    # above 85% accuracy" to "nothing above 35%" between back-to-back runs
    # (see git history), which made picks unreproducible. Pin it down.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

    reasoner = model.thinker
    puzzle_emb_len = reasoner.inner.puzzle_emb_len
    if not getattr(reasoner, "with_halt_head", False):
        raise SystemExit("Puzzle selection needs a halt head — add thinker.with_halt_head=true.")

    probe_ckpt = torch.load(str(probe_checkpoint), map_location=device, weights_only=True)
    probe = DigitProbe(probe_ckpt["hidden_size"]).to(device)
    probe.load_state_dict(probe_ckpt["probe_state"])
    probe.eval()
    assert probe_ckpt["puzzle_emb_len"] == puzzle_emb_len, "probe/model puzzle_emb_len mismatch"

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    pipeline = model.sampling_pipeline
    cfg_scale: float = vz.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = vz.get("num_inference_steps", pipeline.num_inference_steps)
    logger.info(f"num_inference_steps={num_inference_steps}  cfg_scale={cfg_scale}  n_sup={model.n_sup}")

    cached = _build_cached_batches(model, eval_dl, device, num_scan, seed)

    picks = _rank_candidates(
        model, classifier, cell_size, cached, num_inference_steps, cfg_scale, rank_halt_threshold, num_puzzles,
    )

    out = {
        "checkpoint": str(checkpoint), "probe_checkpoint": str(probe_checkpoint), "tag": tag,
        "num_inference_steps": num_inference_steps, "cfg_scale": cfg_scale, "cell_size": cell_size,
        "puzzle_order": [str(p["global_idx"]) for p in picks],
        "puzzles": {},
    }

    for pick in picks:
        key, idx, cb = pick["global_idx"], pick["idx"], cached[pick["batch_idx"]]
        logger.info(
            f"Puzzle key={key} (batch {pick['batch_idx']}, sample {idx})  "
            f"num_wrong_steps={pick['num_wrong_steps']}/{num_inference_steps}  last_wrong_step={pick['last_wrong_step']}"
        )
        sample = _slice_sample(cb["conditions"], idx)
        x_init = cb["x_init"][idx:idx + 1]
        solution = cb["solutions"][idx].cpu().numpy().tolist()
        given_mask = cb["given_masks"][idx].cpu().numpy().tolist() if cb["given_masks"] is not None else None

        puzzle_img = _to_uint8_image(sample.spatial_conditions[0])
        if display_size:
            puzzle_img = _resize_frames([puzzle_img], display_size)[0]

        puzzle_out = {
            "puzzle_idx": key, "solution": solution, "given_mask": given_mask,
            "puzzle_img_png_b64": _png_b64(puzzle_img), "configs": {},
        }
        given_arr = np.array(given_mask, dtype=bool) if given_mask is not None else None
        sol_arr = np.array(solution)

        # All configs are (re-)sampled fresh here — the ranking scan above
        # is deliberately a different, cheaper trajectory (native batched
        # halting) that doesn't produce fine-grained per-frame data, so
        # there is nothing to reuse from it.
        for c in CONFIGS:
            if c["key"] not in configs_to_run:
                continue
            traj = _run_visual_trajectory(
                model, probe, classifier, cell_size, puzzle_emb_len, sample, x_init,
                num_inference_steps, cfg_scale, c["granularity"],
                reset_fn=_make_reset_fn(c["reset_every"]), halt_threshold=c.get("halt_threshold", 0.0),
            )
            # Ground truth is derived per config (see _package_trajectory) —
            # the five configs routinely converge to different completions.
            packaged = _package_trajectory(traj, display_size or puzzle_img.shape[0], sol_arr, given_arr)
            puzzle_out["configs"][c["key"]] = packaged
            _log_size(c["key"], packaged)
            logger.info(
                f"    {c['key']}: ground truth "
                f"{'DERIVED (valid, different completion)' if packaged['solution_derived_from_model'] else 'kept as dataset solution'}"
            )
            del traj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out["puzzles"][str(key)] = puzzle_out

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"Saved → {out_path}  ({size_mb:.2f} MB)")

    return out_path


if __name__ == "__main__":
    main()
