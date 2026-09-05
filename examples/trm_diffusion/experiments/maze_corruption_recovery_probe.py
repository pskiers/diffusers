"""experiments/maze_corruption_recovery_probe.py — wrong-path recovery probe (AMAZE, 13x13).

Question: given a *partially drawn and deliberately wrong* solution as the
starting point of the reverse process, does the model repair it?

Each board is rendered at a context level p (the first p% of the GT path drawn
in blue on the unsolved maze), corrupted, re-noised to t_start, and denoised
back to 0 with the model's own sampler. Three corruption modes:

  ADD   at p in {0, 20, 50}%      — a wrong path walked out of the prefix's
                                    frontier: it follows the GT path only while
                                    that is the only opening, then takes the
                                    first side opening and runs to a dead end
                                    without ever re-entering the GT path.
  WALL  at p in {10, 30, 50, 75}% — a straight shortcut drawn from the prefix's
                                    frontier to the target, straight through walls.
  GAP   x in {10, 30, 50, 75}%    — a contiguous x% of the *full* GT path erased
                                    (so its context level is always p = 100%).

Every combination runs at t_start in {10, 30, 50, 70, 90} (out of
num_train_timesteps = 100), i.e. from "barely re-noised, the corruption is
still plainly visible" to "most of the signal destroyed".

Each cell is reported against two references measured on the SAME boards, the
SAME t_start and the SAME sampler noise:
  floor    — the corrupted image scored as-is, no denoising ("do nothing").
  ceiling  — the uncorrupted context denoised from the same t_start ("the best
             this model does from a clean start").
so `denoised` between the two is the recovery signal, and neither model's
absolute skill nor the amount of noise injected can be mistaken for it.

Usage: see slurm_scripts/sample_amaze/maze_corruption_recovery.sh.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from PIL import Image

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import AmazeMetrics
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")

LEVELS = {
    "add": [0.0, 0.20, 0.50],
    "gap": [0.10, 0.30, 0.50, 0.75],
    "wall": [0.10, 0.30, 0.50, 0.75]
}
# Dla GAP kontekstem bazowym jest 1.0 (całe rozwiązanie), dla reszty - odpowiadający im procent.
CONTEXT_LEVELS = sorted(set(LEVELS["add"] + LEVELS["wall"] + [1.0]))

# WALL: a board counts as "still routed through a wall" when this fraction of
# the shortcut's wall-crossing pixels is still blue in the output. A bare > 0
# test would be dominated by resampling blur — the model works at 144x144 and
# the mask is read at the cell_map's ~1342x1342, so a cleanly erased shortcut
# still leaves a few bleed pixels on a 10px wall.
STILL_WALL_FRAC = 0.25


@torch.no_grad()
def _denoise_from_full(model, conditions, x_t: torch.Tensor, run_timesteps: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Run the reverse process from x_t down to t=0 over `run_timesteps`.

    `generator` is forwarded to scheduler.step so that the clean-context
    (ceiling) run and the corrupted run of the same board share an identical
    noise sequence. The default scheduler here is DDPMScheduler, whose step()
    is *ancestral* — it draws fresh Gaussian noise at every t > 0. Without a
    shared generator, "the corrupted run ended up somewhere else than the clean
    run" would partly just be sampler noise, which would make
    matches_clean_run_rate meaningless.
    """
    T = model.scheduler.config.num_train_timesteps
    predictor = model.sampling_pipeline.predictor
    x = x_t
    for t in run_timesteps:
        t_batch = t.expand(x.shape[0]).to(x.device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        pred = predictor.predict(model, step_sample, int(t.item()), T)
        kw = {"generator": generator} if generator is not None else {}
        x = model.scheduler.step(pred, t, x, **kw).prev_sample
    return x


def _to_native(img_t: torch.Tensor, ms: mcl.MazeSample) -> np.ndarray:
    """Model output (C, H, W) in [0, 1] -> uint8 RGB at the cell_map's own
    resolution, which is where cell ids can be read off without interpolating
    them."""
    a = (img_t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(a, (ms.ids.shape[1], ms.ids.shape[0]), interpolation=cv2.INTER_LINEAR)


def _new_metric_acc() -> dict:
    return {"n": 0, "coverage": 0.0, "violation": 0.0, "pass": 0.0}


def _add_metric(acc: dict, rec: dict) -> None:
    acc["n"] += 1
    acc["coverage"] += float(rec["gt_cell_coverage"])
    acc["violation"] += float(rec["background_violation"])
    acc["pass"] += float(rec["pass"])


def _mean_metric(acc: dict) -> dict:
    n = max(1, acc["n"])
    return {"n": acc["n"], "coverage": acc["coverage"] / n,
            "violation": acc["violation"] / n, "pass": acc["pass"] / n}


def _new_rec_acc() -> dict:
    return dict(n_units=0, rec_units=0, adapt_units=0, n_seen=0, n_applied=0, full_rec=0,
                rec_and_valid=0, still_wall=0, collat_cand=0, collat_broken=0,
                matches_clean=0, wall_line_px=0, wall_line_cleared=0,
                remain_cand=0, remain_hit=0, size_units=0, branch_off=0, branch_n=0)


@torch.no_grad()
def _run_one_tstart(model, ds, scorer, device, t_start, run_timesteps, n_total, batch_size,
                    seed, dump_dir, dump_n, model_name) -> tuple:
    # Ceiling over every board (the clean-context reference for this t_start).
    ceil_acc = {p: _new_metric_acc() for p in CONTEXT_LEVELS}
    # Everything below is restricted to boards where the corruption actually
    # applied — see `applied` in the loop. ADD is a silent no-op on a board
    # whose walk reaches the goal without ever passing an opening, and
    # averaging an uncorrupted board into "denoised" would make the model look
    # better the *less* it was asked to do.
    ceil_m_acc = {(p, c): _new_metric_acc() for c in LEVELS for p in LEVELS[c]}
    denoise_acc = {(p, c): _new_metric_acc() for c in LEVELS for p in LEVELS[c]}
    floor_acc = {(p, c): _new_metric_acc() for c in LEVELS for p in LEVELS[c]}
    rec = {(p, c): _new_rec_acc() for c in LEVELS for p in LEVELS[c]}

    dumped_counts = defaultdict(int)
    processed = 0

    for start in range(0, n_total, batch_size):
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, mcl.build_maze_sample(s.metadata)) for s in raw]
        pairs = [(s, ms) for s, ms in pairs if ms is not None]
        if not pairs: continue
        samples, mss = [s for s, _ in pairs], [ms for _, ms in pairs]
        meta_list = [s.metadata for s in samples]
        B = len(samples)
        conditions = model._batch_to_sample(AmazeDataset.collate_fn(samples), device)
        processed += B

        # Budowanie referencyjnych (czystych) denoise-owań per kontekst
        clean_ctx = {p: [mcl.render_partial(ms, p) for ms in mss] for p in CONTEXT_LEVELS}
        ceil_pred, ceil_recs = {}, {}
        for p in CONTEXT_LEVELS:
            clean_t = torch.stack([ds.transform(Image.fromarray(img)) for img, _ in clean_ctx[p]]).to(device)
            base_seed = seed + start * 977 + int(p * 1000)
            g = torch.Generator(device=device).manual_seed(base_seed)
            noise = torch.randn(clean_t.shape, generator=g, device=device)
            tb = torch.full((B,), t_start, device=device, dtype=torch.long)

            x_clean = model.scheduler.add_noise(clean_t, noise, tb)
            step_g = torch.Generator(device=device).manual_seed(base_seed + 1)
            ceil_out = model.decode_for_eval(
                _denoise_from_full(model, conditions, x_clean, run_timesteps, step_g)).clamp(0, 1)
            ceil_recs[p] = scorer.compute_and_accumulate_metrics(ceil_out, meta_list)
            for r in ceil_recs[p]:
                _add_metric(ceil_acc[p], r)
            ceil_native = [_to_native(ceil_out[i], mss[i]) for i in range(B)]
            ceil_pred[p] = [mcl.predicted_cells(ceil_native[i], mss[i]) for i in range(B)]

            for ctype in LEVELS:
                # GAP zawsze operuje na bazie p=1.0.
                # ADD i WALL operują na bazie p (procent ścieżki pokazany)
                if (ctype == "gap" and p != 1.0) or (ctype != "gap" and p not in LEVELS[ctype]):
                    continue

                # Dla GAP, p_loop iteruje przez zdefiniowane luki wycinane z bazy p=1.0.
                # Dla ADD/WALL p_loop to po prostu p.
                p_loop = LEVELS["gap"] if ctype == "gap" else [p]

                for active_p in p_loop:
                    rng = random.Random(seed + start * 131 + int(active_p * 1000) + sum(ord(c) for c in ctype))
                    corr_imgs, records = [], []

                    for i, ms in enumerate(mss):
                        base, shown = clean_ctx[p][i]
                        remaining = set(ms.path[len(shown):])
                        if ctype == "gap":
                            img, erased = mcl.apply_gap(base, ms, shown, rng, gap_frac=active_p)
                            records.append({"kind": "gap", "cells": set(erased), "shown": set(shown) - set(erased),
                                            "remaining": set(), "applied": bool(erased), "size": len(erased)})
                        elif ctype == "add":
                            img, added, gt_walked, diverge = mcl.apply_add(base, ms, shown, rng)
                            # gt_walked is drawn but correct: it joins the cells
                            # that must survive, and leaves the "still undrawn"
                            # tail that remaining_gt_recall scores.
                            records.append({"kind": "add", "cells": set(added),
                                            "shown": set(shown) | set(gt_walked),
                                            "remaining": remaining - set(gt_walked),
                                            "applied": bool(added), "size": len(added),
                                            "branch_off": diverge})
                        else: # WALL (shortcut)
                            img, line_mask, in_wall, off_path = mcl.apply_shortcut(base, ms, shown, rng)
                            npx = 0 if off_path is None else int(off_path.sum())
                            records.append({"kind": "wall", "line": off_path, "in_wall": in_wall,
                                            "shown": set(shown), "remaining": remaining,
                                            "applied": npx > 0, "size": npx})
                        corr_imgs.append(img)

                    corr_t = torch.stack([ds.transform(Image.fromarray(img)) for img in corr_imgs]).to(device)
                    floor_recs = scorer.compute_and_accumulate_metrics(corr_t.clamp(0, 1), meta_list)

                    x_corr = model.scheduler.add_noise(corr_t, noise, tb)
                    step_g = torch.Generator(device=device).manual_seed(base_seed + 1)
                    corr_out = model.decode_for_eval(
                        _denoise_from_full(model, conditions, x_corr, run_timesteps, step_g)).clamp(0, 1)
                    corr_recs = scorer.compute_and_accumulate_metrics(corr_out, meta_list)

                    a = rec[(active_p, ctype)]
                    for i in range(B):
                        r0 = records[i]
                        a["n_seen"] += 1
                        if not r0["applied"]:
                            continue
                        a["n_applied"] += 1
                        a["size_units"] += r0["size"]
                        if r0["kind"] == "add" and r0["branch_off"] >= 0:
                            a["branch_off"] += r0["branch_off"]; a["branch_n"] += 1

                        _add_metric(ceil_m_acc[(active_p, ctype)], ceil_recs[p][i])
                        _add_metric(floor_acc[(active_p, ctype)], floor_recs[i])
                        _add_metric(denoise_acc[(active_p, ctype)], corr_recs[i])

                        out_native = _to_native(corr_out[i], mss[i])
                        pred = mcl.predicted_cells(out_native, mss[i])

                        if r0["kind"] == "gap":
                            units = r0["cells"]
                            rec_u = sum(1 for c in units if c in pred)
                            a["n_units"] += len(units); a["rec_units"] += rec_u
                            board_full = (rec_u == len(units))
                        elif r0["kind"] == "add":
                            units = r0["cells"]
                            rem = sum(1 for c in units if c not in pred)
                            a["n_units"] += len(units); a["rec_units"] += rem
                            a["adapt_units"] += (len(units) - rem)
                            board_full = (rem == len(units))
                        else:  # Shortcut
                            line = r0["line"]
                            bm = mcl.blue_mask(out_native, morph_open=True)
                            cleared = int((line & ~bm).sum())
                            a["wall_line_px"] += int(line.sum()); a["wall_line_cleared"] += cleared
                            n_wall = 0 if r0["in_wall"] is None else int(r0["in_wall"].sum())
                            still = n_wall > 0 and int((r0["in_wall"] & bm).sum()) > STILL_WALL_FRAC * n_wall
                            a["still_wall"] += int(still)
                            a["rec_units"] += cleared; a["n_units"] += int(line.sum())
                            board_full = not still

                        a["full_rec"] += int(board_full)
                        a["rec_and_valid"] += int(board_full and float(corr_recs[i]["pass"]) >= 1.0)

                        shown_ok = r0["shown"]
                        a["collat_cand"] += len(shown_ok)
                        a["collat_broken"] += sum(1 for c in shown_ok if c not in pred)
                        a["remain_cand"] += len(r0["remaining"])
                        a["remain_hit"] += sum(1 for c in r0["remaining"] if c in pred)
                        a["matches_clean"] += int(pred == ceil_pred[p][i])

                        # Zapis po dump_n przykładów na kombinację (type, level, t_start)
                        k = (ctype, active_p)
                        if dump_dir and dumped_counts[k] < dump_n:
                            os.makedirs(dump_dir, exist_ok=True)
                            stem = f"{dump_dir}/{model_name}_t{t_start}_{ctype}_p{int(active_p*100)}_{dumped_counts[k]}"
                            Image.fromarray(clean_ctx[p][i][0]).save(f"{stem}_clean.png")
                            Image.fromarray(corr_imgs[i]).save(f"{stem}_corrupt.png")
                            Image.fromarray(out_native).save(f"{stem}_out.png")
                            Image.fromarray(ceil_native[i]).save(f"{stem}_ceilout.png")
                            Image.fromarray(mss[i].sol).save(f"{stem}_gt.png")
                            dumped_counts[k] += 1

    results: dict = {}
    for p in CONTEXT_LEVELS:
        results[f"level={int(p*100)}/ceiling"] = _mean_metric(ceil_acc[p])
    for (p, ctype), a in rec.items():
        nu, na, cc = max(1, a["n_units"]), max(1, a["n_applied"]), max(1, a["collat_cand"])
        entry = {
            "ceiling": _mean_metric(ceil_m_acc[(p, ctype)]),
            "floor": _mean_metric(floor_acc[(p, ctype)]),
            "denoised": _mean_metric(denoise_acc[(p, ctype)]),
            "n_boards": a["n_seen"],
            "n_applied": a["n_applied"],
            "applied_rate": a["n_applied"] / max(1, a["n_seen"]),
            "mean_corruption_size": a["size_units"] / na,
            "recovery_rate": a["rec_units"] / nu,
            "full_recovery_rate": a["full_rec"] / na,
            "recovered_and_valid_rate": a["rec_and_valid"] / na,
            "collateral_break_rate": a["collat_broken"] / cc,
            "matches_clean_run_rate": a["matches_clean"] / na,
        }
        if a["remain_cand"]:
            entry["remaining_gt_recall"] = a["remain_hit"] / a["remain_cand"]
        if ctype == "add":
            entry["adapt_rate"] = a["adapt_units"] / nu
            # How many GT cells the walk had to follow before an opening
            # existed — i.e. how far past the nominal level the drawn (correct)
            # context actually reaches.
            entry["mean_gt_walk"] = a["branch_off"] / max(1, a["branch_n"])
        if ctype == "wall":
            entry["still_through_wall_rate"] = a["still_wall"] / na
            entry["wall_pixel_clear_rate"] = a["wall_line_cleared"] / max(1, a["wall_line_px"])

        results[f"level={int(p*100)}/{ctype}"] = entry
        logger.info(f"[{model_name}] t={t_start} level={int(p*100)} {ctype}: "
                    f"applied={a['n_applied']}/{a['n_seen']} recovery={entry['recovery_rate']:.3f} "
                    f"full={entry['full_recovery_rate']:.3f} "
                    f"pass floor/denoised/ceil={entry['floor']['pass']:.3f}/"
                    f"{entry['denoised']['pass']:.3f}/{entry['ceiling']['pass']:.3f}")
    return results, processed

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pb = cfg.get("probe", {})
    num_samples = int(pb.get("num_samples", 256))
    batch_size = int(pb.get("batch_size", 32))
    seed = int(pb.get("seed", 0))
    t_starts = [int(t) for t in pb.get("t_starts", [10, 30, 50, 70, 90])]
    model_name = str(pb.get("model_name", "model"))
    checkpoint = cfg.get("checkpoint", None)

    default_data = "data/amaze/test_maze/square/n13_square_test.parquet"
    data_parquet = str(pb.get("data_parquet", default_data))
    out_path = str(pb.get("out", f"runs/maze_corruption/{model_name}.json"))

    dump_dir = pb.get("dump_dir", None)
    if dump_dir is None:
        dump_dir = str(Path(out_path).parent / f"dump_{model_name}")
    dump_n = int(pb.get("dump_n", 5))

    if checkpoint is None:
        raise SystemExit("Set +checkpoint=<...>.pt")

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    num_inference_steps = int(pb.get("num_inference_steps", model.sampling_pipeline.num_inference_steps))
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    full_ts = model.scheduler.timesteps
    valid_t = {int(t.item()) for t in full_ts}
    bad = [t for t in t_starts if t not in valid_t]
    if bad:
        raise SystemExit(f"t_starts {bad} not in schedule {sorted(valid_t, reverse=True)}")

    if not os.path.isabs(data_parquet):
        data_parquet = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_parquet)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(
        dataset_path=data_parquet, split="test",
        image_size=int(cfg.data.get("image_size", 144)),
        condition_field=vd.get("condition_field", "m_original_img"),
        target_field=vd.get("target_field", "sol_img"),
        num_channels=int(vd.get("num_channels", 3)), include_metadata=True,
    )
    n_total = min(num_samples, len(ds))
    logger.info(f"[{model_name}] dataset={data_parquet} using {n_total}/{len(ds)} puzzles")

    scorer = AmazeMetrics(device=device, task="maze")
    all_results: dict = {}
    processed = 0

    for t_start in t_starts:
        run_timesteps = full_ts[full_ts <= t_start]
        logger.info(f"[{model_name}] --- t_start={t_start} ---")
        sub, processed = _run_one_tstart(model, ds, scorer, device, t_start, run_timesteps, n_total,
                                         batch_size, seed, dump_dir, dump_n, model_name)
        for k, v in sub.items():
            all_results[f"t={t_start}/{k}"] = v

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"model": model_name, "checkpoint": str(checkpoint), "t_starts": t_starts,
                       "num_inference_steps": num_inference_steps, "n_puzzles": processed,
                       "data": data_parquet, "levels": LEVELS, "seed": seed,
                       "dump_dir": str(dump_dir), "results": all_results}, f, indent=2)
        logger.info(f"[{model_name}] results -> {out_path}")
    return all_results


if __name__ == "__main__":
    main()
