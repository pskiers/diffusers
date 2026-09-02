"""experiments/maze_corruption_recovery_probe.py — maze analog of
corruption_recovery_grid_probe.py, for AMAZE square mazes, comparing TRM vs DiT.

Scenario A (noise-from-start): start from the real solution with only the first
p% of the blue path drawn (context), inject one of three deliberate mistakes into
that context, noise it to ``t_start`` and denoise to the end with the model's OWN
sampler (CFG+thinker for TRM, direct for DiT), then measure whether the model
fixes the mistake.

Corruption types (see maze_corruption_lib.py):
  GAP  — erase a contiguous interior chunk of the shown path (coverage down).
  ADD  — add a legal off-path dead-end spur near the frontier (violation up).
  WALL — draw a blue line through the nearest wall + a few px past it.

Levels (p = fraction of the true path shown as context):
  GAP  ∈ {45, 80}%   (needs a two-sided bracket)
  ADD/WALL ∈ {10, 45, 80}%

Three references per condition: FLOOR (score the corrupted CLEAN image, no
denoise), CONTROLLED CEILING (denoise the clean context, no corruption — shared
per level), CORRUPTED-DENOISED (the run). Whole-solution quality via
AmazeMetrics(task=maze); per-corruption recovery via cell/pixel bookkeeping.

Usage (one model at a time; the slurm launcher runs both):
    # TRM
    python experiments/maze_corruption_recovery_probe.py \
      experiment=amaze_thinker_v2_controlnet \
      painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
      +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +probe.model_name=trm \
      +probe.data_parquet=data/amaze/test_maze/square/all_square_test.parquet \
      +probe.num_samples=128 +probe.out=runs/maze_corruption/trm.json
    # DiT
    python experiments/maze_corruption_recovery_probe.py \
      experiment=amaze_dit_maze \
      +checkpoint=runs/dit_maze_final/checkpoint_final.pt \
      +probe.model_name=dit \
      +probe.data_parquet=data/amaze/test_maze/square/all_square_test.parquet \
      +probe.num_samples=128 +probe.out=runs/maze_corruption/dit.json

    # +probe.smoke=true       -> num_samples=8, dumps PNGs, quick sanity
    # +probe.dump_dir=<path>  -> save clean/corrupt/denoised PNGs for the first few
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import AmazeMetrics
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")

LEVELS = {"gap": [0.45, 0.80], "add": [0.10, 0.45, 0.80], "wall": [0.10, 0.45, 0.80]}
ALL_LEVELS = sorted({p for ps in LEVELS.values() for p in ps})


@torch.no_grad()
def _denoise_from_full(model, conditions, x_t: torch.Tensor, run_timesteps: torch.Tensor) -> torch.Tensor:
    """Full-model denoise from x_t down run_timesteps using the model's OWN
    predictor (CFG+thinker for TRM, direct for DiT). Mirrors
    SamplingPipeline.sample_one_batch's loop but starts from x_t."""
    T = model.scheduler.config.num_train_timesteps
    predictor = model.sampling_pipeline.predictor
    x = x_t
    for t in run_timesteps:
        t_batch = t.expand(x.shape[0]).to(x.device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        pred = predictor.predict(model, step_sample, int(t.item()), T)
        x = model.scheduler.step(pred, t, x).prev_sample
    return x


def _to_728(img_t: torch.Tensor, ms: mcl.MazeSample) -> np.ndarray:
    """(3,144,144) float [0,1] -> (H,W,3) uint8 at native cell-map resolution."""
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


@torch.no_grad()
def _run_one_tstart(model, ds, scorer, device, t_start, run_timesteps, n_total, batch_size,
                    seed, gap_frac, dump_dir, dump_n, model_name) -> tuple:
    """Run the full corruption grid at a single t_start; return (results, processed)."""
    ceil_acc = {p: _new_metric_acc() for p in ALL_LEVELS}
    denoise_acc = {(p, c): _new_metric_acc() for c in LEVELS for p in LEVELS[c]}
    floor_acc = {(p, c): _new_metric_acc() for c in LEVELS for p in LEVELS[c]}
    rec = {(p, c): dict(n_units=0, rec_units=0, adapt_units=0, n_boards=0, full_rec=0,
                        rec_and_valid=0, still_wall=0, collat_cand=0, collat_broken=0,
                        matches_clean=0, wall_line_px=0, wall_line_cleared=0)
           for c in LEVELS for p in LEVELS[c]}
    dumped = 0
    processed = 0
    for start in range(0, n_total, batch_size):
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, mcl.build_maze_sample(s.metadata)) for s in raw]
        pairs = [(s, ms) for s, ms in pairs if ms is not None]
        if not pairs:
            continue
        samples = [s for s, _ in pairs]
        mss = [ms for _, ms in pairs]
        meta_list = [s.metadata for s in samples]
        B = len(samples)
        conditions = model._batch_to_sample(AmazeDataset.collate_fn(samples), device)
        processed += B

        for p in ALL_LEVELS:
            rng = random.Random(seed + start * 131 + int(p * 1000))
            # clean context (ceiling) — shared across corruption types at this level
            clean_ctx728 = [mcl.render_partial(ms, p) for ms in mss]   # (img, shown)
            clean_t = torch.stack([ds.transform(Image.fromarray(img)) for img, _ in clean_ctx728]).to(device)
            g = torch.Generator(device=device).manual_seed(seed + start * 977 + int(p * 1000))
            noise = torch.randn(clean_t.shape, generator=g, device=device)
            tb = torch.full((B,), t_start, device=device, dtype=torch.long)
            x_clean = model.scheduler.add_noise(clean_t, noise, tb)
            ceil_out = model.decode_for_eval(_denoise_from_full(model, conditions, x_clean, run_timesteps)).clamp(0, 1)
            for r in scorer.compute_and_accumulate_metrics(ceil_out, meta_list):
                _add_metric(ceil_acc[p], r)
            ceil_pred = [mcl.predicted_cells(_to_728(ceil_out[i], mss[i]), mss[i]) for i in range(B)]

            for ctype in LEVELS:
                if p not in LEVELS[ctype]:
                    continue
                # build corrupted context per sample + record corruption units
                corr_imgs, records = [], []
                for i, ms in enumerate(mss):
                    base, shown = clean_ctx728[i]
                    if ctype == "gap":
                        img, erased = mcl.apply_gap(base, ms, shown, rng, gap_frac=gap_frac)
                        records.append({"kind": "gap", "cells": set(erased), "shown": set(shown) - set(erased)})
                    elif ctype == "add":
                        img, added = mcl.apply_add(base, ms, shown, rng)
                        records.append({"kind": "add", "cells": set(added), "shown": set(shown)})
                    else:
                        img, line_mask, in_wall = mcl.apply_wall(base, ms, shown, rng)
                        records.append({"kind": "wall", "line": line_mask, "in_wall": in_wall, "shown": set(shown)})
                    corr_imgs.append(img)
                corr_t = torch.stack([ds.transform(Image.fromarray(img)) for img in corr_imgs]).to(device)

                # FLOOR: score corrupted CLEAN image (no denoise)
                for r in scorer.compute_and_accumulate_metrics(corr_t.clamp(0, 1), meta_list):
                    _add_metric(floor_acc[(p, ctype)], r)

                # CORRUPTED-DENOISED (same noise as ceiling)
                x_corr = model.scheduler.add_noise(corr_t, noise, tb)
                corr_out = model.decode_for_eval(_denoise_from_full(model, conditions, x_corr, run_timesteps)).clamp(0, 1)
                corr_recs = scorer.compute_and_accumulate_metrics(corr_out, meta_list)

                a = rec[(p, ctype)]
                for i in range(B):
                    _add_metric(denoise_acc[(p, ctype)], corr_recs[i])
                    out728 = _to_728(corr_out[i], mss[i])
                    pred = mcl.predicted_cells(out728, mss[i])
                    r0 = records[i]
                    board_full = None
                    if r0["kind"] == "gap":
                        units = r0["cells"]
                        if not units:
                            continue
                        rec_u = sum(1 for c in units if c in pred)      # restored
                        a["n_units"] += len(units); a["rec_units"] += rec_u
                        board_full = (rec_u == len(units))
                    elif r0["kind"] == "add":
                        units = r0["cells"]
                        if not units:
                            continue
                        rem = sum(1 for c in units if c not in pred)    # removed = recovered
                        a["n_units"] += len(units); a["rec_units"] += rem
                        a["adapt_units"] += (len(units) - rem)          # kept the wrong branch
                        board_full = (rem == len(units))
                    else:  # wall — pixel-based
                        line = r0["line"]
                        if line is None or int(line.sum()) == 0:
                            continue
                        bm = mcl.blue_mask(out728)
                        cleared = int((line & ~bm).sum())
                        a["wall_line_px"] += int(line.sum()); a["wall_line_cleared"] += cleared
                        still = r0["in_wall"] is not None and int((r0["in_wall"] & bm).sum()) > 0
                        a["still_wall"] += int(still)
                        a["rec_units"] += cleared; a["n_units"] += int(line.sum())
                        board_full = not still

                    a["n_boards"] += 1
                    a["full_rec"] += int(bool(board_full))
                    a["rec_and_valid"] += int(bool(board_full) and float(corr_recs[i]["pass"]) >= 1.0)
                    # collateral: previously-correct shown (non-corrupted) cells now missing
                    shown_ok = r0["shown"]
                    a["collat_cand"] += len(shown_ok)
                    a["collat_broken"] += sum(1 for c in shown_ok if c not in pred)
                    # corruption left no trace?
                    a["matches_clean"] += int(pred == ceil_pred[i])

                # dump a few PNGs for visual verification
                if dump_dir and dumped < dump_n and p == 0.45:
                    os.makedirs(dump_dir, exist_ok=True)
                    Image.fromarray(clean_ctx728[0][0]).save(f"{dump_dir}/{model_name}_t{t_start}_{ctype}_clean.png")
                    Image.fromarray(corr_imgs[0]).save(f"{dump_dir}/{model_name}_t{t_start}_{ctype}_corrupt.png")
                    Image.fromarray(_to_728(corr_out[0], mss[0])).save(f"{dump_dir}/{model_name}_t{t_start}_{ctype}_corrupt_out.png")
                    dumped += 1

    results: dict = {}
    for p in ALL_LEVELS:
        results[f"level={int(p*100)}/ceiling"] = _mean_metric(ceil_acc[p])
    for (p, ctype), a in rec.items():
        nu, nb = max(1, a["n_units"]), max(1, a["n_boards"])
        cc = max(1, a["collat_cand"])
        entry = {
            "denoised": _mean_metric(denoise_acc[(p, ctype)]),
            "floor": _mean_metric(floor_acc[(p, ctype)]),
            "n_boards": a["n_boards"],
            "recovery_rate": a["rec_units"] / nu,
            "full_recovery_rate": a["full_rec"] / nb,
            "recovered_and_valid_rate": a["rec_and_valid"] / nb,
            "collateral_break_rate": a["collat_broken"] / cc,
            "matches_clean_run_rate": a["matches_clean"] / nb,
        }
        if ctype == "add":
            entry["adapt_rate"] = a["adapt_units"] / nu
        if ctype == "wall":
            entry["still_through_wall_rate"] = a["still_wall"] / nb
            entry["wall_pixel_clear_rate"] = a["wall_line_cleared"] / max(1, a["wall_line_px"])
        results[f"level={int(p*100)}/{ctype}"] = entry
        logger.info(f"[{model_name}] t={t_start} level={int(p*100)} {ctype}: recovery={entry['recovery_rate']:.3f} "
                    f"full={entry['full_recovery_rate']:.3f} denoised_pass={entry['denoised']['pass']:.3f} "
                    f"floor_cov={entry['floor']['coverage']:.3f}")
    return results, processed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pb = cfg.get("probe", {})
    smoke = bool(pb.get("smoke", False))
    num_samples = 8 if smoke else int(pb.get("num_samples", 128))
    batch_size = int(pb.get("batch_size", 32))
    seed = int(pb.get("seed", 0))
    t_starts = [int(t) for t in pb.get("t_starts", [int(pb.get("t_start", 40))])]
    gap_frac = float(pb.get("gap_frac", 0.15))
    model_name = str(pb.get("model_name", "model"))
    checkpoint = cfg.get("checkpoint", None)
    default_data = "data/amaze/test_maze/square/all_square_test.parquet"
    data_parquet = str(pb.get("data_parquet", default_data))
    out_path = str(pb.get("out", f"runs/maze_corruption/{model_name}.json"))
    dump_dir = pb.get("dump_dir", None)
    if smoke and dump_dir is None:
        dump_dir = str(Path(out_path).parent / f"dump_{model_name}")
    dump_n = int(pb.get("dump_n", 4))

    if checkpoint is None:
        raise SystemExit("Set +checkpoint=<...>.pt (and painter.checkpoint=<...> for TRM).")

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
        raise SystemExit(f"t_starts {bad} not in {num_inference_steps}-step schedule {sorted(valid_t, reverse=True)}")
    logger.info(f"[{model_name}] steps={num_inference_steps} t_starts={t_starts} cfg_scale={model.sampling_pipeline.cfg_scale}")

    # Resolve data path relative to the trm_diffusion root when not absolute.
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
    logger.info(f"[{model_name}] dataset={data_parquet} using {n_total}/{len(ds)} puzzles, batch={batch_size}")

    scorer = AmazeMetrics(device=device, task="maze")

    all_results: dict = {}
    processed = 0
    for t_start in t_starts:
        run_timesteps = full_ts[full_ts <= t_start]
        logger.info(f"[{model_name}] --- t_start={t_start} ({len(run_timesteps)} denoise steps) ---")
        sub, processed = _run_one_tstart(model, ds, scorer, device, t_start, run_timesteps, n_total,
                                         batch_size, seed, gap_frac, dump_dir, dump_n, model_name)
        for k, v in sub.items():
            all_results[f"t={t_start}/{k}"] = v

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"model": model_name, "checkpoint": str(checkpoint), "t_starts": t_starts,
                       "num_inference_steps": num_inference_steps, "n_puzzles": processed,
                       "data": data_parquet, "results": all_results}, f, indent=2)
        logger.info(f"[{model_name}] results -> {out_path}"
                    + (f"  (PNG dumps -> {dump_dir})" if dump_dir else ""))
    return all_results


if __name__ == "__main__":
    main()
