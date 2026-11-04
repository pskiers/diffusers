"""TEMPORARY diagnostic (cluster only).

(1) Reasoning dynamics: in normal runs, at t in DYN_AT, the conditional pass's thinker logits after each of the
    n_sup reasoning steps (both twins), per-token z_H at ZH_STEPS (clean), and per board the relative change of
    z_H and z_L from one reasoning step to the next.
(2) What the painter needs from the plan: replace the logits the painter receives (conditional pass only) by
    - "batch_mean":  the mean over the batch's boards (≈ the fixed spatial template), or
    - "other_board": the next board's logits (board i gets board i-1's plan),
    and score what gets drawn, incl. how much of the *other* board's path appears.
Same boards, noise and sampler seed as amaze_interpretability.py.
"""
import dataclasses
import random

import cv2
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import amaze_interpretability as ai
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

DYN_AT = {70, 35, 5}
ZH_STEPS = (1, 2, 4, 8, 16)


def rel_step_change(xs):  # list of (B, T, D) per reasoning step -> (B, n-1): ‖x_k - x_{k-1}‖ / ‖x_k‖
    return torch.stack([(b.float() - a.float()).flatten(1).norm(dim=1) / b.float().flatten(1).norm(dim=1)
                        for a, b in zip(xs[:-1], xs[1:])], 1).cpu()


@torch.no_grad()
def denoise(model, conditions, x, run_ts, generator, plan=None, record=None):
    """CFG reverse process. plan: None, "batch_mean" or "other_board" (what the painter gets in the conditional
    pass). record: dict filled with the reasoning-step data at t in DYN_AT (normal runs only)."""
    predictor = model.sampling_pipeline.predictor
    orig_run_painter = model.run_painter
    for t in run_ts:
        tb = t.expand(x.shape[0]).to(x.device)
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=tb)
        kw = {}
        if record is not None and int(t) in DYN_AT:
            kw = {"logits_out": [], "zH_out": [], "zL_out": []}
        if plan is not None:
            def run_painter(sample, logits):
                if plan == "batch_mean":
                    logits = logits.mean(0, keepdim=True).expand_as(logits)
                else:
                    logits = logits.roll(1, dims=0)
                return orig_run_painter(sample, logits)
            model.run_painter = run_painter
        pred_cond, _, _ = model.forward_with_carry(s, **kw)
        model.run_painter = orig_run_painter
        if kw:
            record[int(t)] = {
                "logits": torch.stack([l.float() for l in kw["logits_out"]], 1).cpu(),                    # (B, n_sup, T, V)
                "z_H": torch.stack([kw["zH_out"][k - 1].half() for k in ZH_STEPS], 1).cpu(),             # (B, len(ZH_STEPS), T, D)
                "dz_H": rel_step_change(kw["zH_out"]), "dz_L": rel_step_change(kw["zL_out"]),
            }
        pred_uncond = model(model.null_condition_sample(s)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
        x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    return x


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    saved, rows = [], []
    for start in range(0, n_total, batch_size):
        base_seed = seed + start * 977
        gen = torch.Generator(device=device).manual_seed(base_seed)
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, ms) for s, ms in ((s, mcl.build_maze_sample(s.metadata)) for s in raw) if ms is not None]
        kept, corr_imgs, clean_imgs, mss, recs = [], [], [], [], []
        for i, (s, ms) in enumerate(pairs):
            rng = random.Random(base_seed + i * 131)
            board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
            img_c, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)
            if not wrong:
                continue
            img_k = board.copy()
            full = [shown[-1]] + gt_walked
            for x, y in zip(full[:-1], full[1:]):
                cv2.line(img_k, mcl._center(ms.id2rc[x], ms.cell_px), mcl._center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
            kept.append(s); corr_imgs.append(img_c); clean_imgs.append(img_k); mss.append(ms)
            recs.append({"wrong": set(wrong), "shown": set(shown) | set(gt_walked),
                         "remaining": set(ms.path) - set(shown) - set(gt_walked), "path": set(ms.path)})
        if len(kept) < 2:
            continue
        B = len(kept)
        batch = AmazeDataset.collate_fn(kept)
        conditions = model._batch_to_sample(batch, device)
        ct = torch.stack([ds.transform(Image.fromarray(im)) for im in corr_imgs]).to(device)
        kt = torch.stack([ds.transform(Image.fromarray(im)) for im in clean_imgs]).to(device)
        noise = torch.randn(ct.shape, generator=gen, device=device)
        tb = torch.full((B,), 70, device=device, dtype=torch.long)
        entry = {"n": B, "clean_img": (kt.clamp(0, 1) * 255).byte().cpu(), "corrupted_img": (ct.clamp(0, 1) * 255).byte().cpu(),
                 "solution_img": (batch.images.float().clamp(0, 1) * 255).byte().cpu(),
                 "maze_img": (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()}
        for boards, img in (("corrupted", ct), ("clean", kt)):
            x_t = model.scheduler.add_noise(img, noise, tb)
            for plan in (None, "batch_mean", "other_board"):
                rec = {} if plan is None else None
                x = denoise(model, conditions, x_t, run_ts, torch.Generator(device=device).manual_seed(base_seed + 1), plan, rec)
                if rec is not None:
                    if boards == "corrupted":  # per-token z_H only for the clean boards (size)
                        for v in rec.values():
                            v.pop("z_H")
                    entry[boards] = rec
                out = model.decode_for_eval(x).clamp(0, 1)
                for i, (ms, r) in enumerate(zip(mss, recs)):
                    pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                    j = (i - 1) % B                                   # the board whose plan "other_board" uses
                    own_rc = {ms.id2rc[c] for c in r["path"]}
                    other_rc = {mss[j].id2rc[c] for c in recs[j]["path"]} - own_rc
                    pred_rc = {ms.id2rc[c] for c in pred}
                    rows.append({"boards": boards, "run": plan or "normal",
                                 "dead_end_kept": len(r["wrong"] & pred) / len(r["wrong"]),
                                 "shown_kept": len(r["shown"] & pred) / len(r["shown"]),
                                 "remaining_drawn": len(r["remaining"] & pred) / len(r["remaining"]) if r["remaining"] else None,
                                 "other_off_path": len(pred - r["path"] - r["wrong"]),
                                 "other_boards_path_drawn": len(other_rc & pred_rc) / len(other_rc) if other_rc else None})
        saved.append(entry)
        print(f"batch start={start}: {B} boards", flush=True)

    keys = sorted({(r["boards"], r["run"]) for r in rows})
    for b, run in keys:
        sel = [r for r in rows if r["boards"] == b and r["run"] == run]
        msg = "  ".join(f"{m}={np.mean([r[m] for r in sel if r[m] is not None]):.3f}"
                        for m in ("dead_end_kept", "shown_kept", "remaining_drawn", "other_off_path", "other_boards_path_drawn"))
        print(f"[plan test] {b:9s} {run:12s} n={len(sel)}  {msg}")
    torch.save({"batches": saved, "rows": rows}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
