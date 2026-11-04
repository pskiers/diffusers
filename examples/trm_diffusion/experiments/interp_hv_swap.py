"""TEMPORARY diagnostic (cluster only): can a push along the plan's vertical−horizontal axis swap a stroke's orientation?

Clean boards (prefix + gt_walked visible), same boards / noise / sampler seed as amaze_interpretability.py, CFG.
Phase 1 (normal runs): the logits the painter receives in the conditional pass, per t; from the even boards (half A)
  d_t = mean(logits of GT "V" tokens) − mean(logits of GT "H" tokens)   (11-dim; ×1 = the natural V−H gap).
Phase 2: denoise again, adding alpha·d_t (conditional pass only) to the target tokens' logits:
  h2v    remaining (not visible) path cells where the GT path runs straight horizontally; their GT-H tokens get +alpha·d_t
  h2v_rand  the same tokens, a random 11-dim direction of the same norm per t
  v2h    mirror: remaining straight-vertical cells, their GT-V tokens get −alpha·d_t
  h2v_visible  visible straight-horizontal cells (the painter also sees the stroke in x_t), alpha=4
Scored on the odd boards (half B, held out from d_t): per target cell the stroke orientation in the output (which
cell sides the blue stroke leaves through), blue pixels on walls, whether the cell is still drawn; per pushed token
the side-exit category of its output patch; and on the rest of the board remaining path drawn / off-path cells drawn.
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
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, P, IMG = 12, 12, 144
RUNS = ([("normal", None, 0.0)]
        + [(k, k, a) for k in ("h2v", "h2v_rand", "v2h") for a in (1.0, 2.0, 4.0)]
        + [("h2v_visible", "h2v_visible", 4.0)])


@torch.no_grad()
def denoise(model, conditions, x, run_ts, generator, push=None, record=None):
    """CFG reverse process. push: (mask (B, T) bool, {t: (11,) vector}) added to the conditional pass's logits.
    record: list filled with the painter's conditional logits per t."""
    predictor = model.sampling_pipeline.predictor
    orig_run_painter = model.run_painter
    for t in run_ts:
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))

        def run_painter(sample, logits, t=int(t)):
            if record is not None:
                record.append(logits.float().cpu())
            if push is not None:
                mask, vec = push
                logits = logits + mask[..., None].to(logits.device, logits.dtype) * vec[t].to(logits.device, logits.dtype)
            return orig_run_painter(sample, logits)
        model.run_painter = run_painter
        pred_cond, _, _ = model.forward_with_carry(s)
        model.run_painter = orig_run_painter
        pred_uncond = model(model.null_condition_sample(s)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
        x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    return x


def straight_cells(ms, cells, axis):
    """Cells of `cells` where the GT path enters and leaves along the same row (axis="h") or column (axis="v")."""
    out = []
    for k in range(1, len(ms.path) - 1):
        c = ms.path[k]
        if c not in cells:
            continue
        (r0, c0), (r1, c1), (r2, c2) = ms.id2rc[ms.path[k - 1]], ms.id2rc[c], ms.id2rc[ms.path[k + 1]]
        if (axis == "h" and r0 == r1 == r2) or (axis == "v" and c0 == c1 == c2):
            out.append(c)
    return out


def cell_box(ms, cid):
    r, c = ms.id2rc[cid]
    ch, cw = IMG / ms.height, IMG / ms.width
    return int(round(r * ch)), int(round((r + 1) * ch)), int(round(c * cw)), int(round((c + 1) * cw))


def cell_tokens(ms, cids, cats, cat):
    """(T,) bool: tokens of category `cat` overlapping one of the cells by ≥6 px in both directions."""
    m = np.zeros(G * G, bool)
    for cid in cids:
        y0, y1, x0, x1 = cell_box(ms, cid)
        for tr in range(G):
            for tc in range(G):
                oy = min(y1, (tr + 1) * P) - max(y0, tr * P); ox = min(x1, (tc + 1) * P) - max(x0, tc * P)
                if oy >= 6 and ox >= 6 and cats[tr * G + tc] == cat:
                    m[tr * G + tc] = True
    return m


def cell_orientation(blue, box):
    """Which sides of the cell the stroke leaves through -> "H", "V", "empty" or "other"."""
    y0, y1, x0, x1 = box
    b = blue[y0:y1, x0:x1]
    if b.sum() < 4:
        return "empty"
    h, w = b.shape
    mid_r, mid_c = slice(h // 4, h - h // 4), slice(w // 4, w - w // 4)
    sides = {"L": b[mid_r, :3].sum() >= 3, "R": b[mid_r, -3:].sum() >= 3,
             "T": b[:3, mid_c].sum() >= 3, "B": b[-3:, mid_c].sum() >= 3}
    s = "".join(k for k in "LRTB" if sides[k])
    return {"LR": "H", "TB": "V"}.get(s, "other")


def prepare(ds, model, device, start, n_total, batch_size, seed):
    base_seed = seed + start * 977
    gen = torch.Generator(device=device).manual_seed(base_seed)
    raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
    pairs = [(s, ms) for s, ms in ((s, mcl.build_maze_sample(s.metadata)) for s in raw) if ms is not None]
    kept, clean_imgs, mss, recs = [], [], [], []
    for i, (s, ms) in enumerate(pairs):
        rng = random.Random(base_seed + i * 131)
        board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
        _, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)  # same rng use as the main script
        if not wrong:
            continue
        img_k = board.copy()
        full = [shown[-1]] + gt_walked
        for x, y in zip(full[:-1], full[1:]):
            cv2.line(img_k, mcl._center(ms.id2rc[x], ms.cell_px), mcl._center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
        kept.append(s); clean_imgs.append(img_k); mss.append(ms)
        vis = set(shown) | set(gt_walked)
        recs.append({"visible": vis, "remaining": set(ms.path) - vis, "path": set(ms.path)})
    if not kept:
        return None
    B = len(kept)
    batch = AmazeDataset.collate_fn(kept)
    conditions = model._batch_to_sample(batch, device)
    kt = torch.stack([ds.transform(Image.fromarray(im)) for im in clean_imgs]).to(device)
    noise = torch.randn(kt.shape, generator=gen, device=device)
    x_t = model.scheduler.add_noise(kt, noise, torch.full((B,), 70, device=device, dtype=torch.long))
    sol_u8 = (batch.images.float().clamp(0, 1) * 255).byte().cpu()
    clean_u8 = (kt.clamp(0, 1) * 255).byte().cpu()
    maze_u8 = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
    cats = np.stack([pa.token_labels(sol_u8[i], clean_u8[i])[0] for i in range(B)])
    for i, (ms, r) in enumerate(zip(mss, recs)):
        r["h_rem"] = straight_cells(ms, r["remaining"], "h")
        r["v_rem"] = straight_cells(ms, r["remaining"], "v")
        r["h_vis"] = straight_cells(ms, r["visible"] - {ms.path[0]}, "h")
        r["tok"] = {"h2v": cell_tokens(ms, r["h_rem"], cats[i], "H"), "v2h": cell_tokens(ms, r["v_rem"], cats[i], "V"),
                    "h2v_visible": cell_tokens(ms, r["h_vis"], cats[i], "H")}
        r["tok"]["h2v_rand"] = r["tok"]["h2v"]
        r["cells"] = {"h2v": r["h_rem"], "h2v_rand": r["h_rem"], "v2h": r["v_rem"], "h2v_visible": r["h_vis"]}
        # can the stroke turn vertical / horizontal here? (openings of the cell in the maze)
        r["open"] = {}
        for cid in set(r["h_rem"]) | set(r["v_rem"]) | set(r["h_vis"]):
            rr, cc = ms.id2rc[cid]
            r["open"][cid] = {"up": mcl._is_open(ms.morig, rr, cc, rr - 1, cc, ms.cell_px),
                              "down": mcl._is_open(ms.morig, rr, cc, rr + 1, cc, ms.cell_px),
                              "left": mcl._is_open(ms.morig, rr, cc, rr, cc - 1, ms.cell_px),
                              "right": mcl._is_open(ms.morig, rr, cc, rr, cc + 1, ms.cell_px)}
    return dict(base_seed=base_seed, B=B, conditions=conditions, x_t=x_t, mss=mss, recs=recs, cats=cats,
                sol_u8=sol_u8, clean_u8=clean_u8, maze_u8=maze_u8)


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
    ts = [int(t) for t in run_ts]

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    starts = list(range(0, n_total, batch_size))

    # ---- phase 1: normal runs, the painter's conditional logits per t; the V−H direction from half A ----
    sums = {t: {"H": torch.zeros(11), "V": torch.zeros(11)} for t in ts}
    cnt = {"H": 0, "V": 0}
    n_board = 0
    for start in starts:
        b = prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        rec = []
        denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), record=rec)
        for i in range(b["B"]):
            if (n_board + i) % 2 == 0:
                for k in ("H", "V"):
                    m = torch.from_numpy(b["cats"][i] == k)
                    for t, L in zip(ts, rec):
                        sums[t][k] += L[i][m].sum(0)
                    cnt[k] += int(m.sum())
        n_board += b["B"]
    d = {t: sums[t]["V"] / cnt["V"] - sums[t]["H"] / cnt["H"] for t in ts}
    g = torch.Generator().manual_seed(0)
    d_rand = {}
    for t in ts:
        r = torch.randn(11, generator=g)
        d_rand[t] = r / r.norm() * d[t].norm()
    print("V−H direction per t (norm):", {t: round(float(d[t].norm()), 3) for t in ts}, flush=True)
    print("cos(d_t, d_70):", {t: round(float(d[t] @ d[70] / d[t].norm() / d[70].norm()), 3) for t in ts}, flush=True)

    # ---- phase 2: pushes ----
    rows, outs = [], []
    n_board = 0
    for start in starts:
        b = prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        held = [(n_board + i) % 2 == 1 for i in range(B)]
        batch_out = {}
        for name, kind, alpha in RUNS:
            if kind is None:
                push = None
            else:
                mask = torch.from_numpy(np.stack([r["tok"][kind] for r in recs]))
                base = d_rand if kind == "h2v_rand" else d
                sign = -1.0 if kind == "v2h" else 1.0
                push = (mask, {t: sign * alpha * base[t] for t in ts})
            x = denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), push)
            out = model.decode_for_eval(x).clamp(0, 1)
            out_u8 = (out * 255).byte().cpu()
            batch_out[f"{name}_{alpha:g}"] = out_u8
            for i, (ms, r) in enumerate(zip(mss, recs)):
                rgb = out_u8[i].permute(1, 2, 0).numpy()
                blue = mcl.blue_mask(rgb)  # raw mask: at 144 px the stroke is too thin for the 3x3 opening
                wall = b["maze_u8"][i].float().mean(0).numpy() < 80
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                out_cats = pa.token_labels(out_u8[i], out_u8[i])[0]
                for kind_s in ("h2v", "v2h", "h2v_visible"):   # score every target set in every run (normal = baseline)
                    cells = r["cells"][kind_s]
                    others = r["remaining"] - set(r["h_rem"]) - set(r["v_rem"])
                    for cid in cells:
                        box = cell_box(ms, cid)
                        y0, y1, x0, x1 = box
                        rows.append({"run": name, "alpha": alpha, "set": kind_s, "board": n_board + i, "held_out": held[i],
                                     "cell": cid, "orient": cell_orientation(blue, box), "drawn": cid in pred,
                                     "wall_px": int((blue[y0:y1, x0:x1] & wall[y0:y1, x0:x1]).sum()),
                                     **{f"open_{k}": v for k, v in r["open"][cid].items()},
                                     "other_remaining_drawn": len(others & pred) / len(others) if others else None,
                                     "off_path": len(pred - r["path"])})
                    tok = r["tok"][kind_s]
                    for tk in np.nonzero(tok)[0]:
                        rows.append({"run": name, "alpha": alpha, "set": kind_s + "_tokens", "board": n_board + i,
                                     "held_out": held[i], "token": int(tk), "out_cat": str(out_cats[tk])})
        outs.append({"n": B, "sol": b["sol_u8"], "clean": b["clean_u8"], "maze": b["maze_u8"], "out": batch_out,
                     "masks": {k: np.stack([r["tok"][k] for r in recs]) for k in ("h2v", "v2h", "h2v_visible")},
                     "cells": [{k: r["cells"][k] for k in ("h2v", "v2h", "h2v_visible")} for r in recs],
                     "id2rc": [ms.id2rc for ms in mss], "hw": [(ms.height, ms.width) for ms in mss]})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    # ---- summary on the held-out boards ----
    print("\nheld-out boards (odd): stroke orientation in the target cells (fractions), per run")
    for name, kind, alpha in RUNS:
        for s in ("h2v", "v2h", "h2v_visible"):
            if kind is not None and kind.replace("_rand", "") != s:
                continue
            sel = [r for r in rows if r["run"] == name and r["alpha"] == alpha and r["set"] == s and r["held_out"]]
            if not sel:
                continue
            fr = {o: np.mean([r["orient"] == o for r in sel]) for o in ("H", "V", "other", "empty")}
            vert_ok = [r for r in sel if r["open_up"] or r["open_down"]]
            horz_ok = [r for r in sel if r["open_left"] or r["open_right"]]
            tok = [r for r in rows if r["run"] == name and r["alpha"] == alpha and r["set"] == s + "_tokens" and r["held_out"]]
            tf = {o: np.mean([r["out_cat"] == o for r in tok]) for o in ("H", "V", "off")} if tok else {}
            print(f"  {name:12s} a={alpha:g} cells[{s:11s}] n={len(sel):4d}  H {fr['H']:.2f}  V {fr['V']:.2f}  other {fr['other']:.2f}"
                  f"  empty {fr['empty']:.2f} | drawn {np.mean([r['drawn'] for r in sel]):.2f}  wall px {np.mean([r['wall_px'] for r in sel]):.1f}"
                  f" | V where up/down open (n={len(vert_ok)}) {np.mean([r['orient'] == 'V' for r in vert_ok]) if vert_ok else float('nan'):.2f}"
                  f"  H where left/right open (n={len(horz_ok)}) {np.mean([r['orient'] == 'H' for r in horz_ok]) if horz_ok else float('nan'):.2f}"
                  f" | tokens n={len(tok)} " + " ".join(f"{k} {v:.2f}" for k, v in tf.items())
                  + f" | other remaining drawn {np.mean([r['other_remaining_drawn'] for r in sel if r['other_remaining_drawn'] is not None]):.2f}"
                  f"  off-path cells {np.mean([r['off_path'] for r in sel]):.1f}")
    torch.save({"d": d, "d_rand": d_rand, "rows": rows, "batches": outs, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
