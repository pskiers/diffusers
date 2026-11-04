"""TEMPORARY diagnostic (cluster only): the H/V swap with the stroke and wall parts of V−H pushed separately.

Same boards / noise / seed / pushed tokens as interp_hv_swap.py (GT-H tokens of remaining straight-horizontal cells;
mirror: GT-V tokens of remaining straight-vertical cells). Phase 1 (normal runs, even boards = half A), per t, a least
squares fit of the painter's conditional logits on per-token features
    path + shape + cell walls (L/R/T/B) + cell walls×path + markers + position      (plan_decompose / hv_features)
gives
    stroke     = coef(V stroke) − coef(H stroke)                       (the cell's walls held fixed)
    walls      = coef(walls L+R) − coef(walls T+B), incl. ×path        (the cell turned into a vertical corridor)
    walls_off  = mean(off-path tokens in vertical corridors) − mean(off-path tokens in horizontal corridors)
    full       = mean(V tokens) − mean(H tokens)                       (the direction of interp_hv_swap.py)
Phase 2: alpha × direction added to the target tokens' logits (conditional pass), scored on the odd boards: stroke
orientation per target cell, cell still drawn, new / erased wall pixels near the cell's left+right vs top+bottom
sides, side-exit category of the pushed tokens, the rest of the board.
"""
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import hv_features as hf
import interp_hv_swap as hs
import maze_corruption_lib as mcl
import plan_analysis as pa
import plan_decompose as pdc
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

GROUPS = ["path", "shape", "cell_walls", "cell_walls×path", "markers", "position"]
# (name, direction, sign, target set, alpha)
RUNS = ([("normal", None, 1.0, None, 0.0)]
        + [("stroke", "stroke", 1.0, "h2v", a) for a in (1.0, 2.0, 4.0)]
        + [("walls", "walls", 1.0, "h2v", a) for a in (1.0, 2.0, 4.0)]
        + [("walls_off", "walls_off", 1.0, "h2v", a) for a in (2.0, 4.0)]
        + [("stroke_mirror", "stroke", -1.0, "v2h", 2.0), ("full", "full", 1.0, "h2v", 2.0)])


def features(b, i, ms):
    f, _ = pdc.token_features(b["maze_u8"][i], b["sol_u8"][i], b["clean_u8"][i])
    f.update(hf.wall_features(b["maze_u8"][i].numpy(), ms.height))
    f["cell_walls×path"] = f["cell_walls"] * f["path"]
    return f


def wall_change(out_rgb, maze_u8, box):
    """New / erased dark pixels within 2 px of the cell, near its left+right vs top+bottom sides (5-px bands)."""
    y0, y1, x0, x1 = box
    Y0, Y1, X0, X1 = max(0, y0 - 2), min(144, y1 + 2), max(0, x0 - 2), min(144, x1 + 2)
    od = out_rgb.astype(float).mean(-1) < hf.WALL_DARK
    md = maze_u8.float().mean(0).numpy() < hf.WALL_DARK
    new, gone = (od & ~md)[Y0:Y1, X0:X1], (md & ~od)[Y0:Y1, X0:X1]
    lr = lambda a: int(a[:, :5].sum() + a[:, -5:].sum())
    tb = lambda a: int(a[:5, 5:-5].sum() + a[-5:, 5:-5].sum())
    return {"new_lr": lr(new), "new_tb": tb(new), "gone_lr": lr(gone), "gone_tb": tb(gone)}


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
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    starts = list(range(0, n_total, batch_size))

    # ---- phase 1: normal runs; per-t directions from half A ----
    Xs, Ys, cats_A, cor_A = [], {t: [] for t in ts}, [], []
    n_board = 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        rec = []
        hs.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), record=rec)
        for i, ms in enumerate(b["mss"]):
            if (n_board + i) % 2 == 0:
                f = features(b, i, ms)
                Xs.append(np.concatenate([f[g] for g in GROUPS], 1)); cats_A.append(b["cats"][i]); cor_A.append(f["corridor"])
                for t, L in zip(ts, rec):
                    Ys[t].append(L[i])
        n_board += b["B"]
    f0 = features(b, 0, b["mss"][0])
    off_idx, k = {}, 0
    for g in GROUPS:
        off_idx[g] = k; k += f0[g].shape[1]
    X = torch.from_numpy(np.concatenate(Xs)).float()
    cats_A, cor_A = np.concatenate(cats_A), np.concatenate(cor_A)
    sw = torch.tensor([1.0, 1.0, -1.0, -1.0])
    cw, cwp, sh = off_idx["cell_walls"], off_idx["cell_walls×path"], off_idx["shape"]
    dirs = {n: {} for n in ("stroke", "walls", "walls_off", "full")}
    for t in ts:
        Y = torch.cat(Ys[t]).float()
        X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
        W = torch.linalg.solve(X1.T @ X1 + 1e-2 * torch.eye(X1.shape[1]), X1.T @ Y)
        dirs["stroke"][t] = W[sh + 1] - W[sh + 0]
        dirs["walls"][t] = sw @ W[cw:cw + 4] + sw @ W[cwp:cwp + 4]
        off = torch.from_numpy(cats_A == "off")
        dirs["walls_off"][t] = (Y[off & torch.from_numpy(cor_A[:, 1] > 0)].mean(0) - Y[off & torch.from_numpy(cor_A[:, 0] > 0)].mean(0))
        dirs["full"][t] = Y[torch.from_numpy(cats_A == "V")].mean(0) - Y[torch.from_numpy(cats_A == "H")].mean(0)
    c = lambda a, b: float(a @ b / a.norm() / b.norm())
    for t in (70, 35, 5):
        print(f"t={t}: norms " + "  ".join(f"{n} {float(dirs[n][t].norm()):.3f}" for n in dirs)
              + f" | cos with full: stroke {c(dirs['stroke'][t], dirs['full'][t]):+.2f} walls {c(dirs['walls'][t], dirs['full'][t]):+.2f} "
              f"walls_off {c(dirs['walls_off'][t], dirs['full'][t]):+.2f} | cos(walls, walls_off) {c(dirs['walls'][t], dirs['walls_off'][t]):+.2f}",
              flush=True)

    # ---- phase 2: pushes ----
    rows, outs = [], []
    n_board = 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        held = [(n_board + i) % 2 == 1 for i in range(B)]
        batch_out = {}
        for name, dkey, sign, tset, alpha in RUNS:
            push = None
            if dkey is not None:
                mask = torch.from_numpy(np.stack([r["tok"][tset] for r in recs]))
                push = (mask, {t: sign * alpha * dirs[dkey][t] for t in ts})
            x = hs.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), push)
            out = model.decode_for_eval(x).clamp(0, 1)
            out_u8 = (out * 255).byte().cpu()
            batch_out[f"{name}_{alpha:g}"] = out_u8
            for i, (ms, r) in enumerate(zip(mss, recs)):
                rgb = out_u8[i].permute(1, 2, 0).numpy()
                blue = mcl.blue_mask(rgb)  # raw mask: at 144 px the stroke is too thin for the 3x3 opening
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                out_cats = pa.token_labels(out_u8[i], out_u8[i])[0]
                others = r["remaining"] - set(r["h_rem"]) - set(r["v_rem"])
                for s in ("h2v", "v2h"):
                    if tset is not None and s != tset:
                        continue
                    for cid in r["cells"][s]:
                        box = hs.cell_box(ms, cid)
                        rows.append({"run": name, "alpha": alpha, "set": s, "held_out": held[i], "board": n_board + i, "cell": cid,
                                     "orient": hs.cell_orientation(blue, box), "drawn": cid in pred,
                                     **wall_change(rgb, b["maze_u8"][i], box),
                                     "other_remaining_drawn": len(others & pred) / len(others) if others else None,
                                     "off_path": len(pred - r["path"])})
                    for tk in np.nonzero(r["tok"][s])[0]:
                        rows.append({"run": name, "alpha": alpha, "set": s + "_tokens", "held_out": held[i], "out_cat": str(out_cats[tk])})
        outs.append({"n": B, "sol": b["sol_u8"], "clean": b["clean_u8"], "maze": b["maze_u8"], "out": batch_out,
                     "cells": [{k: r["cells"][k] for k in ("h2v", "v2h")} for r in recs],
                     "id2rc": [ms.id2rc for ms in mss], "hw": [(ms.height, ms.width) for ms in mss]})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    print("\nheld-out boards (odd), target cells: stroke orientation | drawn | wall pixels per cell (new / erased, "
          "near left+right vs top+bottom sides) | pushed tokens | rest of the board")
    for name, dkey, sign, tset, alpha in RUNS:
        for s in ((tset,) if tset else ("h2v", "v2h")):
            sel = [r for r in rows if r["run"] == name and r["alpha"] == alpha and r["set"] == s and r["held_out"]]
            tok = [r for r in rows if r["run"] == name and r["alpha"] == alpha and r["set"] == s + "_tokens" and r["held_out"]]
            m = lambda key: np.mean([r[key] for r in sel])
            fr = " ".join(f"{o} {np.mean([r['orient'] == o for r in sel]):.2f}" for o in ("H", "V", "other", "empty"))
            tf = " ".join(f"{o} {np.mean([r['out_cat'] == o for r in tok]):.2f}" for o in ("H", "V", "off"))
            print(f"  {name:13s} a={alpha:g} [{s}] n={len(sel):3d} | {fr} | drawn {m('drawn'):.2f} | new L/R {m('new_lr'):5.1f} "
                  f"T/B {m('new_tb'):5.1f}  erased L/R {m('gone_lr'):5.1f} T/B {m('gone_tb'):5.1f} | tokens {tf} | "
                  f"other remaining drawn {np.mean([r['other_remaining_drawn'] for r in sel if r['other_remaining_drawn'] is not None]):.2f}"
                  f" off-path {m('off_path'):.1f}")
    torch.save({"dirs": dirs, "rows": rows, "batches": outs, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
