"""TEMPORARY diagnostic (Claude, amaze interpretability): pictures of the mean-difference z_H push.

Same boards / noise / sampler seed / directions as interp_z_edit.py (Phase 8, helios 24769760, runs/amaze_interp_v17):
clean boards (prefix + gt_walked visible) noised to t = 70, CFG; d_{t,c} = mean(path tokens) − mean(off tokens) of
z_H at call c of denoising step t on the even boards; the push is added to the z_H output at EVERY z_H call of the
conditional pass at every denoising step (alpha = 1: one full path−off gap).
Runs, on the first batch only (pictures from the held-out odd boards):
  normal
  -path on a 4-cell stretch of the remaining path: alpha 1, 2; random direction (same norm per (t, c)) alpha 4
  +path on the thinker's own dropped side-branch tokens (drop_concept.pt): alpha 2, 4; random direction alpha 4
Boards shown = the first n_show held-out boards that have a 4-cell stretch and dropped tokens (not hand-picked).
Output in out_dir: z_gallery.png, z_gallery.pt (all images + per-board scores, to draw other boards).
"""
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import amaze_interpretability as ai
import interp_az_push as az
import interp_hv_swap as hs
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from interp_z_edit import denoise

# (column title, kind: push|rand|None, target, sign, alpha)
RUNS = [("normal", None, None, 0, 0.0),
        ("-path x1", "push", "path", -1, 1.0), ("-path x2", "push", "path", -1, 2.0), ("random x4", "rand", "path", -1, 4.0),
        ("+path x2", "push", "drop", 1, 2.0), ("+path x4", "push", "drop", 1, 4.0), ("random x4 ", "rand", "drop", 1, 4.0)]
ORANGE, MAGENTA = "#ff8c00", "#e0119d"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_show = int(ic.get("n_show", 3))
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
    tc = model.thinker.inner.config
    per = tc.L_cycles + 1
    kind_of = lambda c: "z_H" if c % per == per - 1 else "z_L"
    drop_mask = torch.load(str(ic.drop_concept), weights_only=False)["drop_mask"].numpy()
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0

    # ---- phase 1 (as interp_z_edit.py, z_H calls only): per (t, call) mean path / off state on the even boards ----
    S, n_board = {}, 0
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        even = torch.from_numpy(np.array([(n_board + i) % 2 == 0 for i in range(b["B"])]))
        gt = torch.from_numpy(b["cats"] != "off")
        mp = (gt & even[:, None]).float().to(device)[..., None]; mo = ((~gt) & even[:, None]).float().to(device)[..., None]

        def acc(t, c, out):
            if kind_of(c) != "z_H":
                return None
            o = out.float()
            sp, so = (o * mp).sum((0, 1)), (o * mo).sum((0, 1))
            if (t, c) in S:
                S[(t, c)][0] += sp; S[(t, c)][1] += so
            else:
                S[(t, c)] = [sp, so]
            return None
        denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), acc, b["B"])
        S.setdefault("n", [0, 0]); S["n"][0] += float(mp.sum()); S["n"][1] += float(mo.sum())
        n_board += b["B"]
    npth, noff = S.pop("n")
    D = {k: (v[0] / npth - v[1] / noff) for k, v in S.items()}
    g = torch.Generator(device="cpu").manual_seed(0)
    R = {}
    for k, v in sorted(D.items()):
        r = torch.randn(v.numel(), generator=g).to(device)
        R[k] = r / r.norm() * v.norm()
    print(f"phase 1: {n_board} boards, {len(D)} (t, z_H call) directions", flush=True)

    # ---- phase 2: the first batch, all runs ----
    b = hs.prepare(ds, model, device, 0, n_total, batch_size, seed)
    B, mss, recs = b["B"], b["mss"], b["recs"]
    cells_p, mp_ = zip(*[az.stretch_tokens(ms, r_, b["cats"][i]) for i, (ms, r_) in enumerate(zip(mss, recs))])
    masks = {"path": torch.from_numpy(np.stack(mp_)), "drop": torch.from_numpy(drop_mask[:B])}
    imgs, scores = {}, {}
    for name, kind, tgt, sign, alpha in RUNS:
        m = None if tgt is None else masks[tgt].to(device)[..., None].float()

        def fn(t, c, out, kind=kind, m=m, sign=sign, alpha=alpha):
            if kind is None or kind_of(c) != "z_H":
                return None
            vec = (D if kind == "push" else R)[(t, c)]
            return (out.float() + m * (sign * alpha) * vec).to(out.dtype)
        out = denoise(model, b["conditions"], b["x_t"], run_ts,
                      torch.Generator(device=device).manual_seed(b["base_seed"] + 1), fn, B)
        out_u8 = (out * 255).byte().cpu()
        imgs[name] = out_u8
        sc = []
        for i, ms in enumerate(mss):
            pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
            oc = pa.token_labels(out_u8[i], out_u8[i])[0]
            dm = masks["drop"][i].numpy()
            sc.append({"stretch_drawn": len(set(cells_p[i]) & pred) / max(1, len(cells_p[i])),
                       "drop_drawn": float((oc[dm] != "off").mean()) if dm.any() else None})
        scores[name] = sc
        held = [s for i, s in enumerate(sc) if i % 2 == 1]
        dd = [s["drop_drawn"] for s in held if s["drop_drawn"] is not None]
        print(f"  {name:11s} held-out n={len(held)} | stretch drawn {np.mean([s['stretch_drawn'] for s in held]):.3f}  "
              f"branch tokens drawn {np.mean(dd) if dd else float('nan'):.3f}", flush=True)

    # ---- figure ----
    show = [i for i in range(B) if i % 2 == 1 and len(cells_p[i]) == 4 and masks["drop"][i].any()][:n_show]
    cols = [("input (before noise)", None), ("solution", None)] + [(r_[0], r_) for r_ in RUNS]
    fig, axes = plt.subplots(len(show), len(cols), figsize=(2.0 * len(cols), 2.25 * len(show)), squeeze=False)
    for row, i in enumerate(show):
        ms = mss[i]
        for col, (title, run) in enumerate(cols):
            ax = axes[row, col]
            if run is None:
                im = (b["clean_u8"] if col == 0 else b["sol_u8"])[i]
                txt = f"board {i}" if col == 0 else ""
            else:
                im = imgs[run[0]][i]
                s = scores[run[0]][i]
                txt = {"path": f"stretch {s['stretch_drawn'] * 4:.0f}/4",
                       "drop": f"branch {s['drop_drawn']:.0%}"}.get(run[2], f"stretch {s['stretch_drawn'] * 4:.0f}/4, "
                                                                          f"branch {s['drop_drawn']:.0%}")
            ax.imshow(im.permute(1, 2, 0).numpy(), interpolation="nearest")
            tgt = None if run is None else run[2]
            if col >= 2 and tgt in (None, "path"):
                for cid in cells_p[i]:
                    y0, y1, x0, x1 = hs.cell_box(ms, cid)
                    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.4))
            if col >= 2 and tgt in (None, "drop"):
                for tok in np.flatnonzero(masks["drop"][i].numpy()):
                    tr, tc_ = divmod(int(tok), hs.G)
                    ax.add_patch(Rectangle((tc_ * hs.P - 0.5, tr * hs.P - 0.5), hs.P, hs.P, fill=False, ec=MAGENTA, lw=1.0))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xlabel(txt, fontsize=8)
    fig.suptitle("Mean-difference push on z_H at every z_H call, every denoising step (alpha 1 = one path-off gap). "
                 "Orange: 4 targeted solution cells (-path). Magenta: tokens of side branches the thinker drops itself (+path).",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "z_gallery.png", dpi=200)
    torch.save({"runs": RUNS, "imgs": imgs, "scores": scores, "shown": show, "stretch_cells": cells_p,
                "stretch_mask": masks["path"], "drop_mask": masks["drop"], "clean_u8": b["clean_u8"], "sol_u8": b["sol_u8"]},
               out_dir / "z_gallery.pt")
    print("saved", out_dir / "z_gallery.png", flush=True)


if __name__ == "__main__":
    main()
