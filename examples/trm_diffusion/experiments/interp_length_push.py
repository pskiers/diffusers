"""TEMPORARY diagnostic (cluster only): causal test of the board-wide "solution length" signal in the plan code.

Phase 1 (normal runs): per t, the painter's conditional logits; per-position template and the per-token known-feature
fit on the even boards (A), the board offset = mean over off-path tokens of what the features leave unexplained, its
main axis u_t (oriented: + = longer solution, on A) and the spread σ_t of the A boards along it.
Phase 2: add alpha·σ_t·u_t to ALL 144 tokens' logits (conditional pass), alpha in ±2, ±5, ±10 (the boards span about
±2σ); and the same norm along a random direction. Scored on the odd boards: what the painter draws globally (remaining
path drawn, visible path kept, off-path cells drawn, blue pixels, blue pixels per drawn cell ≈ stroke width, dark pixels
vs the maze ≈ walls added/erased) and whether the drawn path reaches the goal.
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
import interp_hv_split as hp
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

GROUPS = ["path", "shape", "drawn", "walls", "markers", "position", "wall_sides", "cell_walls", "corridor"]
RUNS = [("normal", None, 0.0)] + [("length", "u", a) for a in (-10.0, -5.0, -2.0, 2.0, 5.0, 10.0)] \
       + [("random", "r", a) for a in (-10.0, 10.0)]


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

    # ---- phase 1 ----
    Ls, Xs, offmask, lengths, held = {t: [] for t in ts}, [], [], [], []
    n_board = 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        rec = []
        hs.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), record=rec)
        for i, ms in enumerate(b["mss"]):
            f = hp.features(b, i, ms)
            Xs.append(np.concatenate([f[g] for g in GROUPS], 1)); offmask.append(f["path"][:, 0] == 0)
            lengths.append(len(ms.path)); held.append((n_board + i) % 2 == 1)
            for t, L in zip(ts, rec):
                Ls[t].append(L[i])
        n_board += b["B"]
    N = n_board
    held = np.array(held); A = ~held
    X1 = torch.cat([torch.from_numpy(np.concatenate(Xs)).float(), torch.ones(N * 144, 1)], 1)
    Arow = torch.from_numpy(np.repeat(A, 144))
    u, sig = {}, {}
    for t in ts:
        L = torch.stack(Ls[t]).float()                                              # (N, 144, 11)
        R = (L - L[torch.from_numpy(A)].mean(0)).reshape(-1, 11)
        W = torch.linalg.solve(X1[Arow].T @ X1[Arow] + 1e-2 * torch.eye(X1.shape[1]), X1[Arow].T @ R[Arow])
        E = (R - X1 @ W).reshape(N, 144, 11)
        off = torch.stack([E[i][torch.from_numpy(offmask[i])].mean(0) for i in range(N)])
        ut = torch.linalg.eigh(torch.cov(off[torch.from_numpy(A)].T))[1][:, -1]
        s = (off @ ut).numpy()
        sgn = np.sign(np.corrcoef(s[A], np.array(lengths)[A])[0, 1]); ut, s = ut * sgn, s * sgn
        u[t], sig[t] = ut, float(s[A].std())
        if t in (70, 35, 5):
            print(f"t={t}: σ along the axis {sig[t]:.4f}; held-out corr(offset, solution length) "
                  f"{np.corrcoef(s[held], np.array(lengths)[held])[0, 1]:+.2f}", flush=True)
    g = torch.Generator().manual_seed(0)
    rnd = {t: (lambda r: r / r.norm())(torch.randn(11, generator=g)) for t in ts}

    # ---- phase 2 ----
    rows, n_board = [], 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        for name, dk, alpha in RUNS:
            push = None
            if dk is not None:
                push = (torch.ones(B, 144, dtype=torch.bool), {t: alpha * sig[t] * (u[t] if dk == "u" else rnd[t]) for t in ts})
            x = hs.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), push)
            out = model.decode_for_eval(x).clamp(0, 1)
            out_u8 = (out * 255).byte().cpu()
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                rgb = out_u8[i].permute(1, 2, 0).numpy()
                blue = mcl.blue_mask(rgb)
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                od = rgb.astype(float).mean(-1) < hf.WALL_DARK
                md = b["maze_u8"][i].float().mean(0).numpy() < hf.WALL_DARK
                drawn_path = pred & r_["path"]
                rows.append({"run": name, "alpha": alpha, "held_out": bool((n_board + i) % 2 == 1), "length": len(ms.path),
                             "remaining_drawn": len(r_["remaining"] & pred) / len(r_["remaining"]) if r_["remaining"] else None,
                             "visible_kept": len(r_["visible"] & pred) / len(r_["visible"]),
                             "off_path": len(pred - r_["path"]), "blue_px": int(blue.sum()),
                             "px_per_cell": float(blue.sum()) / max(1, len(pred)),
                             "walls_added": int((od & ~md).sum()), "walls_erased": int((md & ~od).sum()),
                             "goal_reached": ms.path[-1] in pred})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    print("\nheld-out boards (odd); alpha in units of σ along the length axis (+ = 'longer solution')")
    keys = ["remaining_drawn", "visible_kept", "off_path", "blue_px", "px_per_cell", "walls_added", "walls_erased", "goal_reached"]
    for name, dk, alpha in RUNS:
        sel = [r_ for r_ in rows if r_["run"] == name and r_["alpha"] == alpha and r_["held_out"]]
        print(f"  {name:7s} alpha={alpha:+5.0f} n={len(sel)} | " + "  ".join(
            f"{k} {np.mean([r_[k] for r_ in sel if r_[k] is not None]):.3g}" for k in keys))
    torch.save({"rows": rows, "u": u, "sigma": sig, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
