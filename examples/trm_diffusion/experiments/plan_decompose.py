"""Offline: what do the thinker's 11 logit channels (and z_H) encode?

(1) Per-channel maps: each logit channel as a 12x12 map next to the maze and the solution.
(2) Concept regression: explain each token's logits / z_H with interpretable per-token features, fitted on half
    of the boards and scored on the other half (held-out R²). Per concept group: R² alone and the unique part
    (drop in R² when that group is removed from the full model). Per logit channel, the same table.
(3) The dead-end: how much the corrupted twin's output differs from the clean twin's at the dead-end tokens,
    compared with how far apart path and off-path tokens are.
Usage: python plan_decompose.py plan_states.pt out_dir
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import plan_analysis as pa

G, P = 12, 12


def token_features(maze_u8, sol_u8, clean_u8):
    """Per-token features of one board (all from images): dict of name -> (144, k) float array."""
    cats, frac, drawn = pa.token_labels(sol_u8, clean_u8)
    m = maze_u8.permute(1, 2, 0).numpy().astype(int)
    wall = (m.mean(-1) < 80)                                                   # dark pixels
    red = (m[..., 0] > 150) & (m[..., 1] < 110) & (m[..., 2] < 110)          # start / goal markers
    patch = lambda a: a.reshape(G, P, G, P).mean((1, 3)).reshape(-1)
    wall_t, red_t = patch(wall), patch(red)
    rr, cc = np.divmod(np.arange(G * G), G)
    # distance (in tokens) to the nearest marker token
    mk = np.nonzero(red_t > 0)[0]
    if len(mk):
        d_marker = np.maximum(np.abs(rr[:, None] - rr[mk][None]), np.abs(cc[:, None] - cc[mk][None])).min(1)
    else:
        d_marker = np.full(G * G, G)
    on = cats != "off"
    shape_names = ["H", "V", "corner_LT", "corner_LB", "corner_RT", "corner_RB", "end"]
    return {
        "path": on[:, None].astype(float),
        "shape": np.stack([cats == s for s in shape_names], 1).astype(float),   # within path: which stroke shape
        "drawn": (on & drawn)[:, None].astype(float),                           # path already in the image
        "walls": np.stack([wall_t, wall_t ** 2], 1),
        "markers": np.stack([red_t > 0, np.exp(-d_marker / 2.0)], 1).astype(float),
        "position": np.concatenate([np.eye(G)[rr], np.eye(G)[cc]], 1),          # row and column one-hots
    }, cats


def fit_r2(Xa, Ya, Xb, Yb, ridge=1e-2):
    """Least squares on A, R² on B, per output column and overall."""
    Xa1 = torch.cat([Xa, torch.ones(len(Xa), 1)], 1); Xb1 = torch.cat([Xb, torch.ones(len(Xb), 1)], 1)
    W = torch.linalg.solve(Xa1.T @ Xa1 + ridge * torch.eye(Xa1.shape[1]), Xa1.T @ Ya)
    res = Yb - Xb1 @ W
    tot = ((Yb - Yb.mean(0)) ** 2)
    return 1 - float((res ** 2).sum() / tot.sum()), (1 - (res ** 2).sum(0) / tot.sum(0)).numpy()


def main(path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    batches = torch.load(path, weights_only=False)
    maze = torch.cat([b["maze_img"] for b in batches]); sol = torch.cat([b["solution_img"] for b in batches])
    clean_img = torch.cat([b["clean_img"] for b in batches]); corr_img = torch.cat([b["corrupted_img"] for b in batches])
    N = len(sol)
    feats, cats = zip(*[token_features(maze[i], sol[i], clean_img[i]) for i in range(N)])
    cats = np.stack(cats)
    groups = list(feats[0])
    F = {g: torch.from_numpy(np.concatenate([f[g] for f in feats])).float() for g in groups}   # (N*144, k)
    board = np.repeat(np.arange(N), G * G)
    A, B = torch.from_numpy(board % 2 == 0), torch.from_numpy(board % 2 == 1)
    dead_end = (((corr_img.float() - clean_img.float()).abs().amax(1) > 12).float()
                .reshape(N, G, P, G, P).mean((2, 4)).reshape(-1).numpy() > 0)

    for t in (70, 35, 5):
        for space in ("logits", "z_H"):
            Y = torch.cat([b["clean"][t][space] for b in batches]).float().reshape(N * G * G, -1)
            full = torch.cat([F[g] for g in groups], 1)
            r2_full, r2_full_ch = fit_r2(full[A], Y[A], full[B], Y[B])
            alone, unique, unique_ch = {}, {}, {}
            for g in groups:
                alone[g] = fit_r2(F[g][A], Y[A], F[g][B], Y[B])[0]
                rest = torch.cat([F[h] for h in groups if h != g], 1)
                r2_rest, r2_rest_ch = fit_r2(rest[A], Y[A], rest[B], Y[B])
                unique[g] = r2_full - r2_rest
                unique_ch[g] = r2_full_ch - r2_rest_ch
            print(f"\n[{space}, t={t}] held-out R², all features: {r2_full:.3f}   unexplained: {1 - r2_full:.3f}")
            print("   " + "".join(f"{g:>10s}" for g in groups))
            print("   alone " + "".join(f"{alone[g]:10.3f}" for g in groups)[6:])
            print("  unique " + "".join(f"{unique[g]:10.3f}" for g in groups)[7:])
            if space == "logits" and t == 70:
                print("   per channel (unique R² per concept; last column = all features):")
                print("   ch " + "".join(f"{g:>9s}" for g in groups) + "      all")
                for c in range(Y.shape[1]):
                    print(f"   {c:2d} " + "".join(f"{unique_ch[g][c]:9.3f}" for g in groups) + f"{r2_full_ch[c]:9.3f}")
                var = Y.var(0); print("   channel variance:", [round(float(v), 3) for v in var])

        # (3) how much of the dead-end reaches the output
        for space in ("logits", "z_H"):
            Yk = torch.cat([b["clean"][t][space] for b in batches]).float().reshape(N * G * G, -1)
            Yc = torch.cat([b["corrupted"][t][space] for b in batches]).float().reshape(N * G * G, -1)
            de = torch.from_numpy(dead_end & (cats.reshape(-1) == "off"))
            path_m, off_m = torch.from_numpy(cats.reshape(-1) != "off"), torch.from_numpy(cats.reshape(-1) == "off")
            d_twin = float((Yc[de] - Yk[de]).norm(dim=-1).mean())
            d_far = float((Yc[off_m & ~de] - Yk[off_m & ~de]).norm(dim=-1).mean())
            gap = float((Yk[path_m].mean(0) - Yk[off_m].mean(0)).norm())
            spread = float((Yk[off_m] - Yk[off_m].mean(0)).norm(dim=-1).mean())
            print(f"[{space}, t={t}] corrupted vs clean twin: mean ‖Δ‖ on dead-end tokens {d_twin:.3f}, on other off-path tokens {d_far:.3f}"
                  f"  | path−off mean gap {gap:.3f}  | typical spread of off-path tokens {spread:.3f}")

    # (1) per-channel maps at t=70, three boards
    t = 70
    L = torch.cat([b["clean"][t]["logits"] for b in batches]).float()   # (N, 144, 11)
    lo, hi = L.reshape(-1, L.shape[-1]).quantile(0.02, 0), L.reshape(-1, L.shape[-1]).quantile(0.98, 0)
    boards = [0, 1, 2]
    fig, axes = plt.subplots(len(boards), 2 + L.shape[-1], figsize=(2.0 * (2 + L.shape[-1]), 2.2 * len(boards)), layout="constrained")
    for r, k in enumerate(boards):
        axes[r, 0].imshow(clean_img[k].permute(1, 2, 0).numpy()); axes[r, 1].imshow(sol[k].permute(1, 2, 0).numpy())
        if r == 0:
            axes[r, 0].set_title("clean board", fontsize=8); axes[r, 1].set_title("GT solution", fontsize=8)
        for c in range(L.shape[-1]):
            axes[r, 2 + c].imshow(L[k, :, c].reshape(G, G).numpy(), cmap="RdBu_r", vmin=float(lo[c]), vmax=float(hi[c]))
            if r == 0:
                axes[r, 2 + c].set_title(f"logit ch {c}", fontsize=8)
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    fig.savefig(out_dir / "logit_channels.png", dpi=110); plt.close(fig)
    print("\nwrote", out_dir / "logit_channels.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
