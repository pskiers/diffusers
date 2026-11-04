"""Offline diagnostics of the thinker's output to the painter ("paint here?"), from interp_plan_collect.py's file.

Tokens are labelled from the GT solution image (blue pixels, 12x12-px patch per token, 12x12 grid):
  off       - no solution pixels in the patch
  H / V     - the stroke leaves the patch through left+right / top+bottom
  corner_xy - through two adjacent sides (LT, LB, RT, RB)
  end       - through one side only
  other     - 3+ sides, or blue without touching a side
and, among path tokens, drawn (blue on the clean board too) vs remaining (still to draw).
Usage: python plan_analysis.py plan_states.pt out_dir
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BLUE_LO, BLUE_HI = np.array([90, 80, 50]), np.array([140, 255, 255])  # HSV blue (overridden below if the lib is importable)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    import maze_corruption_lib as mcl  # noqa
    blue = lambda rgb: mcl.blue_mask(rgb)
except Exception:
    blue = lambda rgb: cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), BLUE_LO, BLUE_HI) > 0

G, P = 12, 12
CATS = ["H", "V", "corner_LT", "corner_LB", "corner_RT", "corner_RB", "end", "other"]


def token_labels(sol_u8, clean_u8):
    """(3,144,144) uint8 images -> per token: category (str), path fraction, drawn on the clean board (bool)."""
    sol = blue(sol_u8.permute(1, 2, 0).numpy())
    drawn = blue(clean_u8.permute(1, 2, 0).numpy())
    cats, frac, dr = [], [], []
    for r in range(G):
        for c in range(G):
            p = sol[r * P:(r + 1) * P, c * P:(c + 1) * P]
            d = drawn[r * P:(r + 1) * P, c * P:(c + 1) * P]
            frac.append(p.mean()); dr.append(d.mean() > 0.02)
            if p.sum() < 4:
                cats.append("off"); continue
            sides = {"L": p[:, 0].sum() >= 2, "R": p[:, -1].sum() >= 2, "T": p[0, :].sum() >= 2, "B": p[-1, :].sum() >= 2}
            s = "".join(k for k in "LRTB" if sides[k])
            cats.append({"LR": "H", "TB": "V", "LT": "corner_LT", "LB": "corner_LB", "RT": "corner_RT", "RB": "corner_RB"}.get(
                s, "end" if len(s) == 1 else "other"))
    return np.array(cats), np.array(frac), np.array(dr)


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def mean_offdiag_cos(x):
    u = x / x.norm(dim=-1, keepdim=True)
    return float((u @ u.T)[~torch.eye(len(u), dtype=torch.bool)].mean())


def main(path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    batches = torch.load(path, weights_only=False)
    sol = torch.cat([b["solution_img"] for b in batches]); clean_img = torch.cat([b["clean_img"] for b in batches])
    corr_img = torch.cat([b["corrupted_img"] for b in batches])
    N = len(sol)
    labels = [token_labels(sol[i], clean_img[i]) for i in range(N)]
    cats = np.stack([l[0] for l in labels]); drawn = np.stack([l[2] for l in labels])
    on_path = cats != "off"
    # dead-end tokens (corrupted boards): where the two boards differ
    diff = ((corr_img.float() - clean_img.float()).abs().amax(1) > 12).float()
    dead_end = diff.reshape(N, G, P, G, P).mean((2, 4)).reshape(N, -1).numpy() > 0
    counts = {c: int((cats == c).sum()) for c in ["off", *CATS]}
    print("token counts:", counts, " drawn path tokens:", int((on_path & drawn).sum()), " remaining:", int((on_path & ~drawn).sum()))

    ts = sorted(batches[0]["clean"], reverse=True)
    for space in ("logits", "z_H"):
        print(f"\n===== {space} =====")
        vp_by_t = {}
        for t in ts:
            X = torch.cat([b["clean"][t][space] for b in batches]).float()        # (N, 144, D)
            Xc = torch.cat([b["corrupted"][t][space] for b in batches]).float()
            off_m = torch.from_numpy(cats == "off")
            pm = torch.from_numpy(on_path)
            # (1) one direction? per-board v_path and its consistency
            vb = torch.stack([X[i][pm[i]].mean(0) - X[i][off_m[i]].mean(0) for i in range(N) if pm[i].any() and off_m[i].any()])
            vp = vb.mean(0); vp_by_t[t] = vp
            off_mean = X[off_m].mean(0)
            # (2) shape: each category's mean minus the off-path mean
            cvec = {c: X[torch.from_numpy(cats == c)].mean(0) - off_mean for c in CATS if (cats == c).sum() >= 30}
            # (3) drawn vs remaining
            v_dr = X[torch.from_numpy(on_path & drawn)].mean(0) - off_mean
            v_rem = X[torch.from_numpy(on_path & ~drawn)].mean(0) - off_mean
            # (4) position: v_path in the four quadrants of the grid
            quad = []
            rr, cc = np.divmod(np.arange(G * G), G)
            for qr in (0, 1):
                for qc in (0, 1):
                    m = torch.from_numpy(((rr // 6) == qr) & ((cc // 6) == qc))[None].expand(N, -1)
                    quad.append(X[pm & m].mean(0) - X[off_m & m].mean(0))
            # (5) variance explained: 1 direction (path vs off) vs full category one-hot (least squares on token means)
            flat = X.reshape(-1, X.shape[-1]); fc = cats.reshape(-1)
            tot = ((flat - flat.mean(0)) ** 2).sum()
            def r2(groups):
                res = flat.clone()
                for g in np.unique(groups):
                    m = torch.from_numpy(groups == g); res[m] -= flat[m].mean(0)
                return 1 - float((res ** 2).sum() / tot)
            r2_path = r2(np.where(fc == "off", "off", "path"))
            r2_cat = r2(fc)
            r2_cat_dr = r2(np.char.add(fc.astype(str), np.where(drawn.reshape(-1), "_d", "_r")))
            # (6) dead-end tokens on corrupted boards along v_path, vs the same tokens on the clean twin and vs off/path
            u = vp / vp.norm()
            de = torch.from_numpy(dead_end & (cats == "off"))
            proj = lambda Z, m: float((Z[m] @ u).mean())
            print(f"t={t:2d}  |v_path|={vp.norm():.3f}  cos across boards {mean_offdiag_cos(vb):.2f}  "
                  f"cos(drawn, remaining) {cos(v_dr, v_rem):.2f}  |drawn| {v_dr.norm():.3f} |remaining| {v_rem.norm():.3f}  "
                  f"quadrants min cos {min(cos(a, b) for i, a in enumerate(quad) for b in quad[i + 1:]):.2f}  "
                  f"R² path/off {r2_path:.3f}  +shape {r2_cat:.3f}  +shape&drawn {r2_cat_dr:.3f}  "
                  f"dead-end tokens along v_path: corrupted {proj(Xc, de):.3f} clean {proj(X, de):.3f} "
                  f"(off {proj(X, off_m):.3f}, path {proj(X, pm):.3f})")
            if t == ts[0]:
                names = list(cvec)
                M = torch.stack([cvec[c] / cvec[c].norm() for c in names])
                C = (M @ M.T).numpy()
                print("   category direction norms:", {c: round(float(cvec[c].norm()), 3) for c in names})
                print("   cos(category, v_path):", {c: round(cos(cvec[c], vp), 2) for c in names})
                print("   cosine between category directions:")
                print("   " + " ".join(f"{c[:9]:>9s}" for c in names))
                for c, row in zip(names, C):
                    print(f"   {c[:9]:>9s} " + " ".join(f"{x:9.2f}" for x in row))
                # residual after removing v_path: what is left of each category direction, and how H vs V differ
                res = {c: cvec[c] - (cvec[c] @ u) * u for c in names}
                print("   share of each category direction along v_path:", {c: round(float((cvec[c] @ u) ** 2 / cvec[c].norm() ** 2), 2) for c in names})
                if "H" in res and "V" in res:
                    print(f"   residual H vs V cosine (after removing v_path): {cos(res['H'], res['V']):.2f}")
                # figure: PCA of tokens coloured by category (logits only, 11-d)
                if space == "logits":
                    sel = np.random.default_rng(0).choice(flat.shape[0], 8000, replace=False)
                    Y = flat[sel] - flat[sel].mean(0)
                    _, s_, vh = torch.linalg.svd(Y, full_matrices=False)
                    pc = (Y @ vh[:2].T).numpy(); var = (s_ ** 2 / (s_ ** 2).sum()).numpy()
                    colors = {"off": "#c9c7c2", "H": "#2a78d6", "V": "#eb6834", "corner_LT": "#1baf7a", "corner_LB": "#1baf7a",
                              "corner_RT": "#1baf7a", "corner_RB": "#1baf7a", "end": "#9b59b6", "other": "#52514e"}
                    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
                    for c in ["off", "H", "V", "corner_LT", "end", "other"]:
                        m = np.isin(fc[sel], [c] if c != "corner_LT" else ["corner_LT", "corner_LB", "corner_RT", "corner_RB"])
                        axes[0].scatter(pc[m, 0], pc[m, 1], s=4, color=colors[c], label=c if c != "corner_LT" else "corners", alpha=0.6)
                    axes[0].set(title=f"thinker logits per token, t={t}: PCA (PC1 {var[0]:.0%}, PC2 {var[1]:.0%})",
                                xlabel="PC1", ylabel="PC2"); axes[0].legend(frameon=False, markerscale=3)
                    for c in ["corner_LT", "corner_LB", "corner_RT", "corner_RB"]:
                        m = fc[sel] == c
                        axes[1].scatter(pc[m, 0], pc[m, 1], s=5, label=c, alpha=0.7)
                    for c, col in (("H", "#2a78d6"), ("V", "#eb6834")):
                        m = fc[sel] == c
                        axes[1].scatter(pc[m, 0], pc[m, 1], s=5, color=col, label=c, alpha=0.5)
                    axes[1].set(title="path tokens only, by stroke shape", xlabel="PC1", ylabel="PC2"); axes[1].legend(frameon=False, markerscale=3)
                    for ax in axes:
                        ax.spines[["top", "right"]].set_visible(False)
                    fig.savefig(out_dir / "logits_pca.png", dpi=130); plt.close(fig)
                    # examples: projection onto v_path per token next to the solution
                    fig, axes = plt.subplots(4, 3, figsize=(9, 12), layout="constrained")
                    for k in range(4):
                        axes[k, 0].imshow(clean_img[k].permute(1, 2, 0).numpy()); axes[k, 0].set_title("clean board" if k == 0 else "")
                        axes[k, 1].imshow(sol[k].permute(1, 2, 0).numpy()); axes[k, 1].set_title("GT solution" if k == 0 else "")
                        im = axes[k, 2].imshow((X[k] @ u).reshape(G, G).numpy(), cmap="Blues")
                        axes[k, 2].set_title(f"logits along v_path, t={t}" if k == 0 else ""); fig.colorbar(im, ax=axes[k, 2], shrink=0.8)
                    for ax in axes.flat:
                        ax.set(xticks=[], yticks=[])
                    fig.savefig(out_dir / "v_path_examples.png", dpi=110); plt.close(fig)
        print("   cos(v_path) across t:", {f"{a}-{b}": round(cos(vp_by_t[a], vp_by_t[b]), 2) for a, b in zip(ts, ts[1:])})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
