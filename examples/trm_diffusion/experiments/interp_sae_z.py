"""TEMPORARY diagnostic (cluster only, no model needed): sparse dictionaries (TopK SAEs) on the thinker's 512-dim states.

Question: is the state a sparse combination of many features (superposition)? Datasets (clean boards, t=70 unless
noted; 98 boards, fit on even boards, scored on odd boards):
  zH_final  final z_H of the pass (the state the plan is read from) at t in {70, 50, 35, 20, 5}   (plan_states.pt)
  zL_step1  z_L after each of the 18 z_L updates of reasoning step 1                              (step1.pt)
Each dataset is centred and scaled (even boards). Test: held-out fraction of variance unexplained (FVU) of the SAE
on the real states vs on Gaussian data with the same mean and covariance (the null: an overcomplete dictionary
fits any data somewhat; sparse features should make the real data clearly easier), and PCA with k components.
For the real SAE: per atom how often it is active (odd boards), what known per-token features predict it (held-out
AUC of a logistic probe on path / stroke shape / walls per side / corridor / markers / position; low = not explained),
enrichment; prototype figure of the least explained frequent atoms.
Usage: python interp_sae_z.py plan_states.pt step1.pt out_dir
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import hv_features as hf
import plan_decompose as pdc

G = 12
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = ["H", "V", "corner_LT", "corner_LB", "corner_RT", "corner_RB", "end"]


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.W_e = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
        self.b_e = torch.nn.Parameter(torch.zeros(m))
        self.D = torch.nn.Parameter(self.W_e.data.T.clone())
        self.b_d = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        a = torch.relu((x - self.b_d) @ self.W_e.T + self.b_e)
        top = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter(-1, top.indices, top.values)

    def forward(self, x):
        c = self.encode(x)
        return c @ (self.D / self.D.norm(dim=0, keepdim=True)).T + self.b_d, c


def train_sae(X, m, k, steps=6000, seed=0):
    torch.manual_seed(seed)
    sae = TopKSAE(X.shape[1], m, k).to(DEV)
    sae.b_d.data = X.mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    scale = float(X.norm(dim=1).mean())
    for s in range(steps):
        idx = torch.randint(len(X), (4096,), device=DEV)
        xh, _ = sae(X[idx])
        loss = ((xh - X[idx]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()
        assert torch.isfinite(loss), f"SAE training diverged at step {s}"
        if s % 1000 == 999 and s < steps - 1000:                                   # revive dead atoms
            with torch.no_grad():
                act = torch.zeros(m, device=DEV)
                for ch in X.split(16384):
                    act += (sae.encode(ch) > 0).float().sum(0)
                dead = act / len(X) < 1e-5
                if dead.any():
                    err = torch.cat([((sae(ch)[0] - ch) ** 2).sum(-1) for ch in X.split(16384)])
                    pick = X[err.topk(int(dead.sum())).indices] - sae.b_d
                    pick = pick / pick.norm(dim=1, keepdim=True)
                    sae.W_e.data[dead] = pick / scale; sae.D.data[:, dead] = pick.T; sae.b_e.data[dead] = 0
    return sae


@torch.no_grad()
def fvu(sae, X):
    err = sum(float(((sae(ch)[0] - ch) ** 2).sum()) for ch in X.split(16384))
    return err / float(((X - X.mean(0)) ** 2).sum())


def fit_probe(X, y, train):
    m, s = X[train].mean(0), X[train].std(0) + 1e-6
    Xa = (X[train] - m) / s
    w = torch.zeros(X.shape[1] + 1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=100)

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w[:-1] + w[-1], y[train]) + 1e-3 * w[:-1].pow(2).sum()
        l.backward()
        return l
    opt.step(cl)
    w = w.detach()
    return lambda Z: ((Z - m) / s) @ w[:-1] + w[-1]


def auc(score, y):
    p, n = score[y > 0.5], score[y < 0.5]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    g = torch.Generator(device=score.device).manual_seed(0)
    i = torch.randint(len(p), (20000,), device=score.device, generator=g); j = torch.randint(len(n), (20000,), device=score.device, generator=g)
    return float((p[i] > n[j]).float().mean())


def known_features(maze, sol, clean):
    N = len(sol)
    feats, cats = zip(*[pdc.token_features(maze[i], sol[i], clean[i]) for i in range(N)])
    walls = [hf.wall_features(maze[i].numpy(), 8) for i in range(N)]
    cat = lambda g: np.concatenate([f[g] for f in feats]); catw = lambda g: np.concatenate([w[g] for w in walls])
    named = {"path": cat("path")[:, 0], "drawn": cat("drawn")[:, 0], "marker": cat("markers")[:, 0]}
    for j, s in enumerate(SHAPES):
        named["stroke_" + s] = cat("shape")[:, j]
    for j, s in enumerate("LRTB"):
        named["cellwall_" + s] = catw("cell_walls")[:, j]
    for j, s in enumerate(["Hcorr", "Vcorr", "othercell"]):
        named["cell_" + s] = catw("corridor")[:, j]
    K = np.concatenate([np.stack(list(named.values()), 1), catw("wall_sides"), cat("markers")[:, 1:], cat("position")], 1)
    return named, K.astype(np.float32)


def analyse(name, X, board, named, K, sol, out_dir, reps, group):
    """X: (n, 512) tokens; board: (n,) board index; group: (n,) stage (denoising step or z_L update) index;
    named/K: per-token features (n, ...). The mean per (stage, grid position) over the even boards is removed first
    (the fixed template), so a dictionary cannot win by learning one direction per position or stage."""
    A, B = board % 2 == 0, board % 2 == 1
    pos = np.arange(len(X)) % (G * G)
    key = torch.from_numpy(group * (G * G) + pos)
    nk = int(key.max()) + 1
    At_ = torch.from_numpy(A)
    sums = torch.zeros(nk, X.shape[1]).index_add_(0, key[At_], X[At_])
    cnt = torch.zeros(nk).index_add_(0, key[At_], torch.ones(int(At_.sum())))
    X = X - (sums / cnt.clamp(min=1)[:, None])[key]
    sc = X[At_].pow(2).sum(-1).mean().sqrt()
    X = (X / sc).to(DEV)
    XA, XB = X[torch.from_numpy(A).to(DEV)], X[torch.from_numpy(B).to(DEV)]
    cov = torch.cov(XA.T)
    ev = torch.linalg.eigvalsh(cov).flip(0)
    mvn = torch.distributions.MultivariateNormal(XA.mean(0), cov + 1e-5 * torch.eye(cov.shape[0], device=DEV))
    torch.manual_seed(1)
    GA, GB = mvn.sample((len(XA),)), mvn.sample((len(XB),))
    print(f"\n===== {name}: {len(XA)} train / {len(XB)} held-out tokens, dim {X.shape[1]}; "
          f"PCA: 90% of variance in {int((ev.cumsum(0) / ev.sum() < 0.9).sum()) + 1} components =====", flush=True)
    print("  held-out FVU:  SAE on real states | SAE on Gaussian null | PCA with k components", flush=True)
    best = None
    for m, k in reps:
        s_real = train_sae(XA, m, k); f_real = fvu(s_real, XB)
        f_null = fvu(train_sae(GA, m, k), GB)
        f_pca = 1 - float(ev[:k].sum() / ev.sum())
        f_train = fvu(s_real, XA)
        print(f"    m={m:5d} k={k:3d}: real {f_real:.3f} (train {f_train:.3f}) | null {f_null:.3f} | PCA {f_pca:.3f}   (real/null {f_real / f_null:.2f})", flush=True)
        assert 0 <= f_real < 1 and 0 <= f_null < 1, "FVU out of [0, 1): training failed, results invalid"
        best = (s_real, m, k)
    sae, m, k = best
    with torch.no_grad():
        C = torch.cat([sae.encode(ch) for ch in X.split(16384)])
    act = (C > 0)
    freqB = act[torch.from_numpy(B).to(DEV)].float().mean(0).cpu().numpy()
    alive = int((freqB > 0).sum())
    Kt = torch.from_numpy(K).to(DEV); At, Bt = torch.from_numpy(A).to(DEV), torch.from_numpy(B).to(DEV)
    cand = np.argsort(-freqB)[:300]
    cand = [j for j in cand if freqB[j] >= 0.005]
    rows = []
    for j in cand:
        y = act[:, j].float()
        a_ = auc(fit_probe(Kt, y, At)(Kt)[Bt], y[Bt])
        enr = sorted(((named[n][act[:, j].cpu().numpy() & B].mean() / max(named[n][B].mean(), 1e-6), n)
                      for n in named if named[n][B].mean() > 0.005), reverse=True)[:3]
        rows.append((j, freqB[j], a_, enr))
    aucs = np.array([r[2] for r in rows])
    print(f"  atoms (m={m}, k={k}): {alive} active on held-out tokens; {len(rows)} with frequency ≥ 0.5% analysed; "
          f"known features predict activity: AUC median {np.nanmedian(aucs):.2f}, "
          f"≥0.9: {(aucs >= 0.9).mean():.0%}, 0.7–0.9: {((aucs >= 0.7) & (aucs < 0.9)).mean():.0%}, <0.7: {(aucs < 0.7).mean():.0%}", flush=True)
    for j, f, a_, enr in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"    atom {j:5d}: freq {f:.3f}  AUC {a_:.2f}  enriched: " + ", ".join(f"{n} ×{e:.1f}" for e, n in enr))
    low = sorted([r for r in rows if r[1] >= 0.01], key=lambda r: r[2])[:6]
    print("  least explained frequent atoms: " + "; ".join(f"atom {j} (freq {f:.3f}, AUC {a_:.2f}; " + ", ".join(f"{n} ×{e:.1f}" for e, n in enr) + ")"
                                                      for j, f, a_, enr in low))
    if low:
        fig, axes = plt.subplots(len(low), 8, figsize=(13, 1.75 * len(low)), layout="constrained", squeeze=False)
        Cn = C.cpu().numpy()
        for r, (j, f, a_, _) in enumerate(low):
            score = Cn[:, j] * B
            for c_, tok in enumerate(np.argsort(-score)[:8]):
                i, tk = int(board[tok]), int(tok) % (G * G)
                rr, cc = divmod(tk, G)
                y0, x0 = max(0, rr * 12 - 18), max(0, cc * 12 - 18)
                ax = axes[r, c_]
                ax.imshow(sol[i].permute(1, 2, 0).numpy()[y0:y0 + 48, x0:x0 + 48])
                ax.add_patch(plt.Rectangle((cc * 12 - x0 - 0.5, rr * 12 - y0 - 0.5), 12, 12, fill=False, ec="#eb6834", lw=1.5))
                ax.set(xticks=[], yticks=[])
                if c_ == 0:
                    ax.set_ylabel(f"atom {j}\nAUC {a_:.2f}", fontsize=8)
        fig.suptitle(f"{name}: least explained frequent atoms — GT solution around the 8 most strongly activating held-out tokens (orange)", fontsize=9)
        fig.savefig(Path(out_dir) / f"sae_{name}_prototypes.png", dpi=110); plt.close(fig)


def main(plan_path, step1_path, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    reps = [(1024, 8), (2048, 16), (2048, 32)]
    bs = torch.load(plan_path, weights_only=False)
    maze = torch.cat([b["maze_img"] for b in bs]); sol = torch.cat([b["solution_img"] for b in bs]); cl = torch.cat([b["clean_img"] for b in bs])
    N = len(sol)
    named, K = known_features(maze, sol, cl)
    ts = sorted(bs[0]["clean"], reverse=True)
    X = torch.cat([torch.cat([b["clean"][t]["z_H"] for b in bs]).float().reshape(N * G * G, -1) for t in ts])
    board = np.tile(np.repeat(np.arange(N), G * G), len(ts))
    rep = lambda d: {n: np.tile(v, len(ts)) for n, v in d.items()}
    analyse("zH_final", X, board, rep(named), np.tile(K, (len(ts), 1)), sol, out_dir, reps, np.repeat(np.arange(len(ts)), N * G * G))
    del X
    d = torch.load(step1_path, weights_only=False)
    per = d["L_cycles"] + 1
    calls = torch.cat([b["calls"] for b in d["batches"]])                 # (N, 21, T, D) fp16
    sol1 = torch.cat([b["solution_img"] for b in d["batches"]])
    assert torch.equal(sol1, sol), "step1.pt boards differ from plan_states.pt boards"
    zl = [c for c in range(calls.shape[1]) if c % per != per - 1]
    X = calls[:, zl].float().permute(1, 0, 2, 3).reshape(len(zl) * N * G * G, -1)
    board = np.tile(np.repeat(np.arange(N), G * G), len(zl))
    analyse("zL_step1", X, board, {n: np.tile(v, len(zl)) for n, v in named.items()}, np.tile(K, (len(zl), 1)), sol, out_dir, reps, np.repeat(np.arange(len(zl)), N * G * G))


if __name__ == "__main__":
    main(*sys.argv[1:4])
