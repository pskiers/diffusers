"""TEMPORARY analysis (cluster, no model): E4 — successes vs failures of the same maze at budget (1, 2, 6) = 14 calls.

Data: interp_e4_record.py (98 boards x 8 seeds: drawn cells; thinker state after each of the 14 calls at t=70).
Boards rebuilt with the same filter (maze graph for geometry). Mixed boards = solved by some seeds, not by others.
(1) What is missing in failures: where the missing cells lie on the solution (distance to the drawn frontier / to the goal),
    and the side branches touching them (number, longest branch length) vs solution cells that are never missing.
(2) When failures diverge: per call, decoded path probability (per-call probe trained on successful runs of even boards)
    at the tokens of the cells that end up missing, in failed vs successful runs of the same board.
(3) AlphaZero-style contrast: per call, success − failure state at those tokens (same board, same token, same call);
    direction from even boards (mean difference), held-out AUC on odd boards, and its cosine with the path direction.
"""
import random
import sys
from collections import deque

import numpy as np
import torch

import amaze_interpretability as ai
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset

G = 12


def boards(parquet):
    ds = AmazeDataset(dataset_path=parquet, split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    out = []
    for start in range(0, min(100, len(ds)), 16):
        base_seed = start * 977
        raw = [ds[i] for i in range(start, min(start + 16, 100))]
        pairs = [(s, ms) for s, ms in ((s, mcl.build_maze_sample(s.metadata)) for s in raw) if ms is not None]
        for i, (s, ms) in enumerate(pairs):
            rng = random.Random(base_seed + i * 131)
            board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
            _, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)
            if wrong:
                out.append((ms, set(shown) | set(gt_walked)))
    return out


def nbrs(ms):
    nb = {c: [] for c in ms.id2rc}
    for cid, (r, c) in ms.id2rc.items():
        for r2, c2 in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            n = ms.rc2id.get((r2, c2))
            if n is not None and mcl._is_open(ms.morig, r, c, r2, c2, ms.cell_px):
                nb[cid].append(n)
    return nb


def branch_sizes(ms, nb, cell):
    """sizes of the off-path branches attached to a solution cell."""
    path = set(ms.path); sizes = []
    for n in nb[cell]:
        if n in path:
            continue
        seen = {n}; q = deque([n])
        while q:
            x = q.popleft()
            for y in nb[x]:
                if y not in seen and y not in path:
                    seen.add(y); q.append(y)
        sizes.append(len(seen))
    return sizes


def cell_tokens(ms, cid):
    cs = 144 / ms.height; r, c = ms.id2rc[cid]
    y0, y1, x0, x1 = r * cs, (r + 1) * cs, c * cs, (c + 1) * cs
    return [tr * G + tc for tr in range(G) for tc in range(G)
            if min(y1, (tr + 1) * 12) - max(y0, tr * 12) >= 6 and min(x1, (tc + 1) * 12) - max(x0, tc * 12) >= 6]


def fit_probe(X, y, train, dev):
    X, y = X.to(dev), y.to(dev)
    m, s = X[train].mean(0), X[train].std(0) + 1e-6
    w = torch.zeros(X.shape[1] + 1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=150)
    Xa = (X[train] - m) / s
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w[:-1] + w[-1], y[train]) + 1e-3 * w[:-1].pow(2).sum()
        l.backward(); return l
    opt.step(cl)
    w = w.detach()
    return lambda Z: torch.sigmoid(((Z.to(dev) - m) / s) @ w[:-1] + w[-1]).cpu()


def main(rec_dir, parquet):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = torch.load(f"{rec_dir}/e4_runs.pt", weights_only=False)
    S = torch.load(f"{rec_dir}/e4_states_t70.pt", weights_only=False)                 # (N, SEEDS, 14, T, D) fp16
    bl = boards(parquet)
    N, K, C = S.shape[:3]
    assert N == len(bl), (N, len(bl))
    rows = R["rows"]
    solved = np.zeros((N, K), bool); missing = {}
    for r in rows:
        solved[r["board"], r["seed"]] = r["solved"]; missing[(r["board"], r["seed"])] = set(r["missing"])
    mixed = [i for i in range(N) if 0 < solved[i].sum() < K]
    print(f"{N} boards x {K} seeds, solved {solved.mean():.2f}; mixed boards {len(mixed)}", flush=True)

    # (1) geometry of the missing cells
    feats = {"miss": [], "never": []}
    for i in range(N):
        ms, vis = bl[i]; nb = nbrs(ms)
        rem = [c for c in ms.path if c not in vis]
        if not rem:
            continue
        idx = {c: k for k, c in enumerate(rem)}
        ever = set().union(*[missing[(i, k)] for k in range(K)])
        for c in rem[:-1]:
            bs = branch_sizes(ms, nb, c)
            f = (idx[c], len(rem) - 1 - idx[c], len(bs), max(bs) if bs else 0)
            feats["miss" if c in ever else "never"].append(f)
    for k, v in feats.items():
        v = np.array(v)
        print(f"  solution cells {k:5s} (n={len(v)}): distance from the drawn frontier {v[:, 0].mean():.1f}, to the goal {v[:, 1].mean():.1f}, "
              f"side branches attached {v[:, 2].mean():.2f}, longest attached branch {v[:, 3].mean():.1f} cells", flush=True)
    per_seed_miss = [len(missing[(i, k)]) for i in range(N) for k in range(K) if not solved[i, k]]
    print(f"  failed runs: missing cells per run median {np.median(per_seed_miss):.0f}, mean {np.mean(per_seed_miss):.2f}")

    # (2)+(3) at the tokens of the cells that are missing in some seeds of a mixed board
    gt = np.zeros((N, G * G), bool)
    for i, (ms, vis) in enumerate(bl):
        for c in ms.path:
            for t in cell_tokens(ms, c):
                gt[i, t] = True
    y = torch.from_numpy(np.repeat(gt[:, None], K, 1).reshape(-1).astype(np.float32))
    board = np.repeat(np.arange(N), K * G * G); seedix = np.tile(np.repeat(np.arange(K), G * G), N)
    succ_row = torch.from_numpy(np.repeat(solved.reshape(-1), G * G))
    trainA = torch.from_numpy(board % 2 == 0) & succ_row
    pairs = []                                                               # (board, token, success seeds, failed seeds where it is missing)
    for i in mixed:
        ms, vis = bl[i]
        ever = set().union(*[missing[(i, k)] for k in range(K) if not solved[i, k]])
        for c in ever:
            fs = [k for k in range(K) if not solved[i, k] and c in missing[(i, k)]]
            ss = [k for k in range(K) if solved[i, k]]
            for t in cell_tokens(ms, c):
                if gt[i, t]:
                    pairs.append((i, t, ss, fs))
    print(f"\n(2) {len(pairs)} (board, token) pairs on {len(mixed)} mixed boards; decoded path probability at the tokens of the cells missing "
          f"in the failed runs (per-call probe trained on successful runs of even boards):", flush=True)
    print("   call | successful runs  failed runs | held-out (odd boards): successful  failed")
    dirs, cos_path = {}, {}
    for c in range(C):
        X = S[:, :, c].float().reshape(-1, S.shape[-1])
        p = fit_probe(X, y, trainA, dev)(X).reshape(N, K, G * G).numpy()
        sp = [p[i, ss, t].mean() for i, t, ss, fs in pairs]; fp = [p[i, fs, t].mean() for i, t, ss, fs in pairs]
        spB = [p[i, ss, t].mean() for i, t, ss, fs in pairs if i % 2]; fpB = [p[i, fs, t].mean() for i, t, ss, fs in pairs if i % 2]
        print(f"   {c + 1:4d} |      {np.mean(sp):.3f}           {np.mean(fp):.3f}   |      {np.mean(spB):.3f}      {np.mean(fpB):.3f}", flush=True)
        # (3) success − failure direction at these tokens (even boards), and the path direction (successful runs, even boards)
        Z = S[:, :, c].float()
        dA = torch.stack([Z[i, ss, t].mean(0) - Z[i, fs, t].mean(0) for i, t, ss, fs in pairs if i % 2 == 0]).mean(0)
        on = torch.from_numpy(gt); succ = torch.from_numpy(solved)
        zs = Z[torch.from_numpy(np.arange(N) % 2 == 0)]
        mpath = torch.stack([zs[j][succ[torch.from_numpy(np.arange(N) % 2 == 0)][j]][:, on[torch.from_numpy(np.arange(N) % 2 == 0)][j]].reshape(-1, Z.shape[-1]).mean(0)
                             for j in range(len(zs)) if succ[torch.from_numpy(np.arange(N) % 2 == 0)][j].any()]).mean(0)
        moff = torch.stack([zs[j][succ[torch.from_numpy(np.arange(N) % 2 == 0)][j]][:, ~on[torch.from_numpy(np.arange(N) % 2 == 0)][j]].reshape(-1, Z.shape[-1]).mean(0)
                            for j in range(len(zs)) if succ[torch.from_numpy(np.arange(N) % 2 == 0)][j].any()]).mean(0)
        dpath = mpath - moff
        dirs[c] = dA; cos_path[c] = float(dA @ dpath / dA.norm() / dpath.norm())
    print("\n(3) success − failure direction at the missing-cell tokens (from even boards): held-out AUC (odd boards: a successful run's"
          " state vs a failed run's state at the same token), size relative to the path−off gap, cosine with the path direction")
    for c in range(C):
        u = dirs[c] / dirs[c].norm()
        Z = S[:, :, c].float()
        sc_s = [float(Z[i, k, t] @ u) for i, t, ss, fs in pairs if i % 2 for k in ss]
        sc_f = [float(Z[i, k, t] @ u) for i, t, ss, fs in pairs if i % 2 for k in fs]
        # paired AUC: per (board, token), successful mean vs failed mean
        ps = np.array([np.mean([float(Z[i, k, t] @ u) for k in ss]) for i, t, ss, fs in pairs if i % 2])
        pf = np.array([np.mean([float(Z[i, k, t] @ u) for k in fs]) for i, t, ss, fs in pairs if i % 2])
        print(f"   call {c + 1:2d}: paired held-out success > failure {np.mean(ps > pf):.2f}, unpaired AUC "
              f"{np.mean([a > b for a in sc_s[:400] for b in sc_f[:400]]):.2f} | cos with path direction {cos_path[c]:+.2f}", flush=True)
    torch.save({"dirs": {c: v for c, v in dirs.items()}, "cos_path": cos_path, "pairs": [(i, t) for i, t, _, _ in pairs], "mixed": mixed},
               f"{rec_dir}/e4_analysis.pt")
    print("saved", f"{rec_dir}/e4_analysis.pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
