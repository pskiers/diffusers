"""TEMPORARY (Claude): where does a model start pruning the lollipop (side loop) of a loop maze, and does it at all?

Reads <run_dir>/loops_first_<kind>.pt of interp_loops_first.py and regenerates the maze geometry with the same settings
(check_loops.variants, seed 0; the token sets E are checked against the saved ones). Lollipop = corridor + loop kept by
dead-end filling but not on the shortest path. A = solution cell(s) where the corridor leaves the solution, J = loop
cell(s) where the corridor enters the loop (= A if the loop touches the solution directly).
Per lollipop token (assigned to the cell containing the token's centre): drop = first z_L update after its last
"path" reading (logit > 0); within-board Spearman rho of the drop vs the cell's distance (open passages) from J / A.
Per board: loop length (cells on the cycle) and the share of lollipop tokens still "path" at the end.

python experiments/interp_loops_junction.py --run_dir <dir> --kind plain --n 11 --side_pick longest [--min_side 10]
"""
import argparse
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from check_loops import render_labels, variants


def bfs(nb, src, allowed):
    d = {s: 0 for s in src}; q = deque(src)
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v in allowed and v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--kind", default="plain")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--side_pick", default="random")
    ap.add_argument("--min_side", type=int, default=0); ap.add_argument("--num_boards", type=int, default=100)
    a = ap.parse_args()
    ds = AmazeDataset(dataset_path=f"data/amaze/maze/square/n{a.n}_test.parquet", split="test", image_size=144,
                      condition_field="m_original_img", target_field="sol_img", num_channels=3, include_metadata=True)
    rng, geo, i = random.Random(0), {}, 0                       # same order and random draws as interp_loops_first.py
    while i < len(ds) and len(geo) < a.num_boards:
        ms = mcl.build_maze_sample(ds[i].metadata)
        if ms is not None and ms.width == a.n:
            vs = variants(ms, rng, a.side_pick, a.min_side)
            if vs is not None:
                geo[i] = (ms, vs)
        i += 1
    cp = 144 / a.n
    tok_cell = lambda t: (int((12 * (t // 12) + 6) // cp), int((12 * (t % 12) + 6) // cp))

    d = torch.load(f"{a.run_dir}/loops_first_{a.kind}.pt", weights_only=False)
    by = {(x["board"], x["variant"]): x for x in d["items"]}
    rows, boards, mism = [], [], 0
    for b, (ms, vs) in geo.items():
        if (b, "side") not in by:
            continue
        x = by[(b, "side")]
        v = vs["side"]; nb = v["nb"]; alive = v["def_"]; sp = v["sp"]
        E_tok = (render_labels(ds, ms, v["img"], nb, alive) == 1) & (render_labels(ds, ms, v["img"], nb, sp) == 0)
        if not np.array_equal(x["sets"]["E_side"], E_tok):
            mism += 1; continue
        extra = alive - sp
        A = [u for u in sp if any(w in extra for w in nb[u])]
        core = set(extra) | set(A); changed = True                      # 2-core = the cycle
        while changed:
            changed = False
            for u in list(core):
                if sum(w in core for w in nb[u]) <= 1:
                    core.discard(u); changed = True
        loop = core & extra
        J = [u for u in loop if any(w in extra - loop or w in A for w in nb[u])] or A
        loop_len = len(core)
        dJ, dA = bfs(nb, J, alive), bfs(nb, A, alive)
        L = x["logit"].float()
        E = torch.tensor(x["sets"]["E_side"])
        boards.append(dict(b=b, loop_len=loop_len, corridor=len(extra - loop), n_tok=int(E.sum()),
                           left=float((L[-1, E] > 0).float().mean()) if E.any() else np.nan,
                           ever=float((L[:, E] > 1.39).any(0).float().mean()) if E.any() else np.nan))
        for t in np.nonzero(x["sets"]["E_side"])[0]:
            cell = tok_cell(int(t))
            if cell not in extra:
                continue
            above = (L[:, t] > 0).numpy()
            if not above.any():
                continue
            drop = int(np.nonzero(above)[0].max()) + 1
            rows.append(dict(b=b, drop=drop if drop < L.shape[0] else np.nan, dJ=dJ.get(cell, np.nan), dA=dA.get(cell, np.nan),
                             part="loop" if cell in loop else "corridor", isJ=cell in J, loop_len=loop_len))
    R, B = pd.DataFrame(rows), pd.DataFrame(boards)
    print(f"######## {a.kind}, n{a.n}, side_pick={a.side_pick}, min_side={a.min_side}: {len(B)} boards (geometry mismatches: {mism})")
    print(f"loop length in cells: median {B.loop_len.median():.0f}, range {B.loop_len.min()}-{B.loop_len.max()}; corridor length median {B.corridor.median():.0f}")
    bins = [0, 5, 9, 13, 17, 100]
    B["len_bin"] = pd.cut(B.loop_len, bins, labels=["4", "6-8", "10-12", "14-16", ">16"])
    g = B.groupby("len_bin", observed=True).agg(n=("b", "size"), ever=("ever", "mean"), left=("left", "mean"))
    print("by loop length: lollipop tokens ever confidently path / still path at the end")
    for k, r in g.iterrows():
        print(f"   loop {k:>5s} cells: n={int(r.n):3d}  ever {r.ever:4.0%}  left at the end {r.left:4.0%}")
    D = R[R["drop"].notna()]
    print(f"dropped lollipop tokens: {len(D)} of {len(R)} ever read as path")
    if len(D):
        print("median drop update: " + ", ".join(f"{p} {D[D.part == p]['drop'].median():.0f} (n={int((D.part == p).sum())})" for p in ("loop", "corridor"))
              + f" | J {D[D.isJ]['drop'].median():.0f}, other loop cells {D[(D.part == 'loop') & ~D.isJ]['drop'].median():.0f}")
        for col, name in (("dJ", "loop entry J"), ("dA", "attachment A")):
            ok = D[col].notna()
            wb = [spearmanr(gg[col], gg["drop"]).correlation for _, gg in D[ok].groupby("b")
                  if gg[col].nunique() > 1 and gg["drop"].nunique() > 1]
            prof = D[ok].groupby(col)["drop"].median()
            print(f"drop vs cells from {name:13s}: within-board rho {np.nanmean(wb):+.2f} (n={len(wb)}) | median drop by distance: "
                  + " ".join(f"{int(k)}:{v:.0f}" for k, v in prof.items() if k <= 14))
    B.to_csv(f"{a.run_dir}/junction_boards_{a.kind}.csv", index=False)


if __name__ == "__main__":
    main()
