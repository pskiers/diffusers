"""TEMPORARY diagnostic (cluster only, no model): save TopK SAEs of the thinker states, name candidate atoms with
programmatic features, and link atoms to the thinker's own drops (AlphaZero-style contrast).

SAEs as interp_sae_z.py (template per stage x position removed on the even boards, m=1024, k=8):
  zH_final  (plan_states.pt, final z_H, t in {70,50,35,20,5})    zL_step1 (step1.pt, z_L, 18 updates of step 1)
Candidate features, per token (from the maze / solution images, maze cells of 144/8 px):
  wall_tip         the token contains an interior lattice point where exactly one wall segment ends
  tip_wrapped3/4   ... and at least 3 / all 4 of the cells around that point are on the solution
  marker_edge      the token contains a start/goal marker and lies on the grid border
For each feature: the atoms whose activation best detects it (held-out AUC) and how selective they are
(share of the atom's active tokens that have the feature).
Drops (zL_step1): for the thinker's strict drops (drop_concept.pt: board, token, last-path update a, drop update b),
change of each atom's activation from a to b, vs stable-off tokens over the same updates and vs dips (path tokens
p>0.8 → <0.5 → >0.8); atoms that switch off / on specifically at drops, and the atoms most aligned with the drop vector v.
Usage: python interp_sae_name.py plan_states.pt step1.pt drop_concept.pt out_dir
"""
import sys
from pathlib import Path

import numpy as np
import torch

import hv_features as hf
import interp_sae_z as sz

G = 12


def remove_template(X, board, group):
    A = torch.from_numpy(board % 2 == 0)
    pos = np.arange(len(X)) % (G * G)
    key = torch.from_numpy(group * (G * G) + pos)
    nk = int(key.max()) + 1
    sums = torch.zeros(nk, X.shape[1]).index_add_(0, key[A], X[A])
    cnt = torch.zeros(nk).index_add_(0, key[A], torch.ones(int(A.sum())))
    tmpl = sums / cnt.clamp(min=1)[:, None]
    Xr = X - tmpl[key]
    sc = Xr[A].pow(2).sum(-1).mean().sqrt()
    return Xr / sc, tmpl, float(sc)


def candidate_features(maze, sol, n_cells=8):
    """(N*144,) bool arrays per feature."""
    N = len(maze); cs = 144 / n_cells
    out = {k: np.zeros((N, G * G), bool) for k in ("wall_tip", "tip_wrapped3", "tip_wrapped4", "marker_edge")}
    for i in range(N):
        m = maze[i].float().numpy(); dark = m.mean(0) < hf.WALL_DARK
        s = sol[i].permute(1, 2, 0).numpy().astype(int)
        blue = (s[..., 2] > 120) & (s[..., 0] < 100)
        red = (s[..., 0] > 150) & (s[..., 1] < 110) & (s[..., 2] < 110)
        walls = {(r, c): hf.cell_wall_sides(dark, r, c, cs) for r in range(n_cells) for c in range(n_cells)}
        onpath = {}
        for r in range(n_cells):
            for c in range(n_cells):
                y0, x0 = int(r * cs + cs / 3), int(c * cs + cs / 3)
                onpath[(r, c)] = blue[y0:y0 + int(cs / 3), x0:x0 + int(cs / 3)].mean() > 0.1
        for r in range(1, n_cells):
            for c in range(1, n_cells):
                inc = [walls[(r - 1, c)][0], walls[(r, c)][0], walls[(r, c - 1)][2], walls[(r, c)][2]]   # up, down, left, right
                if sum(inc) != 1:
                    continue
                tk = int(r * cs // 12) * G + int(c * cs // 12)
                npth = sum(onpath[x] for x in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)))
                out["wall_tip"][i, tk] = True
                out["tip_wrapped3"][i, tk] |= npth >= 3
                out["tip_wrapped4"][i, tk] |= npth == 4
        for tk in range(G * G):
            tr, tc = divmod(tk, G)
            if red[tr * 12:(tr + 1) * 12, tc * 12:(tc + 1) * 12].sum() > 4 and (tr in (0, G - 1) or tc in (0, G - 1)):
                out["marker_edge"][i, tk] = True
    return {k: v.reshape(-1) for k, v in out.items()}


def fit(name, X, board, group, out_dir):
    Xn, tmpl, sc = remove_template(X, board, group)
    Xd = Xn.to(sz.DEV)
    A = torch.from_numpy(board % 2 == 0).to(sz.DEV)
    sae = sz.train_sae(Xd[A], 1024, 8)
    f = sz.fvu(sae, Xd[~A])
    with torch.no_grad():
        C = torch.cat([sae.encode(ch) for ch in Xd.split(16384)]).cpu()
    print(f"{name}: held-out FVU {f:.3f}", flush=True)
    torch.save({"sae": sae.state_dict(), "m": 1024, "k": 8, "template": tmpl, "scale": sc}, Path(out_dir) / f"sae_{name}.pt")
    return sae, C


def describe(name, C, board, named, K, cands):
    B = board % 2 == 1
    act = C > 0
    freq = act[torch.from_numpy(B)].float().mean(0).numpy()
    Kt = torch.from_numpy(K).to(sz.DEV); At, Bt = torch.from_numpy(~B).to(sz.DEV), torch.from_numpy(B).to(sz.DEV)
    info = {}
    for j in np.nonzero(freq >= 0.005)[0]:
        y = act[:, j].float().to(sz.DEV)
        info[int(j)] = {"freq": float(freq[j]), "auc_known": sz.auc(sz.fit_probe(Kt, y, At)(Kt)[Bt], y[Bt]),
                        "enriched": sorted(((named[n][act[:, j].numpy() & B].mean() / max(named[n][B].mean(), 1e-6), n)
                                            for n in named if named[n][B].mean() > 0.005), reverse=True)[:3]}
    print(f"\n[{name}] {len(info)} atoms with frequency ≥ 0.5% on held-out tokens")
    atoms = [j for j in info]
    best = {}
    for fname, fv in cands.items():
        fvB = torch.from_numpy(fv[B].astype(np.float32))
        res = []
        for j in atoms:
            a_ = sz.auc(C[torch.from_numpy(B), j], fvB)
            prec = float(fv[B][act[torch.from_numpy(B), j].numpy()].mean()) if act[torch.from_numpy(B), j].any() else 0.0
            res.append((a_, prec, j))
        res.sort(reverse=True)
        print(f"  {fname} (on {fv[B].mean():.1%} of held-out tokens): best atoms by AUC: " + "; ".join(
            f"atom {j} AUC {a_:.2f}, {prec:.0%} of its active tokens have it (freq {info[j]['freq']:.3f}, known-feature AUC {info[j]['auc_known']:.2f})"
            for a_, prec, j in res[:3]))
        best[fname] = [j for _, _, j in res[:3]]
    info["_best"] = best
    return info


def main(plan_path, step1_path, drop_path, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    bs = torch.load(plan_path, weights_only=False)
    maze = torch.cat([b["maze_img"] for b in bs]); sol = torch.cat([b["solution_img"] for b in bs]); cl = torch.cat([b["clean_img"] for b in bs])
    N = len(sol)
    named, K = sz.known_features(maze, sol, cl)
    cands = candidate_features(maze, sol)
    print("candidate features, share of tokens: " + ", ".join(f"{k} {v.mean():.2%}" for k, v in cands.items()), flush=True)
    ts = sorted(bs[0]["clean"], reverse=True)
    X = torch.cat([torch.cat([b["clean"][t]["z_H"] for b in bs]).float().reshape(N * G * G, -1) for t in ts])
    board = np.tile(np.repeat(np.arange(N), G * G), len(ts)); group = np.repeat(np.arange(len(ts)), N * G * G)
    sae_h, C_h = fit("zH_final", X, board, group, out_dir)
    tile = lambda d, r: {k: np.tile(v, r) for k, v in d.items()}
    info_h = describe("zH_final", C_h, board, tile(named, len(ts)), np.tile(K, (len(ts), 1)), tile(cands, len(ts)))
    # typical activation when active (for the push job)
    act_mean = torch.where(C_h > 0, C_h, torch.nan).nanmean(0).nan_to_num(0)
    torch.save({"info": info_h, "act_mean": act_mean}, Path(out_dir) / "atoms_zH_final.pt")
    del X

    d = torch.load(step1_path, weights_only=False)
    per = d["L_cycles"] + 1
    calls = torch.cat([b["calls"] for b in d["batches"]])
    zl = [c for c in range(calls.shape[1]) if c % per != per - 1]
    U = len(zl)
    X = calls[:, zl].float().permute(1, 0, 2, 3).reshape(U * N * G * G, -1)       # index = u*N*144 + i*144 + t
    board = np.tile(np.repeat(np.arange(N), G * G), U); group = np.repeat(np.arange(U), N * G * G)
    sae_l, C_l = fit("zL_step1", X, board, group, out_dir)
    info_l = describe("zL_step1", C_l, board, tile(named, U), np.tile(K, (U, 1)), tile(cands, U))

    # ---- drops: which atoms switch at the thinker's own rejections? ----
    dc = torch.load(drop_path, weights_only=False)
    P, gt, drops = dc["P"].numpy(), dc["gt"].numpy(), dc["drop_a_b"]
    ix = lambda i, t, u: u * N * G * G + i * G * G + t
    rng = np.random.default_rng(0)
    dD, dS, dP = [], [], []
    for i, t, a, b in drops:
        if i % 2 == 0:
            continue
        dD.append(C_l[ix(i, t, b)] - C_l[ix(i, t, a)])
        stable = np.nonzero((~gt[i]) & (P[i].max(-1) < 0.2))[0]
        for tc in rng.choice(stable, size=min(4, len(stable)), replace=False):
            dS.append(C_l[ix(i, tc, b)] - C_l[ix(i, tc, a)])
    for i in range(1, N, 2):
        for t in np.nonzero(gt[i])[0]:
            p = P[i, t]; hi = np.nonzero(p > 0.8)[0]
            for b in np.nonzero(p < 0.5)[0]:
                bef = [u for u in hi if u < b]; aft = [u for u in hi if u > b]
                if bef and aft:
                    dP.append(C_l[ix(i, t, b)] - C_l[ix(i, t, max(bef))]); break
    dD, dS, dP = torch.stack(dD), torch.stack(dS), torch.stack(dP)
    diff = dD.mean(0) - dS.mean(0)
    se = (dD.var(0) / len(dD) + dS.var(0) / len(dS)).sqrt() + 1e-9
    z = diff / se
    spec = dD.mean(0) - dP.mean(0)
    print(f"\n[drops, held-out boards] {len(dD)} drops, {len(dS)} stable-off controls, {len(dP)} dips")
    for lab, order in (("switch OFF at drops", torch.argsort(z)), ("switch ON at drops", torch.argsort(-z))):
        print(f"  atoms that {lab} (vs stable-off; z-score; difference to dips):")
        for j in order[:8].tolist():
            inf = info_l.get(j, {})
            print(f"    atom {j:4d}: Δ drops {float(dD[:, j].mean()):+.3f} vs stable {float(dS[:, j].mean()):+.3f} vs dips {float(dP[:, j].mean()):+.3f} "
                  f"(z {float(z[j]):+.1f}) | freq {inf.get('freq', float('nan')):.3f} known-feature AUC {inf.get('auc_known', float('nan')):.2f} | "
                  + ", ".join(f"{n} ×{e:.1f}" for e, n in inf.get("enriched", [])))
    # drop vector v (raw z_L space) vs atom decoder directions (template removal + uniform scaling keep directions)
    v = dc["v"].float()
    Dn = (sae_l.D / sae_l.D.norm(dim=0, keepdim=True)).T.detach().cpu()          # (m, 512)
    cosv = Dn @ (v / v.norm())
    top = torch.argsort(-cosv.abs())[:6].tolist()
    print("  atoms most aligned with the drop vector v: " + "; ".join(f"atom {j} cos {float(cosv[j]):+.2f}" for j in top))
    torch.save({"info": info_l, "drop_z": z, "drop_spec": spec, "cos_v": cosv}, Path(out_dir) / "atoms_zL_step1.pt")
    print("saved", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:5])
