"""TEMPORARY diagnostic (Claude, amaze interpretability): follow-up of interp_sae_circuits.py (same SAEs, square 8x8).

Same regime: first denoising step t=95 from pure noise; same test boards and seeds, so the recorded states equal the
ones the SAEs were evaluated on (checked: share of identical top-k codes).
(1) Read-out = logistic probe on the 512-dim FINAL z_H (fit on the 768 train boards; was 0.998 before), linear in z_H.
    Contribution of an atom (both streams, same sign: >0 = pushes the read-out towards the correct plan):
      z_H atoms: exact, -(score change when the atom is removed);
      z_L atoms: removed from the z_L that feeds a z_H update, that update recomputed (one step).
    Pruning time per off-solution token = the z_H call from which its score stays < 0; atoms that fire just BEFORE a
    token is pruned vs on off tokens that are pruned later (same call, same token type).
(2) Input attribution for the selected atoms at their onset call (and z_H 410 also at call 21): gradient x (input -
    mean input) of the atom's pre-activation on up to 64 firing tokens, as maps over the offset (source token - firing
    token) for each input (maze embedding, own previous state, the other stream), and per upstream SAE atom (the other
    stream / own previous state decomposed by its SAE; attribution patching = linear estimate).
(3) Causal tests on the painter's drawing (full generation, 20 steps, first N_CAUSAL parseable test boards). Edits act in
    the conditional pass at given block calls at every denoising step; atom activations are encoded with a per-t
    template (mean over these boards in the unedited run) and the t=95 scale.
"""
import dataclasses

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_sae_circuits as sc
import interp_size_ablation as sa
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG, T, WIN = 12, 144, 144, 4
N_CAUSAL, WALL_DARK, N_PAIRS = 40, 80, 64
FULL, B21, B14 = (16, 3, 6), (1, 3, 6), (1, 2, 6)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    ts = model.scheduler.timesteps
    t0 = ts[0]
    predictor = model.sampling_pipeline.predictor
    inner = model.thinker.inner
    per, D = inner.config.L_cycles + 1, inner.config.hidden_size
    kind = lambda c: "z_H" if c % per == per - 1 else "z_L"

    prev = torch.load(str(ic.sae), weights_only=False)
    REC = prev["REC"]; pos = {c: j for j, c in enumerate(REC)}
    streams = {"z_L": [c for c in REC if kind(c) == "z_L"], "z_H": [c for c in REC if kind(c) == "z_H"]}
    saes, norm = {}, {}
    for s in streams:
        sd = prev[f"sae_{s}"]
        m_ = sc.TopKSAE(D, sd["W_enc"].shape[1], sc.K).to(device); m_.load_state_dict(sd); m_.eval()
        for p_ in m_.parameters():
            p_.requires_grad_(False)
        saes[s] = m_
        mu, scl = prev[f"norm_{s}"]; norm[s] = (mu.float().to(device), scl.float().to(device))
    M = saes["z_L"].W_dec.shape[0]

    # ---------------------------------------------------------------- recording (same as interp_sae_circuits.py)
    st = {"mode": None, "c": 0}

    def pre_hook(module, args, kwargs):
        if st["mode"] is not None:
            if st["c"] == 0:
                st["L0"], st["inj0"], st["cos_sin"] = args[0][:1].clone(), args[1].clone(), kwargs.get("cos_sin")
            if st["c"] == per - 1:
                st["H0"] = args[0][:1].clone()

    def fwd_hook(module, args, kwargs, output):
        if st["mode"] is None:
            return None
        c = st["c"]; st["c"] += 1
        if st["mode"] == "rec":
            if c in st["keep"]:
                st["buf"][:, st["keep"][c]] = output.detach().cpu()
            return None
        if st["mode"] == "edit":
            if c in st.get("mu_calls", ()):
                st["mu_acc"].setdefault((st["t"], c), output.float().mean(0))
            fn = st["edits"].get(c)
            return fn(c, output) if fn is not None else None
    inner.L_level.register_forward_pre_hook(pre_hook, with_kwargs=True)
    inner.L_level.register_forward_hook(fwd_hook, with_kwargs=True)

    vd = cfg.data.get("val_dataset", {})
    mk = lambda p, split, meta: AmazeDataset(dataset_path=p, split=split, image_size=IMG, num_channels=3, include_metadata=meta,
                                             condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"))

    def record(ds, n_max, seed, calls):
        keep = {c: j for j, c in enumerate(calls)}
        n = min(n_max, len(ds)); S, gts, embs = [], [], []
        for i0 in range(0, n, sc.BATCH):
            batch = AmazeDataset.collate_fn([ds[i] for i in range(i0, min(i0 + sc.BATCH, n))])
            cond = model._batch_to_sample(batch, device); B = batch.images.shape[0]
            x = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(seed + i0), device=device)
            s = dataclasses.replace(cond, x_noisy=x, timesteps=t0.expand(B).to(device))
            st.update(mode="rec", c=0, keep=keep, buf=torch.empty((B, len(calls), T, D), dtype=inner.forward_dtype))
            with torch.no_grad():
                model.forward_with_carry(s)
            st["mode"] = None
            S.append(st["buf"]); embs.append((st["inj0"] - st["H0"]).cpu())
            sol = (batch.images.float().clamp(0, 1) * 255).byte().cpu(); maze = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
            gts.append(np.stack([pa.token_labels(sol[i], maze[i])[0] != "off" for i in range(B)]))
        return torch.cat(S), np.concatenate(gts), torch.cat(embs)
    S_tr, gt_tr, _ = record(mk(str(ic.train_subset), "train", False), sc.N_TRAIN, 1000, [335])
    S_te, gt_te, emb_te = record(mk(str(ic.data_parquet), "test", False), 100, 5000, REC)
    Nte = len(S_te)
    L0, H0, cos_sin = st["L0"], st["H0"], st["cos_sin"]
    assert np.array_equal(gt_te, prev["gt"]), "test labels differ from the first run"

    def encode(s, c, z, mu=None):
        mu_, scl = norm[s]; j = streams[s].index(c)
        return saes[s].encode((z.float() - (mu_[j] if mu is None else mu)) / scl[j])
    idx0, _ = prev["codes_z_H"]
    with torch.no_grad():
        same = float((encode("z_H", 335, S_te[:, pos[335]].to(device)).topk(sc.K, -1).indices.sort(-1).values.cpu()
                      == idx0[:, -1].long().sort(-1).values).all(-1).float().mean())
    print(f"t={int(t0)}; {Nte} test boards; tokens with identical z_H top-{sc.K} codes vs the first run (final call): {same:.3f}", flush=True)

    # ---------------------------------------------------------------- (1) read-out on z_H, contributions, pruning
    zf = S_tr[:, 0].float().to(device).reshape(-1, D)
    probe = sa.fit_probe(zf, torch.from_numpy(gt_tr.reshape(-1).astype(np.float32)).to(device), torch.ones(len(zf), dtype=torch.bool, device=device))
    c0 = probe(torch.zeros(1, D, device=device)); u = (probe(torch.eye(D, device=device)) - c0).detach(); c0 = c0.detach()
    del zf, S_tr
    score = lambda z: z.float() @ u + c0
    Hc = streams["z_H"]
    sc_te = torch.stack([score(S_te[:, pos[c]].to(device)).cpu() for c in Hc], 1)      # (N, 10, T)
    gt = torch.from_numpy(gt_te); g = (gt.float() * 2 - 1)
    accs = [float(((sc_te[:, k] > 0) == gt).float().mean()) for k in range(len(Hc))]
    print(f"\n(1) read-out = probe on final z_H: test acc {accs[-1]:.4f}, boards exact {float(((sc_te[:, -1] > 0) == gt).all(1).float().mean()):.2f}; "
          "per z_H call: " + " ".join(f"{c + 1}:{a:.3f}" for c, a in zip(Hc, accs)))
    neg = sc_te < 0                                                                    # (N, 10, T)
    stays = torch.flip(torch.cumprod(torch.flip(neg, [1]).int(), 1), [1]).bool()      # off from this call on
    prune_k = torch.where(stays.any(1), stays.int().argmax(1), torch.full(gt.shape, -1))  # first z_H call index from which off
    off_ok = (~gt) & (prune_k >= 0)
    print("    pruning time of off-solution tokens (z_H call from which the read-out stays 'off'): "
          + ", ".join(f"call {Hc[k] + 1}: {int(((prune_k == k) & off_ok).sum())}" for k in range(len(Hc)) if int(((prune_k == k) & off_ok).sum()))
          + f"; never: {int(((~gt) & (prune_k < 0)).sum())}")

    def parts(c, ix=None):
        sel = (lambda z: z) if ix is None else (lambda z: z[ix])
        stt = lambda cc: sel(S_te[:, pos[cc]]).to(device)
        n = Nte if ix is None else len(ix)
        L0b, H0b, emb = L0.expand(n, -1, -1), H0.expand(n, -1, -1), sel(emb_te).to(device)
        if kind(c) == "z_L":
            pv = c - 1 if (c - 1) % per != per - 1 else c - 2
            return {"hidden": L0b if c == 0 else stt(pv), "zH": H0b if c < per else stt((c // per) * per - 1), "emb": emb}
        return {"hidden": H0b if c == per - 1 else stt(c - per), "zL": stt(c - 1)}

    def src_of(c, key):
        """(stream, call) whose state is input `key` of block call c, or None for constants / the maze embedding."""
        if key == "emb":
            return None
        if kind(c) == "z_L":
            if key == "hidden":
                return None if c == 0 else ("z_L", c - 1 if (c - 1) % per != per - 1 else c - 2)
            return None if c < per else ("z_H", (c // per) * per - 1)
        if key == "hidden":
            return None if c == per - 1 else ("z_H", c - per)
        return ("z_L", c - 1)

    def block(p):
        inj = p["zL"] if "zL" in p else p["zH"] + p["emb"]
        return inner.L_level(p["hidden"], inj, cos_sin=cos_sin)

    contrib = {}
    with torch.no_grad():
        for k, c in enumerate(Hc):                                                     # z_H atoms, exact
            f = encode("z_H", c, S_te[:, pos[c]].to(device))
            du = norm["z_H"][1][streams["z_H"].index(c)] * (saes["z_H"].W_dec @ u)
            part = f * du                                                              # (N, T, M) score written by the atom
            contrib[("z_H", c)] = {"total": (part * g.to(device)[..., None]).sum((0, 1)).cpu() / Nte,
                                   "on_sol": part[gt.to(device)].sum(0).cpu() / Nte, "on_off": part[~gt.to(device)].sum(0).cpu() / Nte,
                                   "writes": du.cpu(), "freq": (f > 0).float().mean((0, 1)).cpu()}
        for c in [x for x in Hc if x <= 62]:                                           # z_L atoms, one step
            p = parts(c); fL = encode("z_L", c - 1, p["zL"]); s0 = score(block(p))
            tot = torch.zeros(M); on_sol = torch.zeros(M); on_off = torch.zeros(M)
            for a in torch.nonzero((fL > 0).float().mean((0, 1)) >= sc.MIN_FREQ).flatten().tolist():
                p2 = dict(p); p2["zL"] = p["zL"] - (fL[..., a:a + 1] * norm["z_L"][1][streams["z_L"].index(c - 1)] * saes["z_L"].W_dec[a]).to(p["zL"].dtype)
                ds = s0 - score(block(p2))                                             # what the atom added
                tot[a] = float((ds * g.to(device)).sum() / Nte); on_sol[a] = float(ds[gt.to(device)].sum() / Nte); on_off[a] = float(ds[~gt.to(device)].sum() / Nte)
            contrib[("z_L", c - 1)] = {"total": tot, "on_sol": on_sol, "on_off": on_off, "freq": (fL > 0).float().mean((0, 1)).cpu()}
            print(f"    z_L one-step contributions at the z_H update of call {c + 1} done", flush=True)
    for key in (("z_H", 335), ("z_H", 13)):
        r = contrib[key]
        print(f"  z_H atoms at call {key[1] + 1}: largest contributions to the correct plan: " + ", ".join(
            f"{a} ({float(r['total'][a]):+.1f}; sol {float(r['on_sol'][a]):+.1f} / off {float(r['on_off'][a]):+.1f})" for a in r["total"].topk(6).indices.tolist()))
    for key in (("z_L", 5), ("z_L", 12)):
        r = contrib[key]
        print(f"  z_L atoms feeding the z_H update of call {key[1] + 2}: " + ", ".join(
            f"{a} ({float(r['total'][a]):+.1f}; sol {float(r['on_sol'][a]):+.1f} / off {float(r['on_off'][a]):+.1f})" for a in r["total"].topk(6).indices.tolist())
            + " | most harmful: " + ", ".join(f"{a} ({float(r['total'][a]):+.1f})" for a in (-r["total"]).topk(3).indices.tolist()))

    # pre-pruning signal: tokens pruned at z_H call k vs off tokens pruned later, at the states just before call k
    prune_sig = {}
    idxH, valH = prev["codes_z_H"]; idxL, valL = prev["codes_z_L"]
    def dense(idx, val, j):
        out = torch.zeros(idx.shape[0], T, M)
        return out.scatter_add_(-1, idx[:, j].long(), val[:, j].float())
    for k in (1, 2):
        now = off_ok & (prune_k == k); later = off_ok & (prune_k > k)
        if now.sum() < 20 or later.sum() < 20:
            continue
        for s, idx, val, c in (("z_H", idxH, valH, Hc[k - 1]), ("z_L", idxL, valL, Hc[k] - 1)):
            A = dense(idx, val, streams[s].index(c))
            d = A[now].mean(0) - A[later].mean(0)
            prune_sig[(s, k)] = {"diff": d, "now": A[now].mean(0), "later": A[later].mean(0), "n": (int(now.sum()), int(later.sum())), "call": c}
            print(f"  about to be pruned at call {Hc[k] + 1} ({int(now.sum())} tokens) vs off tokens pruned later ({int(later.sum())}), {s} at call {c + 1}: "
                  + ", ".join(f"{a} ({float(A[now].mean(0)[a]):.2f} vs {float(A[later].mean(0)[a]):.2f})" for a in d.topk(6).indices.tolist()))

    # ---------------------------------------------------------------- (2) input attribution
    comp_prev = prev["components"]
    targets = [(s, a, v["onset"]) for (s, a), v in comp_prev.items()] + [("z_H", 410, 20), ("z_H", 408, 13), ("z_L", 1733, 12)]
    attrib = {}
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"); rr, cc_ = yy.flatten(), xx.flatten()
    base_mean = {}
    for s, a, c in sorted(set(targets), key=lambda x: x[2]):
        with torch.no_grad():
            fa = encode(s, c, block(parts(c)))[..., a]                                 # (N, T)
        cand = torch.nonzero(fa > 0)
        if len(cand) == 0:
            continue
        cand = cand[fa[cand[:, 0], cand[:, 1]].topk(min(N_PAIRS, len(cand))).indices]
        ix, tok = cand[:, 0].cpu(), cand[:, 1].to(device)
        p = {k_: v.detach().float().clone() for k_, v in parts(c, ix).items()}
        if c not in base_mean:
            base_mean[c] = {k_: v.float().mean(0, keepdim=True) for k_, v in parts(c).items()}
        for v in p.values():
            v.requires_grad_(True)
        with torch.enable_grad():
            out = block(p)
            mu_, scl = norm[s]; j = streams[s].index(c)
            x = (out.float() - mu_[j]) / scl[j]
            pre = ((x - saes[s].b_dec) @ saes[s].W_enc[:, a] + saes[s].b_enc[a])     # (n, T)
            pre[torch.arange(len(ix)), tok].sum().backward()
        res = {"n": len(ix), "inputs": {}, "upstream": {}}
        dr = (rr.to(device)[None] - rr.to(device)[tok][:, None]); dc = (cc_.to(device)[None] - cc_.to(device)[tok][:, None])   # (n, T)
        inwin = (dr.abs() <= WIN) & (dc.abs() <= WIN)
        tot_abs = 0.0
        for key, v in p.items():
            contrib_tok = (v.grad * (v.detach() - base_mean[c][key])).sum(-1)          # (n, T)
            mp = torch.zeros(2 * WIN + 1, 2 * WIN + 1, device=device)
            mp.index_put_(((dr + WIN)[inwin], (dc + WIN)[inwin]), contrib_tok[inwin], accumulate=True)
            res["inputs"][key] = {"map": (mp / len(ix)).cpu(), "total": float(contrib_tok.sum() / len(ix)), "outside": float(contrib_tok[~inwin].sum() / len(ix))}
            tot_abs += float(contrib_tok.abs().sum() / len(ix))
            src = src_of(c, key)
            if src is not None and src[1] in streams[src[0]]:
                s2, c2 = src
                with torch.no_grad():
                    f2 = encode(s2, c2, v.detach())                                    # (n, T, M)
                    w = norm[s2][1][streams[s2].index(c2)] * (v.grad @ saes[s2].W_dec.T)  # (n, T, M) gradient . raw direction
                    at = f2 * w
                    tot_up = at.sum((0, 1)) / len(ix)
                    top = tot_up.abs().topk(8).indices
                    ups = []
                    for b_ in top.tolist():
                        q = at[..., b_]; mq = torch.zeros(2 * WIN + 1, 2 * WIN + 1, device=device)
                        mq.index_put_(((dr + WIN)[inwin], (dc + WIN)[inwin]), q[inwin], accumulate=True)
                        ups.append({"atom": b_, "total": float(tot_up[b_]), "map": (mq / len(ix)).cpu()})
                res["upstream"][key] = {"stream": s2, "call": c2, "atoms": ups, "sae_total": float(tot_up.sum())}
        res["abs_total"] = tot_abs
        attrib[(s, a, c)] = res
        ins = " | ".join(f"{k_} {res['inputs'][k_]['total']:+.2f}" for k_ in res["inputs"])
        ups = " ; ".join(f"{k_}: " + ", ".join(f"{u_['atom']}({u_['total']:+.2f})" for u_ in v["atoms"][:3]) for k_, v in res["upstream"].items())
        print(f"  (2) {s} atom {a:4d} at call {c + 1} ({len(ix)} firing tokens): inputs {ins} || top upstream {ups}", flush=True)

    # ---------------------------------------------------------------- (3) causal tests on the painter
    ds_m = mk(str(ic.data_parquet), "test", True)
    kept, mss = [], []
    for i in range(len(ds_m)):
        ms = mcl.build_maze_sample(ds_m[i].metadata)
        if ms is not None:
            kept.append(i); mss.append(ms)
        if len(kept) == N_CAUSAL:
            break
    batch = AmazeDataset.collate_fn([ds_m[i] for i in kept]); B = len(kept)
    cond = model._batch_to_sample(batch, device)
    maze_u8 = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
    gtk = torch.from_numpy(gt_te[kept]).to(device)                                    # solution tokens of these boards
    x0 = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(77), device=device)

    def matched(s, a, calls, exclude):
        """atom of stream s with the closest frequency and mean activation over `calls` (test codes), small contribution."""
        idx, val = prev[f"codes_{s}"]
        js = [streams[s].index(c) for c in calls]
        fq = prev["stats"][s]["freq"][js].mean(0); ma = prev["stats"][s]["mean_act"][js].mean(0)
        ck = ("z_H", calls[-1]) if s == "z_H" else ("z_L", calls[-1])
        ctot = contrib[ck]["total"].abs() if ck in contrib else torch.zeros(M)
        dist = (fq / (fq[a] + 1e-9)).log().abs() + (ma / (ma[a] + 1e-9)).log().abs() + 0.5 * ctot / (ctot.max() + 1e-9)
        dist[list(exclude)] = float("inf"); dist[fq < 1e-4] = float("inf")
        return int(dist.argmin())

    sel_all = {a for (s_, a) in comp_prev} | {410, 408, 1733}
    ctrl_100, ctrl_1122 = matched("z_H", 100, [335], sel_all), matched("z_H", 1122, [335], sel_all)
    ctrl_410 = matched("z_H", 410, [13, 20], sel_all)
    ctrl_L = [matched("z_L", a, [0, 1, 2, 3, 4, 5], sel_all | {1312, 1613, 873}) for a in (1312, 1613, 873)]
    act410 = prev["codes_z_H"][1][:, Hc.index(13)].float(); act410 = act410[prev["codes_z_H"][0][:, Hc.index(13)].long() == 410]
    alpha410 = float(torch.quantile(act410, 0.9)) if len(act410) else 1.0
    gen_r = torch.Generator(device=device).manual_seed(3); rnd = torch.randn(D, generator=gen_r, device=device); rnd = rnd / rnd.norm()
    print(f"\n(3) causal tests on {B} boards; controls: z_H {ctrl_100} (for 100), {ctrl_1122} (for 1122), {ctrl_410} (for 410), "
          f"z_L {ctrl_L} (for 1312/1613/873); clamp value for 410 = {alpha410:.2f} (90th percentile when active at call 14)", flush=True)

    mu_t = {}
    def ablate(s, atoms, budget):
        def fn(c, out):
            mu = mu_t[(budget, st["t"], c)]
            f = encode(s, c, out, mu)
            dvec = norm[s][1][streams[s].index(c)] * (f[..., atoms] @ saes[s].W_dec[atoms])
            st["log"].append((st["t"], c, float(f[..., atoms].sum() / B)))
            return (out.float() - dvec).to(out.dtype)
        return fn
    def clamp(s, a, direction=None):
        def fn(c, out):
            dvec = norm[s][1][streams[s].index(c)] * alpha410 * (saes[s].W_dec[a] if direction is None else direction)
            return (out.float() + gtk[..., None].float() * dvec).to(out.dtype)
        return fn
    def probe_atom(s, a):
        def fn(c, out):
            st["log"].append((st["t"], c, float(encode(s, c, out, mu_t[(FULL, st["t"], c)])[..., a].sum() / B)))
            return None
        return fn
    conds = [("normal", FULL, {}), ("ablate z_H 408 at the final read-out", FULL, {335: ablate("z_H", [408], FULL)}),
             ("ablate z_L 1733 at the last z_L update", FULL, {334: ablate("z_L", [1733], FULL), 335: probe_atom("z_H", 408)}),
             ("ablate z_H 100 (horizontal path) at the read-out", FULL, {335: ablate("z_H", [100], FULL)}),
             (f"control: ablate z_H {ctrl_100}", FULL, {335: ablate("z_H", [ctrl_100], FULL)}),
             ("ablate z_H 1122 ('not path', boundary rows)", FULL, {335: ablate("z_H", [1122], FULL)}),
             (f"control: ablate z_H {ctrl_1122}", FULL, {335: ablate("z_H", [ctrl_1122], FULL)}),
             ("clamp z_H 410 ON on the solution at the read-out", FULL, {335: clamp("z_H", 410)}),
             ("control: random direction, same size", FULL, {335: clamp("z_H", 410, rnd)}),
             ("normal (21 calls)", B21, {}), ("ablate z_H 410 at calls 14 and 21 (21 calls)", B21, {13: ablate("z_H", [410], B21), 20: ablate("z_H", [410], B21)}),
             (f"control: ablate z_H {ctrl_410} (21 calls)", B21, {13: ablate("z_H", [ctrl_410], B21), 20: ablate("z_H", [ctrl_410], B21)}),
             ("normal (14 calls)", B14, {}), ("ablate z_L 1312+1613+873 in calls 1-6 (14 calls)", B14, {c: ablate("z_L", [1312, 1613, 873], B14) for c in range(6)}),
             (f"control: ablate z_L {'+'.join(map(str, ctrl_L))} (14 calls)", B14, {c: ablate("z_L", ctrl_L, B14) for c in range(6)})]
    mu_calls = {FULL: (334, 335), B21: (13, 20), B14: tuple(range(6))}
    results, base_drawn, saved = [], {}, {}
    for name, budget, edits in conds:
        st["log"] = []
        x = x0; gen = torch.Generator(device=device).manual_seed(78)
        normal = not edits
        with torch.no_grad():
            for t in ts:
                s_ = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(B).to(device))
                st.update(mode="edit", c=0, t=int(t), edits=edits, mu_calls=mu_calls[budget] if normal else (), mu_acc={})
                pred_cond, _, _ = model.forward_with_carry(s_, n_sup=budget[0], H_cycles=budget[1], L_cycles=budget[2])
                if normal:
                    for (tt, c), v in st["mu_acc"].items():
                        mu_t[(budget, tt, c)] = v
                st["mode"] = None
                pred_uncond = model(model.null_condition_sample(s_)).pred
                pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
            out = model.decode_for_eval(x).clamp(0, 1)
        out_u8 = (out * 255).byte().cpu()
        drawn = torch.from_numpy(np.stack([pa.token_labels(out_u8[i], out_u8[i])[0] != "off" for i in range(B)]))
        if normal:
            base_drawn[budget] = drawn
        rows = []
        for i, ms in enumerate(mss):
            od = out_u8[i].float().mean(0).numpy() < WALL_DARK; md = maze_u8[i].float().mean(0).numpy() < WALL_DARK
            blue = mcl.blue_mask(out_u8[i].permute(1, 2, 0).numpy())
            pc_ = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
            path = set(ms.path); inner_c = path - {ms.path[0], ms.path[-1]}
            rows.append({"solved": inner_c <= pc_ and not (pc_ - path), "missing": len(inner_c - pc_), "extra": len(pc_ - path),
                         "blue_px": int(blue.sum()), "walls_changed": int((od & ~md & ~blue).sum() + (md & ~od).sum())})
        bd = base_drawn[budget]
        flip_on = float((drawn & ~bd).sum()) / B; flip_off = float((~drawn & bd).sum()) / B
        m = lambda k_: float(np.mean([r[k_] for r in rows]))
        log = np.array([v for (_, _, v) in st["log"]]) if st["log"] else None
        results.append({"name": name, "budget": budget, "rows": rows, "drawn": drawn, "flip_on": flip_on, "flip_off": flip_off,
                        "atom_log": st["log"]})
        saved[name] = out_u8[:8]
        print(f"  {name:52s} | solved {m('solved'):.2f}  missing {m('missing'):5.2f}  extra {m('extra'):5.2f}  blue px {m('blue_px'):6.0f}  "
              f"walls changed {m('walls_changed'):5.0f} | tokens drawn vs same-budget normal: +{flip_on:.2f} / -{flip_off:.2f} per board"
              + (f" | atom activity per board during edits (mean) {log.mean():.2f}" if log is not None else ""), flush=True)
    # FVU of the SAEs at the read-out over denoising steps (validity of the per-t templates)
    print("  SAE z_H FVU at the final read-out per denoising step is not recomputed here (see atom_log); per-t templates saved")
    torch.save({"contrib": contrib, "prune_k": prune_k, "score": sc_te, "acc": accs, "prune_sig": prune_sig, "attrib": attrib,
                "causal": results, "saved_imgs": saved, "kept": kept, "controls": {"100": ctrl_100, "1122": ctrl_1122, "410": ctrl_410, "zL": ctrl_L},
                "alpha410": alpha410, "maze_u8": maze_u8[:8], "sol_u8": (batch.images.float().clamp(0, 1) * 255).byte().cpu()[:8]}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
