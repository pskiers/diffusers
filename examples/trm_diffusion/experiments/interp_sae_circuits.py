"""TEMPORARY diagnostic (Claude, amaze interpretability): sparse feature circuits of the thinker, square 8x8 mazes.

States: one thinker pass at the first denoising step of a generation from pure noise (the thinker solves from scratch),
L_level calls 1..63 (reasoning steps 1-3: 54 z_L + 9 z_H updates; the state is near its fixed point by ~call 60) and
the last z_L / z_H update of the pass (calls 335, 336). Boards: square 8x8 mazes of the TRAINING set (original image
830x830 px, unique to square n8 among all shapes x sizes) to fit SAEs and the plan read-out; the 100 test boards
(n8_test) for everything reported.
(1) TopK SAE per stream (z_L, z_H): mean per (call, position) removed and one scale per call (train boards), m atoms,
    k active. Held-out FVU per call. Distance of every early state to the final one.
(2) Dynamics: mean activation per call -> transient atoms (on during calls 1-21, gone at the end) and target atoms
    (grow into the final state); where they fire (solution / off-solution / the thinker's own drops and late additions).
(3) Plan attribution, exact. Path score = logistic probe on lm_head(z_H) (linear in z_H; fit on train boards).
    z_H atoms: removing atom i (z - s f_i d_i, SAE error kept) changes the score by -s f_i (d_i . u) on its own tokens.
    z_L atoms: removed from the z_L that feeds a z_H update, that z_H update is recomputed (ONE step: later updates are
    known to undo edits, so this is the atom's role in that update, not its effect on the final plan). The same
    recomputation gives atom -> atom edges (change of z_H atoms when a z_L atom is removed).
(4) Components -> atoms: at the call where a selected atom switches on most, that block call is recomputed with one
    component mean-ablated (mean over the test boards per position): each of the 16 attention heads, each MLP, the
    state input, the maze embedding / z_H (z_L updates) or z_L (z_H updates) -> change of the atom's activation on
    the tokens where it was active.
"""
import base64
import dataclasses
import os
import struct

import hydra
import numpy as np
import pyarrow as pyar
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import interp_size_ablation as sa
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG, T = 12, 144, 144
N_TRAIN, BATCH = 768, 64
REC = list(range(63)) + [334, 335]                      # 0-based L_level calls kept
M_ATOMS, K, STEPS, BS, LR = 2048, 16, 20000, 4096, 3e-4
N_SEL, MIN_FREQ = 8, 1e-3


def train_subset(src_dir, out_path):
    """Square 8x8 boards of the training set, from maze_dataset_train_144 (ids picked by the 830x830 original image)."""
    if os.path.exists(out_path):
        return
    f = pq.ParquetFile(f"{src_dir}/maze_dataset_train_original.parquet")
    ids = []
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["id", "m_original_img"])
        for i, s in zip(t.column("id").to_pylist(), t.column("m_original_img").to_pylist()):
            if struct.unpack(">II", base64.b64decode(s[:64])[16:24]) == (830, 830):
                ids.append(i)
    t = pq.read_table(f"{src_dir}/maze_dataset_train_144.parquet")
    t = t.filter(pc.is_in(t.column("id"), value_set=pyar.array(ids)))
    pq.write_table(t, out_path)
    print(f"train subset: {len(ids)} square 8x8 ids, {t.num_rows} rows -> {out_path}", flush=True)


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.W_enc = torch.nn.Parameter(torch.randn(d, m) / d ** 0.5)
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.W_dec = torch.nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.renorm()

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8

    def encode(self, x):
        p = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        v, i = p.topk(self.k, dim=-1)
        return torch.zeros_like(p).scatter_(-1, i, v)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


def train_sae(X, name):
    """X: (N, C, T, D) normalised states on the GPU (fp16)."""
    torch.manual_seed(0)
    flat = X.view(-1, X.shape[-1])
    sae = TopKSAE(X.shape[-1], M_ATOMS, K).to(X.device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (STEPS - s) / (0.2 * STEPS)))
    fired = torch.zeros(M_ATOMS, dtype=torch.bool, device=X.device)
    for step in range(STEPS):
        bx = flat[torch.randint(len(flat), (BS,), device=X.device)].float()
        xh, f = sae(bx)
        loss = (xh - bx).pow(2).sum(-1).mean() / (bx - bx.mean(0)).pow(2).sum(-1).mean()      # batch FVU
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step(); sched.step(); sae.renorm()
        if step >= STEPS - 2000:
            fired |= (f > 0).any(0)
        if step % 4000 == 0 or step == STEPS - 1:
            print(f"  SAE {name} step {step:5d}: train FVU {loss.item():.3f}", flush=True)
    assert torch.isfinite(loss) and loss.item() < 1, "SAE training failed"
    print(f"  SAE {name}: {int((~fired).sum())} of {M_ATOMS} atoms dead (never active in the last 2000 steps)", flush=True)
    return sae


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
    t0 = model.scheduler.timesteps[0]
    inner = model.thinker.inner
    layers = inner.L_level.layers
    per, D = inner.config.L_cycles + 1, inner.config.hidden_size
    hd, nH = layers[0].self_attn.head_dim, layers[0].self_attn.num_heads
    kind = lambda c: "z_H" if c % per == per - 1 else "z_L"
    pos = {c: j for j, c in enumerate(REC)}
    print(f"recording at the first denoising step t={int(t0)}; {len(REC)} calls kept", flush=True)

    train_path = str(ic.train_subset)
    train_subset(str(ic.train_src), train_path)
    vd = cfg.data.get("val_dataset", {})
    mk = lambda p, split: AmazeDataset(dataset_path=p, split=split, image_size=IMG, num_channels=3, include_metadata=False,
                                       condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"))
    ds_tr, ds_te = mk(train_path, "train"), mk(str(ic.data_parquet), "test")

    # ---------------------------------------------------------------- recording
    rec = {"on": False, "c": 0}

    def pre_hook(module, args, kwargs):
        if rec["on"]:
            if rec["c"] == 0:
                rec["L0"], rec["inj0"], rec["cos_sin"] = args[0][:1].clone(), args[1].clone(), kwargs.get("cos_sin")
            if rec["c"] == per - 1:
                rec["H0"] = args[0][:1].clone()

    def fwd_hook(module, args, kwargs, output):
        if rec["on"]:
            if rec["c"] in pos:
                rec["buf"][:, pos[rec["c"]]] = output.detach().cpu()
            rec["c"] += 1
    inner.L_level.register_forward_pre_hook(pre_hook, with_kwargs=True)
    inner.L_level.register_forward_hook(fwd_hook, with_kwargs=True)

    def record(ds, n_max, seed):
        n = min(n_max, len(ds))
        S, gts, cats, embs = None, [], [], []
        for i0 in range(0, n, BATCH):
            batch = AmazeDataset.collate_fn([ds[i] for i in range(i0, min(i0 + BATCH, n))])
            cond = model._batch_to_sample(batch, device)
            B = batch.images.shape[0]
            x = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(seed + i0), device=device)
            s = dataclasses.replace(cond, x_noisy=x, timesteps=t0.expand(B).to(device))
            buf = torch.empty((B, len(REC), T, D), dtype=inner.forward_dtype)
            rec.update(on=True, c=0, buf=buf)
            with torch.no_grad():
                model.forward_with_carry(s)
            rec["on"] = False
            assert rec["c"] == 336, f"{rec['c']} L_level calls, expected 336"
            S = buf if S is None else torch.cat([S, buf])
            embs.append((rec["inj0"] - rec["H0"]).cpu())
            sol = (batch.images.float().clamp(0, 1) * 255).byte().cpu()
            maze = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
            c_ = np.stack([pa.token_labels(sol[i], maze[i])[0] for i in range(B)])
            cats.append(c_); gts.append(c_ != "off")
        print(f"  recorded {len(S)} boards", flush=True)
        return S, np.concatenate(gts), np.concatenate(cats), torch.cat(embs)
    S_tr, gt_tr, _, _ = record(ds_tr, N_TRAIN, 1000)
    S_te, gt_te, cat_te, emb_te = record(ds_te, 100, 5000)
    Ntr, Nte = len(S_tr), len(S_te)
    L0, H0, cos_sin = rec["L0"], rec["H0"], rec["cos_sin"]

    # ---------------------------------------------------------------- how far is each early state from the final one
    streams = {"z_L": [c for c in REC if kind(c) == "z_L"], "z_H": [c for c in REC if kind(c) == "z_H"]}
    print("\ndistance to the final state ||z_c - z_final|| / ||z_final|| (median over test boards) and change per update:")
    for name, calls in streams.items():
        zf = S_te[:, pos[calls[-1]]].float().flatten(1)
        d = {c: float(((S_te[:, pos[c]].float().flatten(1) - zf).norm(dim=1) / zf.norm(dim=1)).median()) for c in calls[:-1]}
        ch = {c: float(((S_te[:, pos[c]].float().flatten(1) - S_te[:, pos[p]].float().flatten(1)).norm(dim=1)
                        / S_te[:, pos[c]].float().flatten(1).norm(dim=1)).median()) for p, c in zip(calls[:-2], calls[1:-1])}
        show = [c for c in calls[:-1] if c in (0, 5, 6, 12, 13, 19, 20, 27, 34, 41, 48, 55, 61, 62)]
        print(f"  {name}: " + "  ".join(f"call {c + 1}: {d[c]:.3f}" + (f" (Δ {ch[c]:.3f})" if c in ch else "") for c in show))

    # ---------------------------------------------------------------- plan read-out (linear in z_H)
    W = inner.lm_head.weight.detach().float().to(device)                                     # (vocab, D)
    zf_tr = S_tr[:, pos[335]].float().to(device).reshape(-1, D)
    probe = sa.fit_probe(zf_tr @ W.T, torch.from_numpy(gt_tr.reshape(-1).astype(np.float32)).to(device),
                         torch.ones(len(zf_tr), dtype=torch.bool, device=device))
    c0 = probe(torch.zeros(1, W.shape[0], device=device))
    u = W.T @ (probe(torch.eye(W.shape[0], device=device)) - c0)                       # score(z) = z . u + c0
    del zf_tr
    score = lambda z: z.float() @ u + c0
    sc_te = {c: score(S_te[:, pos[c]].to(device)).cpu() for c in streams["z_H"]}        # (Nte, T) per z_H call
    gt_t = torch.from_numpy(gt_te)
    acc = float(((sc_te[335] > 0) == gt_t).float().mean())
    exact = float(((sc_te[335] > 0) == gt_t).all(1).float().mean())
    drops = (~gt_t) & (sc_te[per - 1] > 0) & (sc_te[335] < 0)
    late = gt_t & (sc_te[per - 1] < 0) & (sc_te[335] > 0)
    print(f"\nplan read-out (probe on lm_head(z_H), fit on {Ntr} train boards): test token acc {acc:.4f}, boards exact {exact:.2f}; "
          f"per z_H call acc: " + " ".join(f"{c + 1}:{float(((sc_te[c] > 0) == gt_t).float().mean()):.3f}" for c in streams["z_H"]))
    print(f"thinker's own drops (off-solution tokens, path at call {per}, off at the end): {int(drops.sum())}; "
          f"late additions (solution tokens, off at call {per}, path at the end): {int(late.sum())}")
    g = gt_t.float() * 2 - 1                                                           # +1 solution, -1 off

    # ---------------------------------------------------------------- SAEs
    out = {"REC": REC, "gt": gt_te, "cats": cat_te, "drops": drops, "late": late, "score": sc_te, "t0": int(t0)}
    saes, norm, codes = {}, {}, {}
    print("\nTopK SAEs (template per call x position removed, one scale per call; train boards):")
    for name, calls in streams.items():
        mu = torch.stack([S_tr[:, pos[c]].float().mean(0) for c in calls])            # (C, T, D)
        scl = torch.stack([(S_tr[:, pos[c]].float() - mu[j]).std() for j, c in enumerate(calls)])
        X = torch.empty((Ntr, len(calls), T, D), dtype=torch.float16, device=device)
        for j, c in enumerate(calls):
            X[:, j] = ((S_tr[:, pos[c]].float() - mu[j]) / scl[j]).half().to(device)
        sae = train_sae(X, name)
        del X
        mu, scl = mu.to(device), scl.to(device)
        saes[name], norm[name] = sae, (mu, scl)
        idx = torch.empty((Nte, len(calls), T, K), dtype=torch.int16); val = torch.empty((Nte, len(calls), T, K), dtype=torch.float16)
        fvu = []
        with torch.no_grad():
            for j, c in enumerate(calls):
                x = (S_te[:, pos[c]].float().to(device) - mu[j]) / scl[j]
                xh, f = sae(x)
                fvu.append(float((x - xh).pow(2).sum() / (x - x.mean((0, 1))).pow(2).sum()))
                v, i = f.topk(K, dim=-1)
                idx[:, j], val[:, j] = i.short().cpu(), v.half().cpu()
        codes[name] = (idx, val)
        print(f"  {name}: held-out FVU per call: " + " ".join(f"{c + 1}:{fv:.2f}" for c, fv in zip(calls, fvu) if c in (0, 6, 12, 13, 20, 27, 41, 62, 334, 335)))
        out[f"sae_{name}"] = {k: v.cpu() for k, v in sae.state_dict().items()}
        out[f"norm_{name}"] = (mu.cpu().half(), scl.cpu())
        out[f"codes_{name}"] = codes[name]
        out[f"fvu_{name}"] = fvu

    def act_of(name, a):
        """dense activation of atom a: (Nte, C, T)."""
        idx, val = codes[name]
        return ((idx.long() == a).float() * val.float()).sum(-1)

    # ---------------------------------------------------------------- (2) dynamics
    stats = {}
    for name, calls in streams.items():
        idx, val = codes[name]
        C = len(calls)
        ma = torch.zeros(C, M_ATOMS); fq = torch.zeros(C, M_ATOMS)
        for j in range(C):
            ma[j].index_add_(0, idx[:, j].flatten().long(), val[:, j].flatten().float())
            fq[j].index_add_(0, idx[:, j].flatten().long(), (val[:, j].flatten() > 0).float())
        ma /= Nte * T; fq /= Nte * T
        early = torch.tensor([c <= 20 for c in calls])
        peak = ma[early].max(0).values
        trans = torch.where((ma[-1] < 0.2 * peak) & (fq[early].max(0).values >= 5 * MIN_FREQ), peak - ma[-1], torch.tensor(-1.0))
        targ = torch.where(fq[-1] >= 5 * MIN_FREQ, ma[-1] - ma[0], torch.tensor(-1.0))
        stats[name] = {"mean_act": ma, "freq": fq, "transient": trans.topk(N_SEL).indices.tolist(), "target": targ.topk(N_SEL).indices.tolist()}
    out["stats"] = stats

    def describe(name, a):
        calls = streams[name]
        A = act_of(name, a)                                                            # (Nte, C, T)
        early = A[:, [j for j, c in enumerate(calls) if c <= 20]].sum(1); fin = A[:, -1]
        def shares(Z):
            tot = float(Z.sum()) + 1e-9
            return (f"sol {float(Z[gt_t].sum()) / tot:.2f} off {float(Z[~gt_t].sum()) / tot:.2f} "
                    f"drop {float(Z[drops].sum()) / tot:.2f} late {float(Z[late].sum()) / tot:.2f}")
        ma = stats[name]["mean_act"][:, a]
        stage = lambda lo, hi: float(ma[[j for j, c in enumerate(calls) if lo <= c <= hi]].mean()) if any(lo <= c <= hi for c in calls) else float("nan")
        Z = A.sum(1)
        cat_mass = {k: float(Z[torch.from_numpy(cat_te == k)].sum()) for k in np.unique(cat_te)}
        top_cat = max(cat_mass, key=cat_mass.get)
        npos = int((Z.sum(0).sort(descending=True).values.cumsum(0) < 0.5 * Z.sum()).sum()) + 1
        return (f"{name} atom {a:4d}: mean act calls 1-7 {stage(0, 6):.3f}, 8-21 {stage(7, 20):.3f}, 22-63 {stage(21, 62):.3f}, final {float(ma[-1]):.3f} | "
                f"early mass: {shares(early)} | final mass: {shares(fin)} | top stroke class {top_cat} "
                f"({cat_mass[top_cat] / (sum(cat_mass.values()) + 1e-9):.2f}) | half of the mass on {npos} of 144 positions")
    base = f"base rates (test tokens): sol {float(gt_t.float().mean()):.2f}, drop {float(drops.float().mean()):.3f}, late {float(late.float().mean()):.3f}"
    print(f"\n(2) dynamics — {base}")
    for name in streams:
        for key in ("transient", "target"):
            print(f"  {key} atoms ({name}):")
            for a in stats[name][key]:
                print("    " + describe(name, a))

    # ---------------------------------------------------------------- recomputation helpers
    st = lambda c: S_te[:, pos[c]].to(device)
    L0b, H0b, emb = L0.expand(Nte, -1, -1), H0.expand(Nte, -1, -1), emb_te.to(device)

    def parts(c):
        if kind(c) == "z_L":
            prev = c - 1 if (c - 1) % per != per - 1 else c - 2
            return {"hidden": L0b if c == 0 else st(prev), "zH": H0b if c < per else st((c // per) * per - 1), "emb": emb}
        return {"hidden": H0b if c == per - 1 else st(c - per), "zL": st(c - 1)}

    def block(p):
        inj = p["zL"] if "zL" in p else p["zH"] + p["emb"]
        with torch.no_grad():
            return inner.L_level(p["hidden"], inj, cos_sin=cos_sin)

    def encode(name, c, z):
        mu, scl = norm[name]; j = streams[name].index(c)
        with torch.no_grad():
            return saes[name].encode((z.float() - mu[j]) / scl[j])

    def dec_raw(name, c, a):
        return norm[name][1][streams[name].index(c)] * saes[name].W_dec[a].detach()      # raw-scale direction of atom a

    errs = [float((block(parts(c)).float() - st(c).float()).norm() / st(c).float().norm()) for c in (0, 5, 6, 7, 13, 21, 62)]
    print("\nsanity: recomputed vs recorded block output, relative error at calls 1, 6, 7, 8, 14, 22, 63: " + " ".join(f"{e:.4f}" for e in errs))

    # ---------------------------------------------------------------- (3) plan attribution + edges
    print("\n(3) plan attribution — support = how much the atom pushes the read-out towards the CORRECT plan "
          "(score units per board; >0 helps); flips = plan tokens per board that flip when the atom is removed")
    attr = {}
    zH_calls = [c for c in streams["z_H"] if c <= 62] + [335]
    for c in zH_calls:                                                                 # z_H atoms, exact and linear
        f = encode("z_H", c, st(c))                                                    # (Nte, T, M)
        du = norm["z_H"][1][streams["z_H"].index(c)] * (saes["z_H"].W_dec.detach() @ u)    # (M,) score per unit of each atom
        dsc = -f * du                                                                  # score change if removed
        s0 = sc_te[c].to(device)[..., None]
        support = -(dsc * g.to(device)[..., None]).sum((0, 1)) / Nte
        flips = (((s0 + dsc) > 0) != (s0 > 0)).float().sum((0, 1)) / Nte
        attr[("z_H", c)] = {"support": support.cpu(), "flips": flips.cpu(), "writes": du.cpu(), "freq": (f > 0).float().mean((0, 1)).cpu()}
    edges = {}
    for c in [x for x in streams["z_H"] if x <= 62]:                                   # z_L atoms, one z_H update recomputed
        p = parts(c)
        zl_c = c - 1
        fL = encode("z_L", zl_c, p["zL"])
        base_out = block(p)
        s0 = score(base_out)
        fH0 = encode("z_H", c, base_out)
        hsel = attr[("z_H", c)]["support"].abs().topk(16).indices.to(device)
        freq = (fL > 0).float().mean((0, 1))
        act_atoms = torch.nonzero(freq >= MIN_FREQ).flatten().tolist()
        sup, flp, loc = torch.zeros(M_ATOMS), torch.zeros(M_ATOMS), torch.full((M_ATOMS,), float("nan"))
        E = torch.zeros(len(act_atoms), 16)
        for n_, a in enumerate(act_atoms):
            p2 = dict(p); p2["zL"] = p["zL"] - (fL[..., a:a + 1] * dec_raw("z_L", zl_c, a)).to(p["zL"].dtype)
            o2 = block(p2)
            ds = score(o2) - s0
            sup[a] = float((ds * g.to(device)).sum() / Nte)
            flp[a] = float((((s0 + ds) > 0) != (s0 > 0)).float().sum() / Nte)
            near = F.max_pool2d((fL[..., a] > 0).float().view(Nte, 1, G, G), 3, 1, 1).view(Nte, T) > 0
            loc[a] = float(ds.abs()[near].sum() / (ds.abs().sum() + 1e-9))
            E[n_] = ((encode("z_H", c, o2) - fH0)[..., hsel]).sum(1).mean(0).cpu()
        attr[("z_L", zl_c)] = {"support": sup, "flips": flp, "local": loc, "freq": freq.cpu()}
        edges[c] = {"zL_atoms": act_atoms, "zH_atoms": hsel.cpu().tolist(), "delta": E}
        print(f"  z_H update at call {c + 1}: {len(act_atoms)} z_L atoms tested; total |support| of removed z_L atoms "
              f"{float(sup.abs().sum()):.1f}", flush=True)
    out["attr"], out["edges"] = attr, edges

    for c in zH_calls:
        a_ = attr[("z_H", c)]
        if c not in (per - 1, 3 * per - 1, 335):
            continue
        print(f"  z_H atoms at call {c + 1} (exact): top support / most harmful")
        for a in a_["support"].topk(5).indices.tolist() + (-a_["support"]).topk(2).indices.tolist():
            print(f"    atom {a:4d}: support {float(a_['support'][a]):+.2f}, flips {float(a_['flips'][a]):.2f}, writes "
                  f"{'+path' if a_['writes'][a] > 0 else '-path'} ({float(a_['writes'][a]):+.2f}/unit), freq {float(a_['freq'][a]):.3f}")
    agg = torch.stack([attr[("z_L", c - 1)]["support"] for c in streams["z_H"] if c <= 62])   # (9, M)
    best_L = agg.abs().max(0).values.topk(N_SEL).indices.tolist()
    print("  z_L atoms with the largest one-step support (any of the 9 z_H updates):")
    for a in best_L:
        j = int(agg[:, a].abs().argmax()); c = [x for x in streams["z_H"] if x <= 62][j]
        r = attr[("z_L", c - 1)]
        print(f"    atom {a:4d}: at the z_H update of call {c + 1}: support {float(r['support'][a]):+.2f}, flips {float(r['flips'][a]):.2f}, "
              f"share of the effect within 1 token of where it fires {float(r['local'][a]):.2f}")
        print("      " + describe("z_L", a))
    print("  strongest z_L -> z_H atom edges (mean change of the z_H atom's total activation per board when the z_L atom is removed):")
    flat_e = [(float(e["delta"][i, j]), c, e["zL_atoms"][i], e["zH_atoms"][j]) for c, e in edges.items()
              for i in range(len(e["zL_atoms"])) for j in range(len(e["zH_atoms"]))]
    for v, c, aL, aH in sorted(flat_e, key=lambda x: -abs(x[0]))[:12]:
        print(f"    call {c + 1}: z_L {aL:4d} -> z_H {aH:4d}: {v:+.3f}")

    # ---------------------------------------------------------------- (4) components -> atoms
    comp = {"on": None}
    def o_pre(l):
        def f(module, args):
            if comp["on"] is None or comp["on"][0] != ("head", l):
                return None
            x = args[0].clone(); h = comp["on"][1]
            x[..., h * hd:(h + 1) * hd] = x[..., h * hd:(h + 1) * hd].mean(0, keepdim=True)
            return (x,)
        return f
    def mlp_post(l):
        def f(module, args, output):
            if comp["on"] is None or comp["on"][0] != ("mlp", l):
                return None
            return output.mean(0, keepdim=True).expand_as(output).contiguous()
        return f
    for l, layer in enumerate(layers):
        layer.self_attn.o_proj.register_forward_pre_hook(o_pre(l))
        layer.mlp.register_forward_hook(mlp_post(l))
    comps = [(("head", l), h) for l in range(len(layers)) for h in range(nH)] + [(("mlp", l), None) for l in range(len(layers))]

    sel = {"z_L": sorted(set(stats["z_L"]["transient"] + stats["z_L"]["target"] + best_L)),
           "z_H": sorted(set(stats["z_H"]["transient"] + stats["z_H"]["target"] + attr[("z_H", 335)]["support"].abs().topk(N_SEL).indices.tolist()))}
    print("\n(4) components -> atoms: share of the atom's activation (on the tokens where it fired) removed when one "
          "component is mean-ablated in the call where the atom switches on most (top 4 components)")
    comp_res = {}
    for name, atoms in sel.items():
        calls = [c for c in streams[name] if c <= 62]
        ma = stats[name]["mean_act"][:len(calls)]
        onset = {a: calls[int(torch.cat([ma[:1, a], ma[1:, a] - ma[:-1, a]]).argmax())] for a in atoms}
        for c in sorted(set(onset.values())):
            here = [a for a in atoms if onset[a] == c]
            p = parts(c)
            f0 = encode(name, c, block(p))[..., here]                                  # (Nte, T, n)
            on = f0 > 0
            tot = (f0 * on).sum((0, 1)) + 1e-9
            res = {}
            runs = [(k, h) for k, h in comps] + [(("input", key), None) for key in p]
            for k, h in runs:
                if k[0] == "input":
                    p2 = dict(p); p2[k[1]] = p[k[1]].mean(0, keepdim=True).expand_as(p[k[1]])
                    f2 = encode(name, c, block(p2))[..., here]
                else:
                    comp["on"] = (k, h)
                    f2 = encode(name, c, block(p))[..., here]
                    comp["on"] = None
                res[(k, h)] = (1 - (f2 * on).sum((0, 1)) / tot).cpu()                  # share removed per atom
            for n_, a in enumerate(here):
                ranked = sorted(res, key=lambda r: -float(res[r][n_]))[:4]
                lab = lambda r: (f"L{r[0][1]}H{r[1]}" if r[0][0] == "head" else f"MLP{r[0][1]}" if r[0][0] == "mlp" else f"input:{r[0][1]}")
                print(f"  {name} atom {a:4d} (onset call {c + 1}): " + ", ".join(f"{lab(r)} {float(res[r][n_]):+.2f}" for r in ranked))
                comp_res[(name, a)] = {"onset": c, "removed": {lab(r): float(res[r][n_]) for r in res}}
    out["components"] = comp_res
    torch.save(out, str(ic.out))
    torch.save({"S_te": S_te, "emb_te": emb_te, "L0": L0.cpu(), "H0": H0.cpu()}, str(ic.out).replace(".pt", "_test_states.pt"))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
