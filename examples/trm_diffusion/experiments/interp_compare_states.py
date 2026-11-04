"""TEMPORARY (Claude): WHAT do z_L / z_H encode in PaTh's thinker vs plain TRMs, and how are the states structured?

Models: PaTh (checkpoint; maze noised to t = 70, conditional pass) and the plain TRMs in PLAIN_CKPTS="name:path,...".
Mazes as interp_loops_first.py (maze_size, side_pick, min_side, seed 0): the unedited maze and the lollipop maze per board.
Snapshots: encoder output (emb), z_L after z_L updates SNAP_L and the last, z_H after z_H updates SNAP_H and the last.
Every token is assigned the maze cell containing its centre (square mazes).

1. decoding (linear, 2-fold over boards; ridge, closed form): per snapshot and feature
     path            token label path / off (unedited maze; ignored tokens skipped)        balanced accuracy
     def_round       dead-end filling round in which the cell is removed (off cells)        R^2
     d_start/d_goal  distance from the start / goal along open passages (all cells)         R^2
     open_bits       the 4 open-passage bits of the cell (mean over the 4)                  balanced accuracy
     row/col         token position                                                         R^2
     in_lollipop     lollipop maze: off tokens of the lollipop vs other off tokens          balanced accuracy
2. structure: share of the state's energy that is board-specific (minus the mean over boards at the same token position);
   effective dimension (participation ratio of the PCA spectrum) of the centred state; relative change per z_L update
3. cross-model: ridge from one model's state to another's (same snapshot), share of variance explained (2-fold)
Output: <out_dir>/compare_states.pt, printed tables.
"""
import dataclasses
import logging
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from check_loops import render_labels, variants
from interp_prune_mech import CONFIG_DIR, T

G = 12
SNAP_L = [0, 5, 11, 17, 35]
SNAP_H = [0, 1, 2, 5]
N_TRACE = 48                                                    # z_L updates with the relative change recorded


def bfs(nb, src):
    d = {src: 0}; q = deque([src])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def def_rounds(nb, s, g):
    alive, rnd, k = set(nb), {}, 0
    while True:
        k += 1
        dead = [c for c in alive if c not in (s, g) and sum(b in alive for b in nb[c]) <= 1]
        if not dead:
            return rnd
        for c in dead:
            rnd[c] = k; alive.discard(c)


def ridge_fit(X, Y, lam):
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc = X - Xm
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), Xc.T @ (Y - Ym))
    return W, Xm, Ym


def ridge_cv(X, Y, fold, lam=10.0):
    """2-fold ridge; returns held-out predictions (n, k)."""
    P = torch.zeros_like(Y)
    for f in (0, 1):
        tr, te = fold == f, fold != f
        W, Xm, Ym = ridge_fit(X[tr], Y[tr], lam)
        P[te] = (X[te] - Xm) @ W + Ym
    return P


def r2(P, Y):
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def bal_acc(score, y):
    pred = score > 0.5
    return float(0.5 * ((pred[y == 1]).float().mean() + (~pred[y == 0]).float().mean()))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 100)), int(ic.get("batch_size", 22)), int(ic.get("seed", 0))
    maze_size, side_pick, min_side = int(ic.get("maze_size", 8)), str(ic.get("side_pick", "random")), int(ic.get("min_side", 0))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)

    # ---------------------------------------------------------------- mazes + per-token features
    rng = random.Random(seed)
    boards = []
    i = 0
    cp = 144 / maze_size
    tok_cell = [(int((12 * (t // 12) + 6) // cp), int((12 * (t % 12) + 6) // cp)) for t in range(T)]
    while i < len(ds) and len(boards) < n_boards:
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == maze_size:
            vs = variants(ms, rng, side_pick, min_side)
            if vs is not None:
                n0, v = vs["none"], vs["side"]
                nb = n0["nb"]
                rc = [ms.id2rc[p] for p in ms.path]
                s, g = rc[0], rc[-1]
                dS, dG, rnd = bfs(nb, s), bfs(nb, g), def_rounds(nb, s, g)
                feat = dict(path=render_labels(ds, ms, n0["img"], nb, n0["sp"]).astype(np.float32),
                            def_round=np.array([rnd.get(c, 0) for c in tok_cell], np.float32),
                            on_sol=np.array([c in n0["sp"] for c in tok_cell], bool),
                            d_start=np.array([dS.get(c, -1) for c in tok_cell], np.float32),
                            d_goal=np.array([dG.get(c, -1) for c in tok_cell], np.float32),
                            open_bits=np.array([[(r2, c2) in nb.get(c, []) for (r2, c2) in ((c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0] + 1, c[1]), (c[0], c[1] - 1))]
                                                for c in tok_cell], np.float32))
                lab_side = render_labels(ds, ms, v["img"], v["nb"], v["sp"])
                E = (render_labels(ds, ms, v["img"], v["nb"], v["def_"]) == 1) & (lab_side == 0)
                mk = lambda k: dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(vs[k]["img"])))
                boards.append(dict(board=i, fold=len(boards) % 2, feat=feat, E=E, lab_side=lab_side, none=mk("none"), side=mk("side")))
        i += 1
    N = len(boards)
    logging.info(f"{N} boards")

    # ---------------------------------------------------------------- models
    models = {}
    pm = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(pm, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    pm = accelerator.unwrap_model(accelerator.prepare(pm)); pm.eval()
    pm.scheduler.set_timesteps(20, device=device)
    models["PaTh"] = ("path", pm, pm.thinker.inner)
    from train_plain_trm import PlainTRM
    for spec in os.environ.get("PLAIN_CKPTS", "").split(","):
        if not spec:
            continue
        name, path = spec.split(":", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        m.load_state_dict(ck["ema"]); m = m.to(device).eval()
        models[name] = ("plain", m, m.thinker.inner)

    st = {"on": False}

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def make_hook(inner, per):
        def hook(module, args, kwargs, output):
            if not st["on"]:
                return None
            c = st["call"]
            z_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
            if c == 1:
                inj = args[1] if len(args) > 1 else kwargs["input_injection"]
                st["rec"]["emb"] = (inj.float() - inner.H_init.float().view(1, 1, -1)).half().cpu()
            if c % per != 0:
                st["zl_k"] += 1; k = st["zl_k"]
                if k in SNAP_L:
                    st["rec"][f"zL{k}"] = output.half().cpu()
                st["rec"]["zL_last"] = output.half().cpu()
                if k < N_TRACE:
                    st["rec"].setdefault("rel_change", []).append(
                        ((output.float() - z_in.float()).norm(dim=-1) / z_in.float().norm(dim=-1).clamp_min(1e-6)).mean().item())
            else:
                st["zh_k"] += 1; k = st["zh_k"]
                if k in SNAP_H:
                    st["rec"][f"zH{k}"] = output.half().cpu()
                st["rec"]["zH_last"] = output.half().cpu()
            return None
        return hook

    for name, (kind, m, inner) in models.items():
        inner.L_level.register_forward_pre_hook(pre_hook)
        inner.L_level.register_forward_hook(make_hook(inner, inner.config.L_cycles + 1), with_kwargs=True)

    data, rel = {}, {}
    for name, (kind, m, inner) in models.items():
        for which in ("none", "side"):
            parts, rc_ = [], []
            for k0 in range(0, N, bs):
                chunk = boards[k0:k0 + bs]
                st.update(on=True, call=0, zl_k=-1, zh_k=-1, rec={})
                with torch.no_grad():
                    smps = [b[which] for b in chunk]
                    if kind == "plain":
                        torch.manual_seed(seed + k0)
                        m.think(torch.stack([s.spatial_conditions for s in smps]).to(device))
                    else:
                        batch = AmazeDataset.collate_fn(smps)
                        c_ = m._batch_to_sample(batch, device)
                        empty = batch.spatial_conditions.to(device)
                        noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                        tt = torch.full((len(smps),), 70, device=device, dtype=torch.long)
                        m.forward_with_carry(dataclasses.replace(c_, x_noisy=m.scheduler.add_noise(empty, noise, tt), timesteps=tt))
                st["on"] = False
                rc_.append(st["rec"].pop("rel_change")); parts.append(st["rec"])
            data[(name, which)] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            if which == "none":
                rel[name] = np.mean(np.array(rc_), 0)
        logging.info(f"recorded {name}")

    names = list(models)
    snaps = ["emb"] + [f"zL{k}" for k in SNAP_L] + ["zL_last"] + [f"zH{k}" for k in SNAP_H] + ["zH_last"]
    fold_b = torch.tensor([b["fold"] for b in boards])
    fold_t = fold_b[:, None].expand(N, T).reshape(-1)
    F_ = {k: torch.tensor(np.stack([b["feat"][k] for b in boards])).reshape(N * T, *np.stack([b["feat"][k] for b in boards]).shape[2:])
          for k in boards[0]["feat"]}
    rows_ = torch.arange(T).repeat(N) // G; cols_ = torch.arange(T).repeat(N) % G
    E = torch.tensor(np.stack([b["E"] for b in boards])).reshape(-1)
    lab_side = torch.tensor(np.stack([b["lab_side"] for b in boards])).reshape(-1)

    # ---------------------------------------------------------------- 1. decoding
    res = {}
    print(f"\n[compare_states] n{maze_size}: {N} boards; models {names}")
    print("1) linear decoding, 2-fold over boards (balanced accuracy for path / open_bits / in_lollipop, R^2 for the rest)")
    hdr = ["path", "def_round", "d_start", "d_goal", "open_bits", "row", "col", "in_lollipop"]
    print("   " + f"{'model':8s} {'snapshot':8s} " + " ".join(f"{h:>11s}" for h in hdr))
    for name in names:
        for snap in snaps:
            X = data[(name, "none")][snap].reshape(N * T, -1).float().to(device)
            r = {}
            y = F_["path"].to(device); k = y >= 0
            P = ridge_cv(X[k], y[k, None], fold_t.to(device)[k]); r["path"] = bal_acc(P[:, 0], y[k])
            off = (F_["path"] == 0).to(device) & (F_["def_round"] > 0).to(device)
            r["def_round"] = r2(ridge_cv(X[off], F_["def_round"].to(device)[off, None], fold_t.to(device)[off]), F_["def_round"].to(device)[off, None])
            for key in ("d_start", "d_goal"):
                yy = F_[key].to(device); kk = yy >= 0
                r[key] = r2(ridge_cv(X[kk], yy[kk, None], fold_t.to(device)[kk]), yy[kk, None])
            ob = F_["open_bits"].to(device)
            Pb = ridge_cv(X, ob, fold_t.to(device))
            r["open_bits"] = float(np.mean([bal_acc(Pb[:, j], ob[:, j]) for j in range(4)]))
            r["row"] = r2(ridge_cv(X, rows_.float().to(device)[:, None], fold_t.to(device)), rows_.float().to(device)[:, None])
            r["col"] = r2(ridge_cv(X, cols_.float().to(device)[:, None], fold_t.to(device)), cols_.float().to(device)[:, None])
            Xs = data[(name, "side")][snap].reshape(N * T, -1).float().to(device)
            offs = (lab_side == 0).to(device)
            yl = E.float().to(device)[offs]
            r["in_lollipop"] = bal_acc(ridge_cv(Xs[offs], yl[:, None], fold_t.to(device)[offs])[:, 0], yl)
            res[(name, snap)] = r
            print(f"   {name:8s} {snap:8s} " + " ".join(f"{r[h]:11.2f}" for h in hdr), flush=True)

    # ---------------------------------------------------------------- 2. structure
    print("2) structure of the state (unedited mazes): board-specific energy share | effective dimension (participation ratio) | norm")
    struct = {}
    for name in names:
        for snap in snaps:
            Z = data[(name, "none")][snap].float()                                   # (N, T, D)
            spec = Z - Z.mean(0, keepdim=True)
            share = float(spec.pow(2).sum() / Z.pow(2).sum())
            Zc = Z.reshape(N * T, -1); Zc = Zc - Zc.mean(0)
            ev = torch.linalg.svdvals(Zc).pow(2)
            pr = float(ev.sum() ** 2 / ev.pow(2).sum())
            struct[(name, snap)] = (share, pr, float(Z.norm(dim=-1).mean()))
        print(f"   {name:8s} " + " | ".join(f"{snap} {struct[(name, snap)][0]:.3f} {struct[(name, snap)][1]:5.1f}" for snap in snaps))
    print("   relative change of z_L per update (updates 0..47): ")
    for name in names:
        print(f"   {name:8s} " + " ".join(f"{x:.3f}" for x in rel[name]))

    # ---------------------------------------------------------------- 3. cross-model mapping
    print("3) cross-model linear mapping (ridge, 2-fold): share of the target's variance explained, source -> target")
    cross = {}
    for snap in ("emb", "zL5", "zL_last", "zH0", "zH_last"):
        line = []
        for a in names:
            for b in names:
                if a == b:
                    continue
                Xa = data[(a, "none")][snap].reshape(N * T, -1).float().to(device)
                Xb = data[(b, "none")][snap].reshape(N * T, -1).float().to(device)
                v = r2(ridge_cv(Xa, Xb, fold_t.to(device)), Xb)
                cross[(snap, a, b)] = v; line.append(f"{a}->{b} {v:.2f}")
        print(f"   {snap:7s} " + "  ".join(line))
    torch.save(dict(decoding=res, structure=struct, rel_change=rel, cross=cross, models=names, snaps=snaps), out_dir / "compare_states.pt")
    print(f"saved {out_dir}/compare_states.pt")


if __name__ == "__main__":
    main()
