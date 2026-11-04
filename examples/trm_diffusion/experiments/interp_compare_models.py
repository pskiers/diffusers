"""TEMPORARY (Claude): encoder test (b) and representation comparison (c) across PaTh and plain TRM variants.

Models: PaTh (checkpoint, maze noised to t = 70, conditional pass) and the plain TRMs in the environment variable
PLAIN_CKPTS="name:path,name:path,..." (noisy variants: pure-noise x_t, as in their eval).
Mazes as interp_loops_first.py (maze_size, side_pick, min_side, seed 0); per board the unedited and the loop maze.
Recorded per model and maze: the encoder output emb (= input injection of call 1 − H_init; for PaTh it includes x_t and
the timestep bias), z_L after z_L updates SNAP_L and after the last one, z_H after z_H updates SNAP_H and the last one.

(b) encoder: logistic probes on emb (2-fold over boards):
    - loop vs no loop: the lollipop tokens E in the loop maze (1) vs the same tokens in the unedited maze (0), accuracy by
      Chebyshev distance of the token from the opening (the tokens of the removed wall)
    - path vs off on the unedited maze (token labels of the shortest path), accuracy and share of boards all right
    the same two probes on z_L after update 5 and z_H after the first z_H update, for comparison
(c) linear CKA between every pair of models at every snapshot, over all tokens of all mazes, and over the loop-induced
    change on the E tokens (state in the loop maze − state in the unedited maze)
Output: <out_dir>/compare_models.pt, printed summary.
"""
import dataclasses
import logging
import os
import random
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
from interp_prune_mech3 import train_linear

G = 12
_r, _c = torch.arange(T) // G, torch.arange(T) % G
CHEB = torch.maximum((_r[:, None] - _r[None]).abs(), (_c[:, None] - _c[None]).abs())
SNAP_L = [0, 5, 11, 17]
SNAP_H = [0, 1, 2]


def cka(X, Y):
    X = X.double() - X.double().mean(0); Y = Y.double() - Y.double().mean(0)
    return float((X.T @ Y).pow(2).sum() / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 100)), int(ic.get("batch_size", 33)), int(ic.get("seed", 0))
    maze_size, side_pick, min_side = int(ic.get("maze_size", 8)), str(ic.get("side_pick", "random")), int(ic.get("min_side", 0))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)

    rng = random.Random(seed)
    boards = []
    i = 0
    while i < len(ds) and len(boards) < n_boards:
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == maze_size:
            vs = variants(ms, rng, side_pick, min_side)
            if vs is not None:
                v = vs["side"]
                lab = {k: render_labels(ds, ms, vs[k]["img"], vs[k]["nb"], vs[k]["sp"]) for k in ("none", "side")}
                E = (render_labels(ds, ms, v["img"], v["nb"], v["def_"]) == 1) & (lab["side"] == 0)
                gap = render_labels(ds, ms, v["img"], v["nb"], set(v["wall"])) == 1
                mk = lambda k: dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(vs[k]["img"])))
                boards.append(dict(board=i, fold=len(boards) % 2, E=E, gap=gap, lab_none=lab["none"], none=mk("none"), side=mk("side")))
        i += 1
    logging.info(f"{len(boards)} boards")

    # ---------------------------------------------------------------- models
    models = {}
    pm = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(pm, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    pm = accelerator.unwrap_model(accelerator.prepare(pm)); pm.eval()
    pm.scheduler.set_timesteps(20, device=device)
    models["PaTh"] = ("path", pm, pm.thinker.inner)
    from train_plain_trm import PlainTRM
    for spec in os.environ.get("PLAIN_CKPTS", str(ic.get("plain_ckpts", ""))).split(","):
        if not spec:
            continue
        name, path = spec.split(":", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        m.load_state_dict(ck["ema"]); m = m.to(device).eval()
        models[name] = ("plain", m, m.thinker.inner)
        logging.info(f"[{name}] {path}: step {ck['step']}, noisy={m.noisy}, loss={ck['args'].get('loss', 'ce')}")

    st = {"on": False}

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def make_hook(inner, per):
        def hook(module, args, kwargs, output):
            if not st["on"]:
                return None
            c = st["call"]
            if c == 1:
                inj = args[1] if len(args) > 1 else kwargs["input_injection"]
                st["rec"]["emb"] = (inj.float() - inner.H_init.float().view(1, 1, -1)).half().cpu()
            if c % per != 0:
                st["zl_k"] += 1
                if st["zl_k"] in SNAP_L:
                    st["rec"][f"zL{st['zl_k']}"] = output.half().cpu()
                st["rec"]["zL_last"] = output.half().cpu()
            else:
                st["zh_k"] += 1
                if st["zh_k"] in SNAP_H:
                    st["rec"][f"zH{st['zh_k']}"] = output.half().cpu()
                st["rec"]["zH_last"] = output.half().cpu()
            return None
        return hook

    for name, (kind, m, inner) in models.items():
        inner.L_level.register_forward_pre_hook(pre_hook)
        inner.L_level.register_forward_hook(make_hook(inner, inner.config.L_cycles + 1), with_kwargs=True)

    data = {}
    for name, (kind, m, inner) in models.items():
        for which in ("none", "side"):
            parts = []
            for k0 in range(0, len(boards), bs):
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
                parts.append(st["rec"])
            data[(name, which)] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}         # (N, T, D)
        logging.info(f"recorded {name}")

    E = torch.tensor(np.stack([b["E"] for b in boards]))
    gap = torch.tensor(np.stack([b["gap"] for b in boards]))
    lab = torch.tensor(np.stack([b["lab_none"] for b in boards]))
    F_ = torch.tensor([b["fold"] for b in boards])
    N = len(boards)
    # distance of every token from the opening
    dgap = torch.stack([CHEB[:, g].min(1).values if g.any() else torch.full((T,), 99) for g in gap])        # (N, T)

    def probe_acc(Xn, Xs):
        """loop-vs-no-loop on E tokens and path-vs-off on the unedited maze; 2-fold over boards; returns dicts."""
        out_loop, out_path = {}, {}
        # loop vs no loop
        accs = torch.zeros(N, T); have = torch.zeros(N, T, dtype=torch.bool)
        for f in (0, 1):
            tr, te = F_ == f, F_ != f
            Xtr = torch.cat([Xs[tr][E[tr]], Xn[tr][E[tr]]]).float()
            Ytr = torch.cat([torch.ones(int(E[tr].sum())), torch.zeros(int(E[tr].sum()))])
            w, b = train_linear(Xtr, Ytr, 2000, device)
            w = w.cpu()
            ps = (Xs[te].float() @ w + b) > 0; pn = (Xn[te].float() @ w + b) <= 0
            idx = torch.nonzero(te).flatten()
            for j, bi in enumerate(idx):
                m_ = E[bi]
                accs[bi, m_] = 0.5 * (ps[j][m_].float() + pn[j][m_].float()); have[bi, m_] = True
        for lo, hi, nm in ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")):
            sel = have & (dgap >= lo) & (dgap <= hi)
            out_loop[nm] = (float(accs[sel].mean()) if sel.any() else float("nan"), int(sel.sum()))
        # path vs off
        right, allright = [], []
        for f in (0, 1):
            tr, te = F_ == f, F_ != f
            w, b = train_linear(Xn[tr].flatten(0, 1).float(), lab[tr].flatten(), 2000, device)
            pred = (Xn[te].float() @ w.cpu() + b) > 0
            y = lab[te]; known = y >= 0
            right.append(((pred == (y == 1)) & known).sum() / known.sum())
            allright.append(((pred == (y == 1)) | ~known).all(1).float().mean())
        out_path = (float(torch.stack(right).mean()), float(torch.stack(allright).mean()))
        return out_loop, out_path

    print(f"\n[compare] n{maze_size} side_pick={side_pick}: {N} boards, {int(E.sum())} lollipop tokens; models {list(models)}")
    print("(b) linear probes, 2-fold over boards. loop-vs-no-loop accuracy on lollipop tokens by distance from the opening "
          "(tokens 0 | 1 | 2 | 3-4 | 5+, chance 50%) || path-vs-off on the unedited maze: token accuracy, boards all right")
    res_b = {}
    for name in models:
        for snap in ("emb", "zL5", "zH0"):
            ol, op = probe_acc(data[(name, "none")][snap], data[(name, "side")][snap])
            res_b[(name, snap)] = (ol, op)
            print(f"   {name:9s} {snap:4s} " + " | ".join(f"{a:4.0%} (n={n})" for a, n in ol.values()) + f" || {op[0]:5.1%}, {op[1]:4.0%}")

    print("(c) linear CKA between models (all tokens of both mazes) | CKA of the loop-induced change on lollipop tokens")
    names = list(models)
    snaps = ["emb"] + [f"zL{k}" for k in SNAP_L] + ["zL_last"] + [f"zH{k}" for k in SNAP_H] + ["zH_last"]
    res_c = {}
    for snap in snaps:
        rows = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                A = torch.cat([data[(names[a], w)][snap].flatten(0, 1) for w in ("none", "side")])
                B = torch.cat([data[(names[b], w)][snap].flatten(0, 1) for w in ("none", "side")])
                dA = (data[(names[a], "side")][snap] - data[(names[a], "none")][snap])[E]
                dB = (data[(names[b], "side")][snap] - data[(names[b], "none")][snap])[E]
                v = (cka(A, B), cka(dA, dB)); res_c[(snap, names[a], names[b])] = v
                rows.append(f"{names[a]}~{names[b]} {v[0]:.2f}|{v[1]:.2f}")
        print(f"   {snap:7s} " + "  ".join(rows))
    torch.save(dict(res_b=res_b, res_c=res_c, models=names, boards=[{k: b[k] for k in ("board", "fold", "E", "gap", "lab_none")} for b in boards]),
               out_dir / "compare_models.pt")
    print(f"saved {out_dir}/compare_models.pt")


if __name__ == "__main__":
    main()
