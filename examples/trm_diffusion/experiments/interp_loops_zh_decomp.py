"""TEMPORARY (Claude): where does the z_H update write "remove" for lollipop tokens (plain TRM), and why not in PaTh?

z_H update (calls 7, 14, 21, ...): z_H' = L_level(z_H, z_L), the same two blocks as a z_L update with z_L as the injection.
Readout: linear probe on the last z_H of the unedited mazes (2-fold over boards); also the z_L probe (last z_L) on the z_L
that enters each z_H update. Each z_H update is replayed in float32 and split exactly into
  dilution, injection (= the token's OWN z_L), att0, mlp0, att1, mlp1     (see interp_prune_mech.py)
and the attention pieces into senders (Chebyshev distance on the token grid; sender category: lollipop E, the opening
"gap", the shortest path, other; head). Lollipop maze vs the unedited maze of the same board (same noise), E tokens.
Mazes as interp_loops_first.py (maze_size, side_pick, min_side, seed 0). PaTh: maze noised to t = 70, conditional pass.
Output: <out_dir>/loops_zh_decomp_<kind>.pt, printed summary.
"""
import dataclasses
import logging
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
from interp_prune_mech import CONFIG_DIR, PIECES, T, replay
from interp_prune_mech3 import train_linear

G = 12
_r, _c = torch.arange(T) // G, torch.arange(T) % G
CHEB = torch.maximum((_r[:, None] - _r[None]).abs(), (_c[:, None] - _c[None]).abs())
N_ZH = 4                                                     # z_H updates analysed (calls 7, 14, 21, 28)


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
    kind = str(ic.get("model_kind", "path"))
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    if kind == "plain":
        from train_plain_trm import PlainTRM
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
        thinker = plain.thinker
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ck['step']}")
    else:
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
        model.scheduler.set_timesteps(20, device=device)
        thinker = model.thinker
    inner = thinker.inner
    layers = list(inner.L_level.layers)
    nL, nH = len(layers), layers[0].self_attn.num_heads
    per = inner.config.L_cycles + 1
    cheb = CHEB.to(device)
    dbin = torch.nn.functional.one_hot(cheb, G).float()

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
                boards.append(dict(board=i, fold=len(boards) % 2, E=E, gap=gap, sp=lab["side"] == 1, lab_none=lab["none"],
                                   none=mk("none"), side=mk("side")))
        i += 1
    logging.info(f"{len(boards)} boards")

    st = {"on": False}

    def dotw(x, W):
        return (x.float() * W.view(W.shape[0], *([1] * (x.dim() - 2)), -1)).sum(-1)

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def level_hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        c = st["call"]
        if c % per != 0:
            st["last_zl"] = output.detach().float()
            return None
        k = c // per - 1
        if st["mode"] == "fit":
            st["last_zh"] = output.detach().float()
            return None
        if k >= N_ZH:
            return None
        z_in = args[0] if len(args) > 0 else kwargs["hidden_states"]          # z_H before the update
        inj = args[1] if len(args) > 1 else kwargs["input_injection"]        # z_L
        W = st["WH"]
        z_out, lay, s_all = replay(layers, z_in, inj, kwargs.get("cos_sin"))
        before = dotw(z_in, W)
        pcs = [before * s_all[..., 0] - before, dotw(inj, W) * s_all[..., 0]]
        for d in lay:
            pcs += [dotw(d["A"], W) * d["s_att"][..., 0], dotw(d["M"], W) * d["s_mlp"][..., 0]]
        E = st["E"]
        rec = dict(pieces=torch.stack(pcs, 1).cpu(), zh_before=(before + st["BH"][:, None]).cpu(),
                   zh_after=(dotw(output, W) + st["BH"][:, None]).cpu(),
                   zl_in=(dotw(inj, st["WL"]) + st["BL"][:, None]).cpu(), err=float((dotw(z_out - output.float(), W)).abs().max()))
        src, dist = [], []
        for d in lay:
            contrib = d["a"] * dotw(d["vo"], W)[:, :, None, :] * d["s_att"][:, None, :, :]   # (B, H, i, j)
            ce = contrib * E[:, None, :, None]
            src.append(ce.sum(2).cpu()); dist.append(torch.einsum("bhij,ijd->bhd", ce, dbin).cpu())
        rec["src"] = torch.stack(src, 1); rec["dist"] = torch.stack(dist, 1)                  # (B, nL, H, T / G)
        st["recs"].append(rec)
        return None

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)

    def run(chunk, which, k0, mode):
        if mode != "fit":
            folds = torch.tensor([b["fold"] for b in chunk], device=device)
            st["WH"] = torch.stack(st["pH"][0])[1 - folds]; st["BH"] = torch.tensor(st["pH"][1], device=device)[1 - folds]
            st["WL"] = torch.stack(st["pL"][0])[1 - folds]; st["BL"] = torch.tensor(st["pL"][1], device=device)[1 - folds]
            st["E"] = torch.tensor(np.stack([b["E"] for b in chunk]), device=device).float()
        st.update(on=True, mode=mode, call=0, recs=[])
        try:
            with torch.no_grad():
                smps = [b[which] for b in chunk]
                if kind == "plain":
                    torch.manual_seed(seed + k0)
                    plain.think(torch.stack([s.spatial_conditions for s in smps]).to(device))
                else:
                    batch = AmazeDataset.collate_fn(smps)
                    c_ = model._batch_to_sample(batch, device)
                    empty = batch.spatial_conditions.to(device)
                    noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                    tt = torch.full((len(smps),), 70, device=device, dtype=torch.long)
                    model.forward_with_carry(dataclasses.replace(c_, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        finally:
            st["on"] = False

    XH, XL, Y = [], [], []
    for k0 in range(0, len(boards), bs):
        chunk = boards[k0:k0 + bs]
        run(chunk, "none", k0, "fit")
        XH.append(st["last_zh"].cpu()); XL.append(st["last_zl"].cpu()); Y.append(torch.tensor(np.stack([b["lab_none"] for b in chunk])))
    XH, XL, Y = torch.cat(XH), torch.cat(XL), torch.cat(Y)
    F_ = torch.tensor([b["fold"] for b in boards])
    st["pH"], st["pL"] = ([], []), ([], [])
    for f in (0, 1):
        for X, p in ((XH, st["pH"]), (XL, st["pL"])):
            w_, b_ = train_linear(X[F_ == f].flatten(0, 1), Y[F_ == f].flatten(), int(ic.get("probe_steps", 3000)), device)
            p[0].append(w_.to(device)); p[1].append(float(b_))
    del XH, XL

    out = {"none": [], "side": []}
    for k0 in range(0, len(boards), bs):
        chunk = boards[k0:k0 + bs]
        for which in ("none", "side"):
            run(chunk, which, k0, "rec")
            out[which].append(st["recs"])
    merged = {w: [{key: torch.cat([o[k][key] for o in out[w]]) if key != "err" else max(o[k]["err"] for o in out[w])
                   for key in out[w][0][k]} for k in range(N_ZH)] for w in out}
    torch.save(dict(rec=merged, boards=[{k: b[k] for k in ("board", "fold", "E", "gap", "sp")} for b in boards], kind=kind),
               out_dir / f"loops_zh_decomp_{kind}.pt")

    # ---------------------------------------------------------------- summary
    E = torch.tensor(np.stack([b["E"] for b in boards]))
    gap = torch.tensor(np.stack([b["gap"] for b in boards])); sp = torch.tensor(np.stack([b["sp"] for b in boards]))
    print(f"\n[{kind}] n{maze_size} side_pick={side_pick}: {len(boards)} boards, {int(E.sum())} lollipop tokens; "
          f"replay error {max(merged[w][k]['err'] for w in merged for k in range(N_ZH)):.1e}")
    print("per z_H update (call 7, 14, 21, 28), lollipop tokens, unedited -> loop maze:")
    for k in range(N_ZH):
        n, s = merged["none"][k], merged["side"][k]
        f = lambda r, key: r[key][E].mean().item()
        print(f"  z_H update {k + 1}: z_L probe on the incoming z_L {f(n, 'zl_in'):+.2f} -> {f(s, 'zl_in'):+.2f} "
              f"(path {(n['zl_in'][E] > 0).float().mean():.0%} -> {(s['zl_in'][E] > 0).float().mean():.0%}) | "
              f"z_H probe before {f(n, 'zh_before'):+.2f} -> {f(s, 'zh_before'):+.2f}, after {f(n, 'zh_after'):+.2f} -> {f(s, 'zh_after'):+.2f} "
              f"(path {(n['zh_after'][E] > 0).float().mean():.0%} -> {(s['zh_after'][E] > 0).float().mean():.0%})")
        print("     pieces (mean per lollipop token), unedited | loop | loop − unedited: "
              + ", ".join(f"{p} {n['pieces'][:, q][E].mean():+.2f} | {s['pieces'][:, q][E].mean():+.2f} | {(s['pieces'][:, q][E].mean() - n['pieces'][:, q][E].mean()):+.2f}"
                          for q, p in enumerate(PIECES)))
        dd = (s["dist"] - n["dist"]).sum(2).mean(0)                                        # (nL, G)
        for l in range(nL):
            print(f"     attention layer {l}, loop − unedited by sender distance 0..11 (sum over E, mean per board): "
                  + " ".join(f"{x:+.2f}" for x in dd[l].tolist()))
        ds_ = (s["src"] - n["src"]).sum((1, 2))                                             # (B, T)
        Eo = E & ~gap; spo = sp & ~gap & ~E
        cats = {"E": Eo, "gap": gap, "shortest path": spo, "other": ~(E | gap | sp)}
        print("     attention by sender (loop − unedited): " + ", ".join(f"{c} {(ds_ * m).sum(1).mean():+.2f}" for c, m in cats.items()))
        hd = (s["src"] - n["src"]).sum(3).mean(0)                                          # (nL, H)
        top = sorted(((hd[l, h].item(), l, h) for l in range(nL) for h in range(nH)), key=lambda t: abs(t[0]), reverse=True)[:5]
        print("     heads (loop − unedited): " + ", ".join(f"L{l}H{h} {x:+.2f}" for x, l, h in top))
    print(f"saved {out_dir}/loops_zh_decomp_{kind}.pt")


if __name__ == "__main__":
    main()
