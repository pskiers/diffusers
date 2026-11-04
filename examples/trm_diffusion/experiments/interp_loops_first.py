"""TEMPORARY (Claude): how does the thinker treat the loop part of a maze in its FIRST guess? PaTh thinker or plain TRM.

Mazes: the 8x8 test mazes of check_loops.py (seed 0, same removed walls), variants none / side / route.
  +maze_size=<n> (square test set of that size), +side_pick=longest|shortest (+min_side=<cells>): the side loop from the
  wall giving the longest / shortest cycle instead of a random one (check_loops.variants); +interventions=false: skip 2-3.
Token set E of a loop variant (side or route) = kept by dead-end filling but not on the shortest path (the lollipop, or the
second route); rendered like the training targets. E is compared between the loop variant and the unedited maze ("none")
of the same board, so every difference comes from the removed wall.
Readout: linear path probe on the last z_L of the unedited mazes, 2-fold over the analysed boards (fit on even boards ->
applied to odd boards' mazes and the other way round), logit = w · z_L + b.
PaTh: empty (edited) maze noised to t = 70, conditional pass (as check_loops_calls.py). plain: the maze only.

  1. rec    probe logit of every token at every z_L update; for the first n_upd updates the exact split of each update
            (dilution, injection, att0, mlp0, att1, mlp1; see interp_prune_mech.py) and the attention contributions to
            the E tokens' logits by sender token, layer and head (summed over i in E), also binned by Chebyshev
            distance |i − j| on the 12x12 token grid
  2. mask   attention restricted to radius d (Chebyshev, tokens) in every call <= mask_calls (both z_L and z_H calls)
  3. heads  one head (layer, head) removed (its attention weights set to 0) in every call <= head_calls

Output: <out_dir>/loops_first_<kind>.pt, loops_first_<kind>.png, printed summary.
Run (from examples/trm_diffusion, PYTHONPATH=$PWD:$PWD/experiments):
  python experiments/interp_loops_first.py experiment=amaze_thinker_v2_controlnet painter.checkpoint=... \
    +interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint=... +interprebability_config.checkpoint=none \
    +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet +interprebability_config.out_dir=...
"""
import dataclasses
import logging
import random
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
from factory import build_model
from check_loops import KINDS, render_labels, variants
from interp_prune_mech import CONFIG_DIR, LOGIT_HI, PIECES, T, replay
from interp_prune_mech3 import train_linear
from models.trm.layers import apply_rotary_pos_emb

G = 12
_r, _c = torch.arange(T) // G, torch.arange(T) % G
CHEB = torch.maximum((_r[:, None] - _r[None]).abs(), (_c[:, None] - _c[None]).abs())          # (T, T)
NBIN = G
LOOPS = ["side", "route"]


def attn_custom(attn, x, cos_sin, keep=None, head_off=None):
    """Attention output of `attn` on x (fp32) with an optional (S, S) bool mask of allowed pairs and one head removed."""
    B, S, D = x.shape
    H, hd = attn.num_heads, attn.head_dim
    qkv = torch.nn.functional.linear(x, attn.qkv_proj.weight.float()).view(B, S, H + 2 * attn.num_key_value_heads, hd)
    q, k, v = qkv[:, :, :H], qkv[:, :, H:2 * H], qkv[:, :, 2 * H:]
    if cos_sin is not None:
        q, k = apply_rotary_pos_emb(q, k, cos_sin[0].float(), cos_sin[1].float())
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd ** 0.5
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    a = torch.softmax(s, dim=-1)
    if head_off is not None:
        a = a.clone(); a[:, head_off] = 0
    o = torch.einsum("bhqk,bkhd->bqhd", a, v).reshape(B, S, H * hd)
    return torch.nn.functional.linear(o, attn.o_proj.weight.float())


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards = int(ic.get("num_boards", 100))
    bs, seed, n_upd = int(ic.get("batch_size", 33)), int(ic.get("seed", 0)), int(ic.get("n_upd", 18))
    radii = [int(x) for x in str(ic.get("radii", "1,2,3,5")).split(",")]
    mask_calls_list = [int(x) for x in str(ic.get("mask_calls", "7,336")).split(",")]
    head_calls = int(ic.get("head_calls", 7))
    do_iv = bool(ic.get("interventions", True))
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
        torch.manual_seed(seed)                                   # --noisy variants draw their pure-noise x_t in think()
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ck['step']}, noisy={plain.noisy}, loss={ck['args'].get('loss', 'ce')}")
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
    dbin = torch.nn.functional.one_hot(cheb, NBIN).float()                                       # (T, T, NBIN)

    # ---------------------------------------------------------------- mazes (same walls as check_loops.py)
    rng = random.Random(seed)
    items = []
    i = 0
    while i < len(ds) and len({x["board"] for x in items}) < n_boards:
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == maze_size:
            if True:
                vs = variants(ms, rng, side_pick, min_side)
                if vs is not None:
                    L = {name: {k: render_labels(ds, ms, vs[name]["img"], vs[name]["nb"], cells) for k, cells in
                                (("sp", vs[name]["sp"]), ("def", vs[name]["def_"]))} for name in KINDS}
                    sets = {"sp_none": L["none"]["sp"] == 1}
                    for v in LOOPS:
                        gap = render_labels(ds, ms, vs[v]["img"], vs[v]["nb"], set(vs[v]["wall"])) == 1
                        sets[f"E_{v}"] = (L[v]["def"] == 1) & (L[v]["sp"] == 0)
                        sets[f"gap_{v}"] = gap
                        sets[f"sp_{v}"] = L[v]["sp"] == 1
                    for name in KINDS:
                        lab = L[name]["sp"].astype(np.float32)                                 # 1 path, 0 off, -1 ignored
                        items.append(dict(board=i, variant=name, labels=lab, sets=sets,
                                          smp=dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(vs[name]["img"])))))
        i += 1
    order = sorted({x["board"] for x in items})
    for x in items:
        x["fold"] = order.index(x["board"]) % 2                                                # 2-fold probe split
    logging.info(f"{len(items)} analysed mazes ({len(order)} boards)")

    # ---------------------------------------------------------------- hooks
    st = {"on": False, "mode": None, "call": 0, "zl_k": -1, "iv": None}

    def dot(x):
        """x (B, ..., D) · the probe of each row's held-out fold (st["W"] (B, D)) -> (B, ...)."""
        W = st["W"].view(st["W"].shape[0], *([1] * (x.dim() - 2)), -1)
        return (x.float() * W).sum(-1)

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def level_hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        c = st["call"]
        if c % per == 0:
            if kind == "plain" and st["mode"] != "fit":                                       # plain: its own head on z_H
                st["zh"].append(torch.softmax(inner.lm_head(output).float(), -1)[..., 1].half().cpu())
            return None
        st["zl_k"] += 1
        k = st["zl_k"]
        o = output.float()
        if st["mode"] == "fit":
            st["last_zl"] = o
            return None
        st["logits"].append((dot(o) + st["Bv"][:, None]).half().cpu())
        if st["mode"] == "rec" and k < n_upd:
            z_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
            inj = args[1] if len(args) > 1 else kwargs["input_injection"]
            z_out, lay, s_all = replay(layers, z_in, inj, kwargs.get("cos_sin"))
            before = dot(z_in)
            pcs = [before * s_all[..., 0] - before, dot(inj) * s_all[..., 0]]
            for d in lay:
                pcs += [dot(d["A"]) * d["s_att"][..., 0], dot(d["M"]) * d["s_mlp"][..., 0]]
            st["pieces"].append(torch.stack(pcs, 1).half().cpu())                                # (B, 6, T)
            st["replay_err"] = max(st.get("replay_err", 0.0), float(dot(z_out - o).abs().max()))
            src, dist = {}, {}
            for name, E in st["E"].items():                                                    # E: (B, T) float
                s_l, d_l = [], []
                for d in lay:
                    contrib = d["a"] * dot(d["vo"])[:, :, None, :] * d["s_att"][:, None, :, :]  # (B, H, i, j)
                    ce = contrib * E[:, None, :, None]
                    s_l.append(ce.sum(2))                                                     # (B, H, j)
                    d_l.append(torch.einsum("bhij,ijd->bhd", ce, dbin))                        # (B, H, NBIN)
                src[name] = torch.stack(s_l, 1).half().cpu(); dist[name] = torch.stack(d_l, 1).cpu()
            st["src"].append(src); st["dist"].append(dist)
        return None

    def attn_hook(l):
        def f(module, args, kwargs, output):
            iv = st["iv"]
            if not st["on"] or iv is None or st["call"] > iv["calls"]:
                return None
            if iv.get("layer") is not None and iv["layer"] != l:
                return None
            x = kwargs["hidden_states"].float()
            out = attn_custom(module, x, kwargs.get("cos_sin"), keep=iv.get("keep"), head_off=iv.get("head"))
            return out.to(output.dtype)
        return f

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(attn_hook(l), with_kwargs=True)

    def run(chunk, k0, mode, iv=None, E=None):
        if mode != "fit":
            folds = torch.tensor([x["fold"] for x in chunk], device=device)
            st["W"] = torch.stack(st["probes_w"])[1 - folds]                                       # the probe NOT fit on it
            st["Bv"] = torch.tensor(st["probes_b"], device=device)[1 - folds]
        st.update(on=True, mode=mode, call=0, zl_k=-1, iv=iv, logits=[], pieces=[], src=[], dist=[], zh=[], E=E)
        try:
            with torch.no_grad():
                if kind == "plain":
                    plain.think(torch.stack([x["smp"].spatial_conditions for x in chunk]).to(device))
                else:
                    batch = AmazeDataset.collate_fn([x["smp"] for x in chunk])
                    cond = model._batch_to_sample(batch, device)
                    empty = batch.spatial_conditions.to(device)
                    noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                    tt = torch.full((len(chunk),), 70, device=device, dtype=torch.long)
                    model.forward_with_carry(dataclasses.replace(cond, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        finally:
            st["on"] = False

    # ---------------------------------------------------------------- 1. probe on the last z_L of unedited fit boards
    unedited = [x for x in items if x["variant"] == "none"]
    X, Y, F_ = [], [], []
    for k0 in range(0, len(unedited), bs):
        chunk = unedited[k0:k0 + bs]
        run(chunk, k0, "fit")
        X.append(st["last_zl"].cpu()); Y.append(torch.tensor(np.stack([x["labels"] for x in chunk])))
        F_ += [x["fold"] for x in chunk]
    X, Y, F_ = torch.cat(X), torch.cat(Y), torch.tensor(F_)
    st["probes_w"], st["probes_b"] = [], []
    for f in (0, 1):
        w_, b_ = train_linear(X[F_ == f].flatten(0, 1), Y[F_ == f].flatten(), int(ic.get("probe_steps", 3000)), device)
        st["probes_w"].append(w_.to(device)); st["probes_b"].append(float(b_))
    del X
    logging.info("[fit] 2-fold probes trained")

    def E_masks(chunk):
        return {f"E_{v}": torch.tensor(np.stack([x["sets"][f"E_{v}"] for x in chunk]), dtype=torch.float32, device=device) for v in LOOPS}

    # ---------------------------------------------------------------- 2. record
    for k0 in range(0, len(items), bs):
        chunk = items[k0:k0 + bs]
        run(chunk, k0, "rec", E=E_masks(chunk))
        lg = torch.stack(st["logits"], 1); pc = torch.stack(st["pieces"], 1)
        zh = torch.stack(st["zh"], 1) if st["zh"] else None
        for j, x in enumerate(chunk):
            x["logit"] = lg[j]
            x["zh_p"] = zh[j] if zh is not None else None                                        # plain: (n_zH, T) p(path)
            x["pieces"] = pc[j]                                                                 # (n_upd, 6, T)
            x["src"] = {n: torch.stack([s[n][j] for s in st["src"]]) for n in st["src"][0]}      # (n_upd, nL, H, T)
            x["dist"] = {n: torch.stack([s[n][j] for s in st["dist"]]) for n in st["dist"][0]}  # (n_upd, nL, H, NBIN)
        logging.info(f"[rec] {k0 + len(chunk)}/{len(items)}, replay error {st.get('replay_err', 0):.2e}")

    # ---------------------------------------------------------------- 3. interventions (keep only the logit traces)
    ivs = [("base", None)]
    for mc in (mask_calls_list if do_iv else []):
        for r in radii:
            ivs.append((f"radius{r}_calls{mc}", dict(calls=mc, keep=(cheb <= r))))
    for l in range(nL if do_iv else 0):
        for h in range(nH):
            ivs.append((f"head_L{l}H{h}", dict(calls=head_calls, layer=l, head=h)))
    iv_out = {}
    for name, iv in ivs:
        tr = []
        for k0 in range(0, len(items), bs):
            run(items[k0:k0 + bs], k0, "iv", iv=iv)
            lg = torch.stack(st["logits"], 1)
            tr.append(lg[:, [5, n_upd - 1, lg.shape[1] - 1]])                                   # end of H-cycle 1, n_upd, last
        iv_out[name] = torch.cat(tr)                                                            # (n_items, 3, T)
        logging.info(f"[iv] {name}")

    # ---------------------------------------------------------------- save
    keep_keys = ("board", "variant", "labels", "sets", "logit", "zh_p", "pieces", "src", "dist")
    torch.save(dict(items=[{k: x[k] for k in keep_keys} for x in items], iv=iv_out, iv_names=[n for n, _ in ivs],
                    probes_w=[w_.cpu() for w_ in st["probes_w"]], probes_b=st["probes_b"], kind=kind, n_upd=n_upd), out_dir / f"loops_first_{kind}.pt")

    # ---------------------------------------------------------------- summary
    by = {(x["board"], x["variant"]): x for x in items}
    boards = sorted({x["board"] for x in items})
    U_all = items[0]["logit"].shape[0]
    show = [0, 1, 2, 5, 6, 11, 17, 41, U_all - 1]
    print(f"\n[{kind}] probe on the last z_L; {len(boards)} boards; z_L updates in a run: {U_all}; replay error {st.get('replay_err', 0):.1e}")
    print("1) E tokens (loop part, off in the right answer): share read as path (logit > 0) at z_L updates "
          + ", ".join(map(str, show)) + "; ever confident (p > 0.8) in the first n_upd / in the whole run")
    for v in LOOPS:
        for maze in ("none", v):
            Z = [by[(bd, maze)]["logit"][:, by[(bd, maze)]["sets"][f"E_{v}"]].float() for bd in boards]
            Z = torch.cat([z for z in Z if z.numel()], 1)
            print(f"   E_{v:5s} in maze {maze:5s} n_tok={Z.shape[1]:4d}: " + " ".join(f"{(Z[k] > 0).float().mean():4.0%}" for k in show)
                  + f" | ever {(Z[:n_upd] > LOGIT_HI).any(0).float().mean():4.0%} / {(Z > LOGIT_HI).any(0).float().mean():4.0%}")

    if kind == "plain":
        print("1b) the same with plain's own head on z_H (p > 0.5 / p > 0.8), at z_H updates 1, 2, 3, 6, last")
        for v in LOOPS:
            for maze in ("none", v):
                Z = torch.cat([by[(bd, maze)]["zh_p"][:, by[(bd, maze)]["sets"][f"E_{v}"]].float() for bd in boards], 1)
                print(f"   E_{v:5s} in maze {maze:5s} n_tok={Z.shape[1]:4d}: " + " ".join(f"{(Z[k] > 0.5).float().mean():4.0%}" for k in (0, 1, 2, 5, Z.shape[0] - 1))
                      + f" | ever confident {(Z > 0.8).any(0).float().mean():4.0%}")
    print(f"2) pieces of the z_L updates 0-5 (first H-cycle), mean logit change on E tokens: maze with loop − unedited maze")
    for v in LOOPS:
        P = {m: torch.cat([by[(bd, m)]["pieces"][:6][:, :, by[(bd, m)]["sets"][f"E_{v}"]].float() for bd in boards], 2) for m in ("none", v)}
        print(f"   E_{v}: " + " | ".join(f"{p} " + " ".join(f"{(P[v][k, q] - P['none'][k, q]).mean():+.2f}" for k in range(6))
                                         for q, p in enumerate(PIECES)))
        lt = {m: torch.cat([by[(bd, m)]["logit"][5, by[(bd, m)]["sets"][f"E_{v}"]].float() for bd in boards]) for m in ("none", v)}
        print(f"        logit after update 5: unedited {lt['none'].mean():+.2f}, with loop {lt[v].mean():+.2f}")

    print("3) attention to E tokens, updates 0-5, difference loop − unedited (sum over E tokens, mean per board)")
    for v in LOOPS:
        d_src, d_dist, d_head, cats = [], [], [], {"E": [], "gap": [], "shortest path": [], "other": []}
        for bd in boards:
            a, n = by[(bd, v)], by[(bd, "none")]
            ds_ = (a["src"][f"E_{v}"][:6].float() - n["src"][f"E_{v}"][:6].float()).sum(0)        # (nL, H, T)
            dd_ = (a["dist"][f"E_{v}"][:6] - n["dist"][f"E_{v}"][:6]).sum(0)                      # (nL, H, NBIN)
            d_dist.append(dd_.sum(1)); d_head.append(ds_.sum(2))
            s = a["sets"]; tok = ds_.sum((0, 1))
            gap = torch.tensor(s[f"gap_{v}"]); E = torch.tensor(s[f"E_{v}"]) & ~gap; sp = torch.tensor(s[f"sp_{v}"]) & ~gap & ~E
            cats["gap"].append(tok[gap].sum()); cats["E"].append(tok[E].sum()); cats["shortest path"].append(tok[sp].sum())
            cats["other"].append(tok[~(gap | E | sp)].sum())
        print(f"   E_{v} by sender: " + ", ".join(f"{k} {torch.stack(x).mean():+.2f}" for k, x in cats.items()))
        dd = torch.stack(d_dist).mean(0)
        for l in range(nL):
            print(f"   E_{v} layer {l} by distance 0..{NBIN - 1}: " + " ".join(f"{x:+.2f}" for x in dd[l].tolist()))
        dh = torch.stack(d_head).mean(0)
        top = sorted(((dh[l, h].item(), l, h) for l in range(nL) for h in range(nH)), key=lambda t: abs(t[0]), reverse=True)[:6]
        print(f"   E_{v} heads with the largest difference: " + ", ".join(f"L{l}H{h} {x:+.2f}" for x, l, h in top))

    print(f"4) interventions: share of E tokens read as path after update 5 / at the end; shortest path covered at the end; "
          f"boards exact at the end (all known tokens right)")
    idx = {(x["board"], x["variant"]): n for n, x in enumerate(items)}

    def stats(name, maze, set_key):
        R = iv_out[name]
        e5, eE, spc, ex = [], [], [], []
        for bd in boards:
            n = idx[(bd, maze)]; x = items[n]; S = x["sets"][set_key] if set_key else None
            if S is not None and S.any():
                e5.append((R[n, 0, S] > 0).float().mean()); eE.append((R[n, 2, S] > 0).float().mean())
            sp = x["labels"] == 1; known = x["labels"] >= 0
            spc.append((R[n, 2, sp] > 0).float().mean())
            ex.append(float(((R[n, 2] > 0).numpy() == sp)[known].all()))
        f = lambda z: f"{torch.stack(z).mean():4.0%}" if z else "  - "
        return f"{f(e5)} / {f(eE)}  sp {np.mean([float(z) for z in spc]):4.0%}  exact {np.mean(ex):4.0%}"

    for name, _ in ivs:
        print(f"   {name:18s} none: {stats(name, 'none', None)} | " +
              " | ".join(f"{v}: E {stats(name, v, f'E_{v}')}" for v in LOOPS))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for v, col in zip(LOOPS, ("C1", "C3")):
        for maze, ls in (("none", "--"), (v, "-")):
            Z = torch.cat([by[(bd, maze)]["logit"][:42, by[(bd, maze)]["sets"][f"E_{v}"]].float() for bd in boards], 1)
            ax[0].plot((Z > 0).float().mean(1), ls, color=col, label=f"E_{v} in {'unedited' if maze == 'none' else 'loop'} maze")
    ax[0].set(title="A. E tokens read as path", xlabel="z_L update", ylabel="share"); ax[0].legend(fontsize=7)
    for q, v in enumerate(LOOPS):
        P = {m: torch.cat([by[(bd, m)]["pieces"][:6][:, :, by[(bd, m)]["sets"][f"E_{v}"]].float() for bd in boards], 2) for m in ("none", v)}
        diff = (P[v] - P["none"]).mean(2).sum(0)
        ax[1].bar(np.arange(6) + 0.4 * q, diff.numpy(), 0.4, label=f"E_{v}")
    ax[1].set_xticks(np.arange(6) + 0.2, PIECES, rotation=30); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(title="B. pieces, updates 0-5: loop − unedited (mean per E token)"); ax[1].legend(fontsize=7)
    for v, col in zip(LOOPS, ("C1", "C3")):
        dd = torch.stack([(by[(bd, v)]["dist"][f"E_{v}"][:6] - by[(bd, "none")]["dist"][f"E_{v}"][:6]).sum((0, 2)) for bd in boards]).mean(0)
        for l, ls in zip(range(nL), ("-", ":")):
            ax[2].plot(dd[l].numpy(), ls, color=col, marker="o", ms=3, label=f"E_{v}, layer {l}")
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set(title="C. attention to E by sender distance: loop − unedited",
                                                   xlabel="Chebyshev distance (tokens)"); ax[2].legend(fontsize=7)
    base = iv_out["base"]
    hm = np.zeros((nL, nH))
    for l in range(nL if do_iv else 0):
        for h in range(nH):
            R = iv_out[f"head_L{l}H{h}"]
            vals = [float((R[idx[(bd, 'side')], 0] - base[idx[(bd, 'side')], 0])[items[idx[(bd, 'side')]]["sets"]["E_side"]].float().mean())
                    for bd in boards if items[idx[(bd, 'side')]]["sets"]["E_side"].any()]
            hm[l, h] = np.mean(vals)
    im = ax[3].imshow(hm, cmap="RdBu_r", vmin=-np.abs(hm).max(), vmax=np.abs(hm).max())
    ax[3].set(title=f"D. head removed (calls <= {head_calls}): change of E_side logit after update 5",
              xticks=range(nH), yticks=range(nL), xlabel="head", ylabel="layer"); plt.colorbar(im, ax=ax[3])
    fig.suptitle(f"{kind}: first guess on loop mazes (E = loop part kept by dead-end filling, not on the shortest path)")
    fig.tight_layout(); fig.savefig(out_dir / f"loops_first_{kind}.png", dpi=120)
    print(f"saved {out_dir}/loops_first_{kind}.pt / .png")


if __name__ == "__main__":
    main()
