"""TEMPORARY diagnostic (Claude, amaze interpretability): interp_prune_mech2.py for PaTh's thinker OR a plain TRM.

+interprebability_config.model_kind=path (default): as interp_prune_mech2.py (empty board noised to t = 70).
+interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint=<train_plain_trm.py checkpoint>:
  the plain TRM (EMA weights) reads only the maze; same boards, labels, probes, split, drops and removal check.

v2 of interp_prune_mech.py, adds the z_H route.

Same setup, split of every z_L update, sender maps and drop definition as interp_prune_mech.py (helpers imported).
New:
  - z_H probe (last z_H update): the z_H logit of every token at every z_H update
  - per-call z_L probes (one per z_L update <= c_dec, fit boards): is a drop's peak also "path" for a probe fitted at
    that call (and not only for the final-state probe)?
  - removal check on every side-branch corridor drop whose ★ was off before i crossed (★ crossed first, or ★ never
    read as path): the messages ★ -> i (or ★'s whole cell -> i) replaced by their mean, in z_L updates only / z_H updates
    only / both, from 2 updates before the earlier of ★'s crossing and i's onset to c_stop; controls (both routes):
    other corridor neighbour, wall-covered neighbour, random far token; freeze_zH_i: token i's z_H is not updated
    from one H-cycle before its onset on (the other tokens' z_H is)
  - board-level: the z_H update at call 7 (or 14) skipped (z_H kept as it was) -> new logits of every token
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

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
from factory import build_model
from interp_prune_mech import (CONFIG_DIR, LOGIT_HI, NB, NB_NAMES, OFF, OFFS, PIECES, T, W_PEAK, W_PRE, StopRun,
                               attn_parts, board_geometry, classify, crossing_index, mlp_logit, relation, replay,
                               select_rows, tok_nbrs)

ROUTE = {"zL": 0, "zH": 1, "both": 2}


def token_types(geo):
    """Token geometry (12-px tokens on 144/H-px cells): 'interior' (inside one cell), 'open_link' (straddles an open
    boundary between two cells), 'wall_link' (straddles a closed boundary = a wall between cells), 'corner' (4 cells).
    A boundary counts if the token overlaps both sides by >= 3 px. (A dark-pixel test is not used: most corridor
    tokens touch a side wall.)"""
    ch, cw = 144 / geo["H"], 144 / geo["W"]
    out = []
    for t in range(T):
        r, c = divmod(t, 12)
        y0, x0 = r * 12, c * 12
        by, bx = (int(y0 // ch) + 1) * ch, (int(x0 // cw) + 1) * cw
        sr = by - y0 >= 3 and y0 + 12 - by >= 3
        sc = bx - x0 >= 3 and x0 + 12 - bx >= 3
        a = (int(y0 // ch), int(x0 // cw))
        if sr and sc:
            out.append("corner")
        elif sr or sc:
            b = (a[0] + 1, a[1]) if sr else (a[0], a[1] + 1)
            out.append("open_link" if (a, b) in geo["open_set"] else "wall_link")
        else:
            out.append("interior")
    return np.array(out)


def star_of2(geo, ttype, i, ko, cross_k):
    """★ = a corridor token of the open neighbouring cell nearer the dead-end tip (lower dead-end-filling round), within
    2 token steps of i (with 1.5 tokens per cell, that cell's nearest token is not always adjacent): the one that
    crossed most recently before / at i's onset; else the first to cross later; else (never read as path) the nearest."""
    cell = lambda u: tuple(geo["tok_cell"][u])
    ci = cell(i); ri = geo["rounds"][ci]
    if ri <= 0:
        return None, False
    tip_cells = {b for a, b in geo["open_set"] if a == ci and 0 < geo["rounds"][b] < ri}
    dist = lambda u: abs(u // 12 - i // 12) + abs(u % 12 - i % 12)
    cands = sorted((u for u in range(T) if cell(u) in tip_cells and dist(u) <= 2 and ttype[u] in ("interior", "open_link")),
                   key=dist)
    if not cands:
        return None, False
    before = [u for u in cands if 0 <= cross_k[u] <= ko]
    if before:
        return max(before, key=lambda u: (cross_k[u], -dist(u))), True
    later = [u for u in cands if cross_k[u] >= 0]
    return (min(later, key=lambda u: (cross_k[u], dist(u))) if later else cands[0]), False


def train_linear(X, Y, steps, device):
    keep = Y >= 0
    probe = ZPathEncodingAblation._train_probe(X[keep].float(), Y[keep], use_nonlinearity=False, steps=steps, device=device)
    lin = probe.net
    w = (lin.weight[0] / probe.std).float().detach()
    b = float(lin.bias[0] - (lin.weight[0] / probe.std * probe.mean).sum())
    return w, b


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    K = int(ic.get("template_samples", 70)); N = int(ic.get("num_samples", 30))
    batch_size, seed = int(ic.get("batch_size", 16)), int(ic.get("seed", 0))
    t_start = int(ic.get("t_start", 70))
    c_dec, c_max = int(ic.get("c_dec", 63)), int(ic.get("c_max", 84))
    max_removal = int(ic.get("max_removal", 400)); removal_batch = int(ic.get("removal_batch", 64))
    pc_steps = int(ic.get("percall_probe_steps", 1000))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    kind = str(ic.get("model_kind", "path"))
    if kind == "plain":
        from train_plain_trm import PlainTRM
        ckp = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ckp["args"].get("n_sup", 16)))
        plain.load_state_dict(ckp["ema"])
        plain = plain.to(device).eval()
        thinker = plain.thinker
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ckp['step']}")
    else:
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model))
        model.eval()
        model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
        thinker = model.thinker
    inner = thinker.inner
    layers = list(inner.L_level.layers)
    nL, nH, D = len(layers), layers[0].self_attn.num_heads, inner.config.hidden_size
    if nL != 2:
        raise SystemExit("the piece names assume 2 layers per L_level call")
    per = inner.config.L_cycles + 1
    n_calls = inner.config.H_cycles * per * thinker.n_sup
    zl_calls = [c for c in range(1, n_calls + 1) if c % per != 0]
    zh_calls = [c for c in range(1, n_calls + 1) if c % per == 0]
    n_dec = sum(c <= c_dec for c in zl_calls)
    off_dev = OFF.to(device)
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=int(cfg.data.get("image_size", 144)),
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=int(vd.get("num_channels", 3)), include_metadata=True)
    t70 = torch.tensor(t_start, device=device)
    if kind == "path":
        pe = ZPathEncodingAblation()
        pe.interprebability_config = {"t_start": t_start}
        pe.full_ts, pe.keep_L_indices = model.scheduler.timesteps, None

    def batches(first, last):
        for start in range(first, last, batch_size):
            idxs = list(range(start, min(start + batch_size, last)))
            if kind == "path":
                br = pe._make_runs(model, ds, idxs, seed + start * 977, device)
                if br is not None:
                    conditions, x_t, _ = br.runs["empty"]
                    yield conditions, x_t, br.info
            else:                                   # same usable boards as _make_runs; the maze is the only input
                samples, boards = [], []
                for i in idxs:
                    smp = ds[i]
                    if mcl.build_maze_sample(smp.metadata) is not None:
                        samples.append(smp); boards.append(i)
                if not samples:
                    continue
                b = AmazeDataset.collate_fn(samples)
                maze = b.spatial_conditions.to(device)
                info = {"labels": ZPathEncodingAblation._token_labels(b.images, 12, 0.1), "solutions": b.images.cpu(),
                        "boards": boards}
                yield maze, maze, info

    st = {"on": False, "mode": None, "call": 0, "zl_k": -1, "zh_k": -1, "stop_after": None}

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1
            if st["stop_after"] is not None and st["call"] > st["stop_after"]:
                raise StopRun()

    def level_hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        c = st["call"]
        z_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
        inj = args[1] if len(args) > 1 else kwargs["input_injection"]
        cos_sin = kwargs.get("cos_sin")
        is_zl = c % per != 0
        if is_zl:
            st["zl_k"] += 1
        else:
            st["zh_k"] += 1
        mode = st["mode"]
        if mode == "fit":
            if is_zl:
                st["last_zl"] = output.detach().float()
                if c <= c_dec:
                    st["fit_zl"][:, st["zl_k"]] = output.detach().half().cpu()
            else:
                st["last_zh"] = output.detach().float()
            if c <= c_max:
                _, lay, _ = replay(layers, z_in, inj, cos_sin)
                for l, d in enumerate(lay):
                    for b in range(z_in.shape[0]):
                        m = torch.einsum("hts,hse->tse", d["a"][b], d["vo"][b]).reshape(T * T, -1)
                        st["msum"][l][c].index_add_(0, off_dev.reshape(-1), m)
                if c == 1:
                    st["mcount"] += z_in.shape[0]
        elif mode == "rec":
            o = output.float()
            if is_zl:
                k = st["zl_k"]
                st["logits"][:, k] = (o @ st["w"] + st["b"]).cpu()
                if c <= c_dec:
                    st["logits_pc"][:, k] = (o @ st["Wpc"][k] + st["bpc"][k]).cpu()
                    rec_update(k, z_in, inj, cos_sin, output, c)
            else:
                st["zh_logits"][:, st["zh_k"]] = (o @ st["wH"] + st["bH"]).cpu()
        elif mode == "skip":
            if is_zl:
                st["logits_alt"][:, st["zl_k"]] = (output.float() @ st["w"] + st["b"]).cpu()
            elif c in st["skip"]:
                return z_in                                                       # z_H not updated at this call
        elif mode == "block":
            if is_zl:
                o = output.float()
                st["traj"][:, st["zl_k"]] = (o[torch.arange(o.shape[0], device=o.device), st["tgt"]] @ st["w"] + st["b"]).cpu()
            else:
                fz = st["freeze"]
                sel = (fz["lo"] <= c) & (fz["hi"] >= c)
                if bool(sel.any()):
                    out = output.clone()
                    e, i = fz["e"][sel], fz["i"][sel]
                    out[e, i] = z_in[e, i].to(out.dtype)
                    return out
        return None

    def rec_update(k, z_in, inj, cos_sin, output, c):
        w = st["w"]
        z_out, lay, s_all = replay(layers, z_in, inj, cos_sin)
        before = z_in.float() @ w
        pcs = [before * s_all[..., 0] - before, (inj.float() @ w) * s_all[..., 0]]
        for d in lay:
            pcs += [(d["A"] @ w) * d["s_att"][..., 0], (d["M"] @ w) * d["s_mlp"][..., 0]]
        P_ = torch.stack(pcs, 1)
        st["pieces"][:, k] = P_.cpu()
        st["replay_err"].append(float(((z_out @ w) - (output.float() @ w)).abs().max()))
        st["sum_err"].append(float((before + P_.sum(1) - z_out @ w).abs().max()))
        for l, d in enumerate(lay):
            pv = d["vo"] @ w
            contrib = d["a"] * pv[:, :, None, :] * d["s_att"][:, None, :, :]
            st["maps"][:, k, 2 * l] = contrib.sum(1).half().cpu()
            for n in range(len(NB)):
                st["head_nb"][:, k, l, :, :, n] = contrib[:, :, st["nb_t"], st["nb_s"][n]].mul(st["nb_ok"][n]).half().cpu()
            mb = st["mbar"][l].get(c)
            f_full = (d["M"] @ w) * d["s_mlp"][..., 0]
            for b in range(z_in.shape[0]):
                m = torch.einsum("hts,hse->tse", d["a"][b], d["vo"][b])
                u2 = d["u"][b][:, None, :] - m + (mb[off_dev] if mb is not None else 0)
                f2 = mlp_logit(layers[l], u2, w, d["s_mlp"][b][:, None, 0])
                st["maps"][b, k, 2 * l + 1] = (f_full[b][:, None] - f2).half().cpu()

    def attn_block_hook(l):
        def f(module, args, kwargs, output):
            if not (st["on"] and st["mode"] == "block"):
                return None
            c = st["call"]
            pr = st["pairs"]
            if pr["e"].numel() == 0:
                return None
            is_zh = c % per == 0
            route_ok = (pr["route"] >= 1) if is_zh else (pr["route"] != 1)
            sel = (pr["lo"] <= c) & (pr["hi"] >= c) & route_ok
            if not bool(sel.any()):
                return None
            e, i, j = pr["e"][sel], pr["i"][sel], pr["j"][sel]
            a, vo, _ = attn_parts(module, kwargs["hidden_states"].float(), kwargs.get("cos_sin"))
            msg = (a[e, :, i, j][..., None] * vo[e, :, j, :]).sum(1)
            rep = st["mbar"][l][c][off_dev[i, j]] if c in st["mbar"][l] else torch.zeros_like(msg)
            out = output.float().clone()
            out.index_put_((e, i), rep - msg, accumulate=True)
            return out.to(output.dtype)
        return f

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(attn_block_hook(l), with_kwargs=True)
    nb_t = torch.arange(T, device=device)
    st["nb_t"], st["nb_s"], st["nb_ok"] = nb_t, [], []
    for dr, dc in NB:
        r, c_ = nb_t // 12 + dr, nb_t % 12 + dc
        st["nb_s"].append(r.clamp(0, 11) * 12 + c_.clamp(0, 11))
        st["nb_ok"].append(((r >= 0) & (r < 12) & (c_ >= 0) & (c_ < 12)).float())

    def run_t70(conditions, x_t, mode, stop_after=None):
        st.update(on=True, mode=mode, call=0, zl_k=-1, zh_k=-1, stop_after=stop_after)
        try:
            with torch.no_grad():
                if kind == "plain":
                    plain.think(x_t)
                else:
                    model.forward_with_carry(dataclasses.replace(conditions, x_noisy=x_t, timesteps=t70.expand(x_t.shape[0])))
        except StopRun:
            pass
        finally:
            st["on"] = False

    # ================= 1. fit =================
    st["msum"] = [{c: torch.zeros(OFFS * OFFS, D, device=device) for c in range(1, c_max + 1)} for _ in range(nL)]
    st["mcount"] = 0
    xs, xh, xpc, ys = [], [], [], []
    for conditions, x_t, info in batches(0, K):
        st["fit_zl"] = torch.zeros(x_t.shape[0], n_dec, T, D, dtype=torch.float16)
        run_t70(conditions, x_t, "fit")
        xs.append(st["last_zl"].cpu()); xh.append(st["last_zh"].cpu()); xpc.append(st["fit_zl"]); ys.append(info["labels"])
        logging.info(f"[fit] {st['mcount']} boards")
    pair_count = torch.bincount(OFF.reshape(-1), minlength=OFFS * OFFS).float().to(device) * st["mcount"]
    st["mbar"] = [{c: s / pair_count.clamp_min(1)[:, None] for c, s in st["msum"][l].items()} for l in range(nL)]
    del st["msum"]
    Y = torch.cat(ys).flatten()
    st["w"], st["b"] = train_linear(torch.cat(xs).flatten(0, 1), Y, int(ic.get("probe_steps", 3000)), device)
    st["wH"], st["bH"] = train_linear(torch.cat(xh).flatten(0, 1), Y, int(ic.get("probe_steps", 3000)), device)
    XP = torch.cat(xpc)                                                           # (K, n_dec, T, D)
    Wpc, bpc = [], []
    for k in range(n_dec):
        w_, b_ = train_linear(XP[:, k].flatten(0, 1), Y, pc_steps, device)
        Wpc.append(w_); bpc.append(b_)
    st["Wpc"], st["bpc"] = Wpc, bpc
    del XP, xpc
    logging.info(f"[fit] probes trained (final z_L, final z_H, {n_dec} per-call z_L)")

    # ================= 2. analysis =================
    boards, flips, removal, skips = [], [], [], []
    n_removal = 0
    rng = random.Random(seed)
    st["replay_err"], st["sum_err"] = [], []
    for conditions, x_t, info in batches(K, K + N):
        B = x_t.shape[0]
        st["logits"] = torch.zeros(B, len(zl_calls), T)
        st["logits_pc"] = torch.zeros(B, n_dec, T)
        st["zh_logits"] = torch.zeros(B, len(zh_calls), T)
        st["pieces"] = torch.zeros(B, n_dec, len(PIECES), T)
        st["maps"] = torch.zeros(B, n_dec, 2 * nL, T, T, dtype=torch.float16)
        st["head_nb"] = torch.zeros(B, n_dec, nL, nH, T, len(NB), dtype=torch.float16)
        run_t70(conditions, x_t, "rec")
        n_alt = sum(c <= c_max for c in zl_calls)
        alt = {}
        for name, sk in (("skip_zH7", {7}), ("skip_zH14", {14})):
            st["skip"] = sk
            st["logits_alt"] = torch.full((B, n_alt), float("nan")).unsqueeze(-1).repeat(1, 1, T)
            run_t70(conditions, x_t, "skip", stop_after=c_max)
            alt[name] = st["logits_alt"].numpy().astype(np.float32)
        batch_flips = []
        for bi in range(B):
            ms = mcl.build_maze_sample(ds[info["boards"][bi]].metadata)
            geo = board_geometry(ms)
            ttype = token_types(geo)
            L = st["logits"][bi].numpy()
            cross_k = np.array([crossing_index(L[:, t]) for t in range(T)])
            bid = len(boards)
            boards.append(dict(board=info["boards"][bi], labels=info["labels"][bi].numpy(), logits=L.astype(np.float32),
                               logits_pc=st["logits_pc"][bi].numpy().astype(np.float32),
                               zh_logits=st["zh_logits"][bi].numpy().astype(np.float32),
                               pieces=st["pieces"][bi].numpy().astype(np.float32), cross_k=cross_k,
                               sol=(info["solutions"][bi].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8),
                               ttype=ttype, **{k_: v for k_, v in geo.items() if k_ != "open_set"},
                               open_pairs=sorted(geo["open_set"])))
            skips.append({name: a[bi] for name, a in alt.items()})
            for t in range(T):
                kx = int(cross_k[t])
                if kx <= 0 or L[:kx, t].max() <= LOGIT_HI or L[kx:, t].min() >= -LOGIT_HI:
                    continue
                lo = max(0, kx - W_PEAK)
                ko = lo + int(np.argmax(L[lo:kx, t])) + 1
                f = dict(bid=bid, token=t, ko=ko, kx=kx, onset_call=zl_calls[ko], cross_call=zl_calls[kx],
                         late=zl_calls[kx] > c_dec, label=int(info["labels"][bi, t]),
                         ttype=str(ttype[t]), round=int(geo["rounds"][tuple(geo["tok_cell"][t])]))
                f["corridor"] = f["ttype"] in ("interior", "open_link") and f["round"] > 0
                star, star_before = star_of2(geo, ttype, t, ko, cross_k)
                f.update(star=star, star_before=star_before, star_cross_k=(int(cross_k[star]) if star is not None else None))
                if not f["late"]:
                    sm = st["maps"][bi, :, :, t, :].float().numpy()
                    f["category"] = classify(f, L[:, t], cross_k, boards[-1]["pieces"][:, :, t], sm, geo, zl_calls, per)
                    f["maps"] = st["maps"][bi, :, :, t, :].clone()
                    f["head_nb"] = st["head_nb"][bi, :, :, :, t, :].clone()
                else:
                    f["category"] = None
                nbs = tok_nbrs(t)
                rd = lambda u: geo["rounds"][tuple(geo["tok_cell"][u])]
                f["other_nb"] = next((u for u in nbs if relation(geo, t, u) == "open" and u != star and
                                      (rd(u) == 0 or rd(u) > rd(t))), None)
                f["wall_nb"] = next((u for u in nbs if u not in (star, f["other_nb"])
                                     and (relation(geo, t, u) == "wall" or ttype[u] == "wall_link")), None)
                f["far"] = rng.choice([u for u in range(T) if max(abs(u // 12 - t // 12), abs(u % 12 - t % 12)) >= 4])
                f["star_cell"] = ([u for u in range(T) if tuple(geo["tok_cell"][u]) == tuple(geo["tok_cell"][star])]
                                  if star is not None else [])
                f["bi"] = bi
                batch_flips.append(f)
        logging.info(f"[analysis] boards {len(boards)}: {len(batch_flips)} drops in this batch")

        # ---- removal check ----
        exps = []
        for f in batch_flips:
            sck = f["star_cross_k"]
            ok = (not f["late"] and f["corridor"] and f["star"] is not None
                  and (sck == -1 or (sck is not None and 0 <= sck <= f["kx"])))
            if not ok or n_removal >= max_removal:
                continue
            n_removal += 1
            k0 = (min(sck, f["ko"]) if sck >= 0 else f["ko"]) - 2
            lo = zl_calls[max(0, k0)]
            hi = min(c_max, zl_calls[f["kx"]] + 21)
            lo_f = max(1, f["onset_call"] - per)
            conds = [("none", [], "both", False), ("star_zL", [f["star"]], "zL", False), ("star_zH", [f["star"]], "zH", False),
                     ("star_both", [f["star"]], "both", False), ("cell_zL", f["star_cell"], "zL", False),
                     ("cell_zH", f["star_cell"], "zH", False), ("cell_both", f["star_cell"], "both", False),
                     ("other_both", [f["other_nb"]], "both", False), ("wall_both", [f["wall_nb"]], "both", False),
                     ("far_both", [f["far"]], "both", False), ("freeze_zH_i", [], "both", True)]
            for name, J, route, frz in conds:
                if any(j is None for j in J):
                    continue
                exps.append(dict(f=f, cond=name, J=J, route=ROUTE[route], freeze=frz, lo=(lo_f if frz else lo), hi=hi))
        for s0 in range(0, len(exps), removal_batch):
            chunk = exps[s0:s0 + removal_batch]
            idx = torch.tensor([e["f"]["bi"] for e in chunk], device=device)
            pe_, pi_, pj_, plo, phi, prt = [], [], [], [], [], []
            fe, fi, flo, fhi = [], [], [], []
            for e_i, e in enumerate(chunk):
                for j in e["J"]:
                    pe_.append(e_i); pi_.append(e["f"]["token"]); pj_.append(j); plo.append(e["lo"]); phi.append(e["hi"]); prt.append(e["route"])
                if e["freeze"]:
                    fe.append(e_i); fi.append(e["f"]["token"]); flo.append(e["lo"]); fhi.append(e["hi"])
            tt = lambda v: torch.tensor(v, device=device, dtype=torch.long)
            st["pairs"] = dict(e=tt(pe_), i=tt(pi_), j=tt(pj_), lo=tt(plo), hi=tt(phi), route=tt(prt))
            st["freeze"] = dict(e=tt(fe), i=tt(fi), lo=tt(flo), hi=tt(fhi))
            st["tgt"] = tt([e["f"]["token"] for e in chunk])
            st["traj"] = torch.full((len(chunk), len(zl_calls)), float("nan"))
            cond_sel = conditions[idx] if kind == "plain" else select_rows(conditions, idx, B)
            run_t70(cond_sel, x_t[idx], "block", stop_after=max(e["hi"] for e in chunk))
            for e_i, e in enumerate(chunk):
                n_k = sum(c <= e["hi"] for c in zl_calls)
                tr = st["traj"][e_i, :n_k].numpy()
                removal.append(dict(bid=e["f"]["bid"], token=e["f"]["token"], cond=e["cond"], kx=e["f"]["kx"],
                                    kx_new=crossing_index(tr), n_k=n_k, c_stop=e["hi"], start_call=e["lo"],
                                    traj=tr.astype(np.float32)))
        logging.info(f"[removal] {len(exps)} runs, {n_removal} drops so far")
        flips += [{k_: v for k_, v in f.items() if k_ != "bi"} for f in batch_flips]

    out = dict(model_kind=kind, cfg=dict(K=K, N=N, t_start=t_start, c_dec=c_dec, c_max=c_max, W_PEAK=W_PEAK, W_PRE=W_PRE,
                        logit_hi=LOGIT_HI, per=per),
               zl_calls=zl_calls, zh_calls=zh_calls, n_dec=n_dec, pieces_names=PIECES, nb_names=NB_NAMES,
               w=st["w"].cpu(), b=st["b"], logit_init=float(inner.L_init.float() @ st["w"] + st["b"]),
               boards=boards, flips=flips, removal=removal, skips=skips,
               replay_err=np.array(st["replay_err"]), sum_err=np.array(st["sum_err"]))
    torch.save(out, out_dir / "prune_mech2.pt")
    print(f"[{kind}] saved {out_dir / 'prune_mech2.pt'}: {len(boards)} boards, {len(flips)} drops, {len(removal)} removal runs; "
          f"replay |Δlogit| max {out['replay_err'].max():.3g}, split sum error max {out['sum_err'].max():.3g}", flush=True)
    try:
        import interp_prune_plots2 as pp
        pp.make_all(out_dir / "prune_mech2.pt", out_dir)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print("plotting failed:", repr(ex), flush=True)


if __name__ == "__main__":
    main()
