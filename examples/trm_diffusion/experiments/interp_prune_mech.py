"""TEMPORARY diagnostic (Claude, amaze interpretability): how and why the thinker drops tokens while pruning.

Setup = the path probe (experiments/interpretability/z_path_encoding_ablation.py, helios job 24846271): empty boards
(maze only) noised to t = 70, conditional pass of the first denoising step; boards [0, K) fit, [K, K + N) analysed,
same batches / seeds as base_ablation._batches; token labels = _token_labels (path >= 10% of the token's pixels).
Calls are numbered 1..336 (z_H at 7, 14, ...); "update k" = the k-th z_L update (0-based).

Fit (boards [0, K)):
  - linear path probe on the last z_L update (ZPathEncodingAblation._train_probe): logit = w · z_L + b
  - mean attention message per (layer, call, offset j − i) over all token pairs, calls <= c_max (z_L and z_H calls):
    the replacement for "j sent nothing" in the MLP reaction and in the removal check
Analysis (boards [K, K + N)), every z_L update with call <= c_dec, replayed in float32 from its recorded inputs:
    h0 = z_L(c−1) + (z_H + emb);  h1 = (h0 + att0)/r1;  h2 = (h1 + mlp0)/r2;  h3 = (h2 + att1)/r3;  z_L(c) = (h3 + mlp1)/r4
  => logit(c) − logit(c−1) = dilution + injection + att0 + mlp0 + att1 + mlp1   (exact for the replay)
       dilution = w·z_L(c−1) (1/(r1 r2 r3 r4) − 1)        injection = w·(z_H + emb) / (r1 r2 r3 r4)
       att0 = w·att0 / (r1 r2 r3 r4)   mlp0 = w·mlp0 / (r2 r3 r4)   att1 = w·att1 / (r3 r4)   mlp1 = w·mlp1 / r4
  sender maps (per layer l, target i, sender j):
       att_l[i, j] = Σ_h a_ij^h (w · W_O^h v_j^h) × scale(att_l)_i                 (sums exactly to the att_l piece)
       mlp_l[i, j] = f_l(u_i) − f_l(u_i − m_ij + m̄_l,c(j − i)),   f_l(u) = w · MLP_l(norm(u)) × later scales (fixed)
                     u_i = token i after attention, before the norm;  m_ij = Σ_h a_ij^h W_O^h v_j^h
Flips (strict): logit > +1.39 (p 0.8) at some update, < −1.39 later, and < 0 from the crossing update to the last z_L
update. onset = the update after the highest logit among the W_PEAK updates before the crossing.
Reasons (rules at the onset, in this order; see classify): neighbour, z_H, walls, fade, unclear.
Removal check: for flips whose tip-side neighbour (★) crossed before i's onset, rerun the thinker with the messages
of a source set J to token i replaced by their mean (all heads, both layers) at every call from ★'s crossing to c_stop.

Run (from examples/trm_diffusion, PYTHONPATH=$PWD): python experiments/interp_prune_mech.py experiment=amaze_thinker_v2_controlnet
  painter.checkpoint=... +interprebability_config.checkpoint=... +interprebability_config.data_parquet=...
  +interprebability_config.out_dir=runs/amaze_interp_v33   [num_samples=30 template_samples=70 c_dec=63 c_max=84 ...]
"""
import dataclasses
import logging
import os
import random
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
from factory import build_model
from models.trm.layers import apply_rotary_pos_emb

CONFIG_DIR = os.environ.get("TRM_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "configs"))
G, P, IMG = 12, 12, 144
T = G * G
LOGIT_HI = 1.3863                         # p = 0.8
W_PEAK, W_PRE = 6, 3                      # updates searched for the peak before a crossing / baseline updates before onset
NB = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]   # self, left, right, up, down (token offsets)
NB_NAMES = ["self", "left", "right", "up", "down"]
PIECES = ["dilution", "injection", "att0", "mlp0", "att1", "mlp1"]
CATS = ["neighbour", "z_H", "walls", "fade", "unclear"]
OFFS = 2 * G - 1
_dr = torch.arange(T) // G
_dc = torch.arange(T) % G
OFF = ((_dr[None, :] - _dr[:, None] + G - 1) * OFFS + (_dc[None, :] - _dc[:, None] + G - 1))   # (i, j) -> offset index


class StopRun(Exception):
    pass


# ---------------------------------------------------------------- replay of one L_level call in float32

def rms(y, eps):
    return torch.sqrt(y.pow(2).mean(-1, keepdim=True) + eps)


def attn_parts(attn, x, cos_sin):
    """x (B, S, D) fp32 -> a (B, H, S, S) attention probs, vo (B, H, S, D) = W_O^h v_j^h, A (B, S, D) attention output."""
    B, S, D = x.shape
    H, hd = attn.num_heads, attn.head_dim
    qkv = F.linear(x, attn.qkv_proj.weight.float()).view(B, S, H + 2 * attn.num_key_value_heads, hd)
    q, k, v = qkv[:, :, :H], qkv[:, :, H:2 * H], qkv[:, :, 2 * H:]
    if cos_sin is not None:
        q, k = apply_rotary_pos_emb(q, k, cos_sin[0].float(), cos_sin[1].float())
    a = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q, k) / hd ** 0.5, dim=-1)
    Wo = attn.o_proj.weight.float().view(D, H, hd)
    vo = torch.einsum("bshd,ehd->bhse", v, Wo)
    A = torch.einsum("bhts,bhse->bte", a, vo)
    return a, vo, A


def replay(layers, z_in, inj, cos_sin):
    """Float32 replay of z_out = L_level(z_in, inj). Returns z_out and the per-layer intermediates + norm values."""
    eps = layers[0].norm_eps
    x = z_in.float() + inj.float()
    lay, norms = [], []
    for layer in layers:
        a, vo, A = attn_parts(layer.self_attn, x, cos_sin)
        u = x + A
        r_a = rms(u, eps); x = u / r_a
        M = layer.mlp(x)
        y = x + M
        r_m = rms(y, eps); x = y / r_m
        lay.append(dict(a=a, vo=vo, A=A, u=u, M=M))
        norms += [r_a, r_m]
    # scale of a vector added before norm n in the final state = 1 / (product of norms n, n+1, ...)
    suffix = [None] * len(norms)
    acc = torch.ones_like(norms[0])
    for n in range(len(norms) - 1, -1, -1):
        acc = acc * norms[n]
        suffix[n] = 1.0 / acc
    for l, d in enumerate(lay):
        d["s_att"], d["s_mlp"] = suffix[2 * l], suffix[2 * l + 1]
    return x, lay, suffix[0]


def mlp_logit(layer, u, w, s_mlp):
    """f(u) = w · MLP(norm(u)) × s_mlp for pre-norm vectors u (..., D); s_mlp broadcastable to u[..., 0]."""
    x = u / rms(u, layer.norm_eps)
    return (layer.mlp(x) @ w) * s_mlp


# ---------------------------------------------------------------- board geometry

def board_geometry(ms):
    """Cells (r, c), openness, dead-end-filling rounds, token -> cell, wall tokens."""
    H, W = ms.height, ms.width
    is_open = lambda a, b: mcl._is_open(ms.morig, a[0], a[1], b[0], b[1], ms.cell_px)
    nbrs = {}
    for r in range(H):
        for c in range(W):
            nbrs[(r, c)] = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            if 0 <= r + dr < H and 0 <= c + dc < W and is_open((r, c), (r + dr, c + dc))]
    ends = {ms.id2rc[ms.path[0]], ms.id2rc[ms.path[-1]]}
    alive, rnd, k = set(nbrs), {}, 0
    while True:                                # dead-end filling: round 1 = tips, 0 = never removed (solution)
        k += 1
        dead = [c for c in alive if c not in ends and sum(b in alive for b in nbrs[c]) <= 1]
        if not dead:
            break
        for c in dead:
            rnd[c] = k
        alive -= set(dead)
    rounds = np.zeros((H, W), np.int64)
    for c, v in rnd.items():
        rounds[c] = v
    ch, cw = IMG / H, IMG / W
    tok_cell = np.array([(min(H - 1, int((t // G * P + P / 2) / ch)), min(W - 1, int((t % G * P + P / 2) / cw))) for t in range(T)])
    m144 = cv2.resize(ms.morig, (IMG, IMG), interpolation=cv2.INTER_AREA)
    dark = (m144.max(axis=2) < 100).astype(np.float32)
    wall_tok = dark.reshape(G, P, G, P).mean(axis=(1, 3)).reshape(-1) > 0.08
    walls = []                                 # wall segments in token coordinates (x0, y0, x1, y1)
    for r in range(H):
        for c in range(W):
            if c + 1 < W and (r, c + 1) not in nbrs[(r, c)]:
                walls.append(((c + 1) * cw / P - 0.5, r * ch / P - 0.5, (c + 1) * cw / P - 0.5, (r + 1) * ch / P - 0.5))
            if r + 1 < H and (r + 1, c) not in nbrs[(r, c)]:
                walls.append((c * cw / P - 0.5, (r + 1) * ch / P - 0.5, (c + 1) * cw / P - 0.5, (r + 1) * ch / P - 0.5))
    open_set = {(a, b) for a, bs in nbrs.items() for b in bs}
    return dict(H=H, W=W, rounds=rounds, tok_cell=tok_cell, wall_tok=wall_tok, walls=walls, open_set=open_set,
                maze144=m144, ends=sorted(ends))


def tok_nbrs(t):
    r, c = divmod(t, G)
    return [(r + dr) * G + (c + dc) for dr, dc in NB[1:] if 0 <= r + dr < G and 0 <= c + dc < G]


def relation(geo, i, u):
    a, b = tuple(geo["tok_cell"][i]), tuple(geo["tok_cell"][u])
    if a == b:
        return "same"
    return "open" if (a, b) in geo["open_set"] else "wall"


def crossing_index(L):
    """L (n_upd,) logits -> index of the update from which the logit stays < 0; -2 if it ends >= 0, -1 if never >= 0."""
    nonneg = np.flatnonzero(L >= 0)
    if len(nonneg) == 0:
        return -1
    if nonneg[-1] == len(L) - 1:
        return -2
    return int(nonneg[-1] + 1)


# ---------------------------------------------------------------- reasons

def classify(f, L, cross_k, pieces, smaps, geo, zl_calls, per):
    """Reason category of one flip from its window (onset..crossing) of pieces and sender maps (see module docstring)."""
    i, ko, kx = f["token"], f["ko"], f["kx"]
    win = list(range(ko, kx + 1))
    Pw = pieces[win].sum(0)                                         # (6,)
    fall = float(L[kx] - L[ko - 1])
    tot = smaps[:, :, :].sum(1)                                     # (n_dec, S): total push of each sender per update
    S_raw = tot[win].sum(0)
    pre = list(range(max(0, ko - W_PRE), ko))
    base = tot[pre].mean(0) if pre else np.zeros(T)
    S_chg = S_raw - len(win) * base
    j = int(np.argmin(S_chg))
    rel = relation(geo, i, j) if j in tok_nbrs(i) else ("self" if j == i else "far")
    f.update(pieces_win=Pw, fall=fall, S_raw=S_raw, S_chg=S_chg, jstar=j, jstar_rel=rel)
    # 1. neighbour: the biggest new "off" push comes from a corridor neighbour (open, other cell) that crossed within
    #    the 2 updates before the onset (or at it)
    if rel == "open" and cross_k[j] >= 0 and ko - 2 <= cross_k[j] <= ko:
        return "neighbour"
    # 2. z_H: the onset is the first z_L update after a z_H update and the injection is the biggest push down
    if zl_calls[ko] % per == 1 and zl_calls[ko] > 1 and int(np.argmin(Pw)) == PIECES.index("injection"):
        return "z_H"
    # 3. walls: no corridor neighbour crossed by the onset, and the cell is a dead-end tip or the biggest raw push
    #    comes from a wall-covered token
    nb_open = [u for u in tok_nbrs(i) if relation(geo, i, u) == "open"]
    crossed = [u for u in nb_open if 0 <= cross_k[u] <= ko]
    cell = tuple(geo["tok_cell"][i])
    if not crossed and (geo["rounds"][cell] == 1 or geo["wall_tok"][int(np.argmin(S_raw))]):
        return "walls"
    # 4. fade: dilution is the biggest push down and no single sender carries >= 25% of the fall
    if int(np.argmin(Pw)) == PIECES.index("dilution") and abs(S_chg[j]) < 0.25 * abs(fall):
        return "fade"
    return "unclear"


def star_of(geo, i, ko, cross_k):
    """★ = 4-neighbour token in an open neighbouring cell closer to the dead-end tip (lower dead-end-filling round):
    the one that crossed most recently before / at i's onset, else the first to cross. None for tips / solution cells."""
    cell = tuple(geo["tok_cell"][i])
    ri = geo["rounds"][cell]
    if ri <= 0:
        return None, False
    cands = [u for u in tok_nbrs(i) if relation(geo, i, u) == "open"
             and 0 < geo["rounds"][tuple(geo["tok_cell"][u])] < ri]
    if not cands:
        return None, False
    before = [u for u in cands if 0 <= cross_k[u] <= ko]
    if before:
        return max(before, key=lambda u: cross_k[u]), True
    later = [u for u in cands if cross_k[u] >= 0]
    return (min(later, key=lambda u: cross_k[u]) if later else cands[0]), False


# ---------------------------------------------------------------- main

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    K = int(ic.get("template_samples", 70)); N = int(ic.get("num_samples", 30))
    batch_size, seed = int(ic.get("batch_size", 16)), int(ic.get("seed", 0))
    t_start = int(ic.get("t_start", 70))
    c_dec, c_max = int(ic.get("c_dec", 63)), int(ic.get("c_max", 84))
    mlp_ablation = str(ic.get("mlp_ablation", "mean"))
    max_removal = int(ic.get("max_removal", 200)); removal_batch = int(ic.get("removal_batch", 64))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    full_ts = model.scheduler.timesteps
    inner = model.thinker.inner
    layers = list(inner.L_level.layers)
    nL, nH = len(layers), layers[0].self_attn.num_heads
    per = inner.config.L_cycles + 1
    n_calls = inner.config.H_cycles * per * model.thinker.n_sup
    zl_calls = [c for c in range(1, n_calls + 1) if c % per != 0]
    n_dec = sum(c <= c_dec for c in zl_calls)
    off_dev = OFF.to(device)

    data_parquet = str(ic.data_parquet)
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=data_parquet, split="test", image_size=int(cfg.data.get("image_size", IMG)),
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=int(vd.get("num_channels", 3)), include_metadata=True)
    pe = ZPathEncodingAblation()
    pe.interprebability_config = {"t_start": t_start}
    pe.full_ts, pe.keep_L_indices = full_ts, None
    t70 = torch.tensor(t_start, device=device)

    def batches(first, last):
        for start in range(first, last, batch_size):
            base_seed = seed + start * 977
            br = pe._make_runs(model, ds, list(range(start, min(start + batch_size, last))), base_seed, device)
            if br is not None:
                conditions, x_t, _ = br.runs["empty"]
                yield conditions, x_t, br.info

    # ---- hooks ----
    st = {"on": False, "mode": None, "call": 0, "zl_k": -1, "stop_after": None}

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
        if st["mode"] == "fit":
            if is_zl:
                st["last_zl"] = output.detach().float()
            if c <= c_max:
                with torch.no_grad():
                    _, lay, _ = replay(layers, z_in, inj, cos_sin)
                    for l, d in enumerate(lay):
                        for b in range(z_in.shape[0]):
                            m = torch.einsum("hts,hse->tse", d["a"][b], d["vo"][b]).reshape(T * T, -1)
                            st["msum"][l][c].index_add_(0, off_dev.reshape(-1), m)
                st["mcount"] += z_in.shape[0] if c == 1 else 0
        elif st["mode"] == "rec" and is_zl:
            k = st["zl_k"]
            st["logits"][:, k] = (output.float() @ st["w"] + st["b"]).cpu()
            if c <= c_dec:
                with torch.no_grad():
                    rec_update(k, z_in, inj, cos_sin, output, c)
        elif st["mode"] == "block" and is_zl:
            o = output.float()
            st["traj"][:, st["zl_k"]] = (o[torch.arange(o.shape[0], device=o.device), st["tgt"]] @ st["w"] + st["b"]).cpu()
        return None

    def rec_update(k, z_in, inj, cos_sin, output, c):
        w = st["w"]
        z_out, lay, s_all = replay(layers, z_in, inj, cos_sin)
        before = z_in.float() @ w
        own = before * s_all[..., 0]
        pcs = [own - before, (inj.float() @ w) * s_all[..., 0]]
        for d in lay:
            pcs += [(d["A"] @ w) * d["s_att"][..., 0], (d["M"] @ w) * d["s_mlp"][..., 0]]
        st["pieces"][:, k] = torch.stack(pcs, 1).cpu()                                   # (B, 6, T)
        st["replay_err"].append(float(((z_out @ w) - (output.float() @ w)).abs().max()))
        st["sum_err"].append(float((before + torch.stack(pcs, 1).sum(1) - z_out @ w).abs().max()))
        B = z_in.shape[0]
        for l, d in enumerate(lay):
            pv = d["vo"] @ w                                                              # (B, H, S)
            contrib = d["a"] * pv[:, :, None, :] * d["s_att"][:, None, :, :]             # (B, H, T, S)
            st["maps"][:, k, 2 * l] = contrib.sum(1).half().cpu()
            for n, (dr, dc) in enumerate(NB):
                st["head_nb"][:, k, l, :, :, n] = contrib[:, :, st["nb_t"], st["nb_s"][n]].mul(st["nb_ok"][n]).half().cpu()
            mb = st["mbar"][l][c] if (mlp_ablation == "mean" and c in st["mbar"][l]) else None
            f_full = (d["M"] @ w) * d["s_mlp"][..., 0]                                    # (B, T)
            for b in range(B):
                m = torch.einsum("hts,hse->tse", d["a"][b], d["vo"][b])                   # (T, S, D)
                u2 = d["u"][b][:, None, :] - m
                if mb is not None:
                    u2 = u2 + mb[off_dev]
                f2 = mlp_logit(layers[l], u2, w, d["s_mlp"][b][:, None, 0])               # (T, S)
                st["maps"][b, k, 2 * l + 1] = (f_full[b][:, None] - f2).half().cpu()

    def attn_block_hook(l):
        def f(module, args, kwargs, output):
            if not (st["on"] and st["mode"] == "block"):
                return None
            c = st["call"]
            pr = st["pairs"]
            sel = (pr["lo"] <= c) & (pr["hi"] >= c) & ((c % per != 0) | pr["zh"])
            if not bool(sel.any()):
                return None
            e, i, j = pr["e"][sel], pr["i"][sel], pr["j"][sel]
            x = kwargs["hidden_states"].float()
            a, vo, _ = attn_parts(module, x, kwargs.get("cos_sin"))
            msg = (a[e, :, i, j][..., None] * vo[e, :, j, :]).sum(1)                      # (P, D)
            rep = st["mbar"][l][c][off_dev[i, j]] if c in st["mbar"][l] else torch.zeros_like(msg)
            out = output.float().clone()
            out.index_put_((e, i), rep - msg, accumulate=True)
            return out.to(output.dtype)
        return f

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(attn_block_hook(l), with_kwargs=True)
    # neighbour index tables for head_nb
    nb_t = torch.arange(T, device=device)
    st["nb_t"] = nb_t
    st["nb_s"], st["nb_ok"] = [], []
    for dr, dc in NB:
        r, c_ = nb_t // G + dr, nb_t % G + dc
        ok = (r >= 0) & (r < G) & (c_ >= 0) & (c_ < G)
        st["nb_s"].append((r.clamp(0, G - 1) * G + c_.clamp(0, G - 1)))
        st["nb_ok"].append(ok.float())

    def run_t70(conditions, x_t, mode, stop_after=None):
        s = dataclasses.replace(conditions, x_noisy=x_t, timesteps=t70.expand(x_t.shape[0]))
        st.update(on=True, mode=mode, call=0, zl_k=-1, stop_after=stop_after)
        try:
            with torch.no_grad():
                model.forward_with_carry(s)
        except StopRun:
            pass
        finally:
            st["on"] = False

    # ================= 1. fit: probe + mean messages =================
    D = inner.config.hidden_size
    st["msum"] = [{c: torch.zeros(OFFS * OFFS, D, device=device) for c in range(1, c_max + 1)} for _ in range(nL)]
    st["mcount"] = 0
    xs, ys = [], []
    for conditions, x_t, info in batches(0, K):
        run_t70(conditions, x_t, "fit")
        xs.append(st["last_zl"].cpu()); ys.append(info["labels"])
        logging.info(f"[fit] {st['mcount']} boards")
    pair_count = torch.bincount(OFF.reshape(-1), minlength=OFFS * OFFS).float().to(device) * st["mcount"]
    st["mbar"] = [{c: s / pair_count.clamp_min(1)[:, None] for c, s in st["msum"][l].items()} for l in range(nL)]
    del st["msum"]
    X = torch.cat(xs).flatten(0, 1); Y = torch.cat(ys).flatten()
    keep = Y >= 0
    probe = ZPathEncodingAblation._train_probe(X[keep], Y[keep], use_nonlinearity=False,
                                               steps=int(ic.get("probe_steps", 3000)), device=device)
    lin = probe.net
    w = (lin.weight[0] / probe.std).float().detach()
    b = float(lin.bias[0] - (lin.weight[0] / probe.std * probe.mean).sum())
    if nL != 2:
        raise SystemExit("the piece names assume 2 layers per L_level call")
    st["w"], st["b"] = w.detach(), b
    logging.info(f"[fit] probe trained on {int(keep.sum())} tokens of {st['mcount']} boards")

    # ================= 2. analysis =================
    boards, flips, removal = [], [], []
    n_removal = 0
    rng = random.Random(seed)
    st["replay_err"], st["sum_err"] = [], []
    for conditions, x_t, info in batches(K, K + N):
        B = x_t.shape[0]
        st["logits"] = torch.zeros(B, len(zl_calls), T)
        st["pieces"] = torch.zeros(B, n_dec, len(PIECES), T)
        st["maps"] = torch.zeros(B, n_dec, 2 * nL, T, T, dtype=torch.float16)
        st["head_nb"] = torch.zeros(B, n_dec, nL, nH, T, len(NB), dtype=torch.float16)
        run_t70(conditions, x_t, "rec")
        logits0 = float((inner.L_init.float() @ w) + b)
        batch_flips = []
        for bi in range(B):
            ms = mcl.build_maze_sample(ds[info["boards"][bi]].metadata)
            geo = board_geometry(ms)
            L = st["logits"][bi].numpy()
            cross_k = np.array([crossing_index(L[:, t]) for t in range(T)])
            bid = len(boards)
            boards.append(dict(board=info["boards"][bi], labels=info["labels"][bi].numpy(), logits=L.astype(np.float32),
                               pieces=st["pieces"][bi].numpy().astype(np.float32), cross_k=cross_k,
                               sol=(info["solutions"][bi].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8),
                               **{k_: v for k_, v in geo.items() if k_ != "open_set"},
                               open_pairs=sorted(geo["open_set"])))
            for t in range(T):
                kx = int(cross_k[t])
                if kx <= 0 or L[:kx, t].max() <= LOGIT_HI or L[kx:, t].min() >= -LOGIT_HI:
                    continue
                lo = max(0, kx - W_PEAK)
                ko = lo + int(np.argmax(L[lo:kx, t])) + 1
                f = dict(bid=bid, token=t, ko=ko, kx=kx, onset_call=zl_calls[ko], cross_call=zl_calls[kx],
                         late=zl_calls[kx] > c_dec, label=int(info["labels"][bi, t]))
                star, star_before = star_of(geo, t, ko, cross_k)
                f.update(star=star, star_before=star_before, star_cross_k=(int(cross_k[star]) if star is not None else None))
                if not f["late"]:
                    sm = st["maps"][bi, :, :, t, :].float().numpy()                  # (n_dec, 4, S)
                    f["category"] = classify(f, L[:, t], cross_k, boards[-1]["pieces"][:, :, t], sm, geo, zl_calls, per)
                    f["maps"] = st["maps"][bi, :, :, t, :].clone()
                    f["head_nb"] = st["head_nb"][bi, :, :, :, t, :].clone()
                else:
                    f["category"] = None
                nbs = tok_nbrs(t)
                f["other_nb"] = next((u for u in nbs if relation(geo, t, u) == "open" and u != star and
                                      (geo["rounds"][tuple(geo["tok_cell"][u])] == 0 or
                                       geo["rounds"][tuple(geo["tok_cell"][u])] > geo["rounds"][tuple(geo["tok_cell"][t])])), None)
                f["wall_nb"] = next((u for u in nbs if geo["wall_tok"][u] and u not in (star, f["other_nb"])
                                     and relation(geo, t, u) != "open"), None)
                far = [u for u in range(T) if max(abs(u // G - t // G), abs(u % G - t % G)) >= 4]
                f["far"] = rng.choice(far)
                f["star_cell"] = ([u for u in range(T) if tuple(geo["tok_cell"][u]) == tuple(geo["tok_cell"][star])]
                                  if star is not None else [])
                f["bi"] = bi
                batch_flips.append(f)
        logging.info(f"[analysis] boards {len(boards)}: {len(batch_flips)} flips in this batch")

        # ---- removal check on this batch's flips ----
        exps = []
        for fi, f in enumerate(batch_flips):
            if n_removal >= max_removal or f["late"] or f["star"] is None or not f["star_before"]:
                continue
            n_removal += 1
            start_call = zl_calls[max(0, f["star_cross_k"] - 1)]
            c_stop = min(c_max, zl_calls[min(len(zl_calls) - 1, f["kx"])] + 21)
            conds = [("none", [], False), ("star", [f["star"]], False), ("star_cell", f["star_cell"], False),
                     ("star_cell_all_calls", f["star_cell"], True), ("other_nb", [f["other_nb"]], False),
                     ("wall_nb", [f["wall_nb"]], False), ("far", [f["far"]], False)]
            for name, J, zh in conds:
                if any(j is None for j in J):
                    continue
                exps.append(dict(f=f, cond=name, J=J, zh=zh, lo=start_call, hi=c_stop))
        for s0 in range(0, len(exps), removal_batch):
            chunk = exps[s0:s0 + removal_batch]
            idx = torch.tensor([e["f"]["bi"] for e in chunk], device=device)
            cond_sel = select_rows(conditions, idx, B)
            pe_, pi_, pj_, plo, phi, pzh = [], [], [], [], [], []
            for e_i, e in enumerate(chunk):
                for j in e["J"]:
                    pe_.append(e_i); pi_.append(e["f"]["token"]); pj_.append(j)
                    plo.append(e["lo"]); phi.append(e["hi"]); pzh.append(e["zh"])
            tt = lambda v, dt=torch.long: torch.tensor(v, device=device, dtype=dt)
            st["pairs"] = dict(e=tt(pe_), i=tt(pi_), j=tt(pj_), lo=tt(plo), hi=tt(phi), zh=tt(pzh, torch.bool))
            st["tgt"] = tt([e["f"]["token"] for e in chunk])
            stop = max(e["hi"] for e in chunk)
            st["traj"] = torch.full((len(chunk), len(zl_calls)), float("nan"))
            run_t70(cond_sel, x_t[idx], "block", stop_after=stop)
            for e_i, e in enumerate(chunk):
                n_k = sum(c <= e["hi"] for c in zl_calls)
                tr = st["traj"][e_i, :n_k].numpy()
                kx_new = crossing_index(tr)
                removal.append(dict(bid=e["f"]["bid"], token=e["f"]["token"], cond=e["cond"], kx=e["f"]["kx"],
                                    kx_new=kx_new, n_k=n_k, c_stop=e["hi"], start_call=e["lo"], traj=tr.astype(np.float32),
                                    category=e["f"]["category"]))
        logging.info(f"[removal] {len(exps)} runs, {n_removal} flips so far")
        flips += [{k_: v for k_, v in f.items() if k_ not in ("bi",)} for f in batch_flips]

    out = dict(cfg=dict(K=K, N=N, t_start=t_start, c_dec=c_dec, c_max=c_max, mlp_ablation=mlp_ablation,
                        W_PEAK=W_PEAK, W_PRE=W_PRE, logit_hi=LOGIT_HI, per=per),
               zl_calls=zl_calls, n_dec=n_dec, pieces_names=PIECES, cats=CATS, nb_names=NB_NAMES,
               w=w.cpu(), b=b, logit_init=logits0, boards=boards, flips=flips, removal=removal,
               replay_err=np.array(st["replay_err"]), sum_err=np.array(st["sum_err"]))
    torch.save(out, out_dir / "prune_mech.pt")
    print(f"saved {out_dir / 'prune_mech.pt'}: {len(boards)} boards, {len(flips)} flips, {len(removal)} removal runs; "
          f"replay |Δlogit| max {out['replay_err'].max():.3g}, split sum error max {out['sum_err'].max():.3g}", flush=True)
    try:
        import interp_prune_plots as pp
        pp.make_all(out_dir / "prune_mech.pt", out_dir)
    except Exception as ex:                    # the plots can be redone offline from prune_mech.pt
        print("plotting failed:", repr(ex), flush=True)


def select_rows(s, idx, B):
    """DataSample with the rows idx (tensor fields and per-sample lists with batch dim B)."""
    upd = {}
    il = idx.tolist()
    for f in dataclasses.fields(s):
        v = getattr(s, f.name)
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
            upd[f.name] = v[idx.to(v.device)]
        elif isinstance(v, list) and len(v) == B:
            upd[f.name] = [v[j] for j in il]
    return dataclasses.replace(s, **upd)


if __name__ == "__main__":
    main()
