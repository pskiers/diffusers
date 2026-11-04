"""TEMPORARY diagnostic (cluster only): E2c — causal tests of the directional message-passing heads.

Conditional pass, every call, every denoising step (full budget). Clean boards, as before.
(a) Head ablation (head output zeroed before o_proj): L0H7 (looks right), L0H2 (up), L0H0 / L0H3 / L0H6 (down), all three
    down heads, L1H0, L1H1, and the heads that look left in the z_H updates (L0H3 + L0H5) only at the z_H calls.
    Unpruned side-branch cells (drawn, off the solution) are classified by where their branch's dead-end tip lies
    (the direction of the neighbour cell towards the tip), to test whether removing a head that looks in direction X
    leaves branches whose tip is in direction X.
(b) Edge cuts: for the thinker's own dropped tokens (drop_concept.pt), the attention of that token to ONE neighbour token
    is removed (renormalised) in every head of both layers: the neighbour towards the tip, or (control) the neighbour
    away from it. Outcome: share of the dropped tokens drawn, extra off-path cells.
"""
import dataclasses
from collections import deque

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_hv_swap as hs
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from models.trm.layers import apply_rotary_pos_emb

G = 12
RUNS = [("normal", None, None), ("L0H7 (right)", [(0, 7)], "all"), ("L0H2 (up)", [(0, 2)], "all"),
        ("L0H0 (down)", [(0, 0)], "all"), ("L0H3 (down)", [(0, 3)], "all"), ("L0H6 (down)", [(0, 6)], "all"),
        ("all 3 down heads", [(0, 0), (0, 3), (0, 6)], "all"), ("L1H0", [(1, 0)], "all"), ("L1H1", [(1, 1)], "all"),
        ("left heads at z_H calls (L0H3+L0H5)", [(0, 3), (0, 5)], "zH"), ("cut tip-ward edge", "tip", None), ("cut away edge", "away", None)]
DIRS = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}


def graph(ms):
    nb = {c: [] for c in ms.id2rc}
    for cid, (r, c) in ms.id2rc.items():
        for r2, c2 in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            n = ms.rc2id.get((r2, c2))
            if n is not None and mcl._is_open(ms.morig, r, c, r2, c2, ms.cell_px):
                nb[cid].append(n)
    return nb


def tip_info(ms):
    """per off-path cell: direction (dr, dc) of its neighbour towards the nearest dead-end tip, and its distance to it."""
    nb = graph(ms); path = set(ms.path)
    tips = [c for c in nb if c not in path and len(nb[c]) == 1]
    d = {t: 0 for t in tips}; q = deque(tips)
    while q:
        x = q.popleft()
        for y in nb[x]:
            if y not in d and y not in path:
                d[y] = d[x] + 1; q.append(y)
    out = {}
    for c in nb:
        if c in path or c not in d:
            continue
        cand = [n for n in nb[c] if n in d and d[n] < d[c]]
        if d[c] == 0 or not cand:
            out[c] = (None, 0); continue
        n = cand[0]; (r, cc_), (r2, c2) = ms.id2rc[c], ms.id2rc[n]
        away = [m for m in nb[c] if m != n]
        out[c] = ((r2 - r, c2 - cc_), d[c], [(ms.id2rc[m][0] - r, ms.id2rc[m][1] - cc_) for m in away])
    return out


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
    ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
    predictor = model.sampling_pipeline.predictor
    inner = model.thinker.inner
    layers = inner.L_level.layers
    hd, nH = layers[0].self_attn.head_dim, layers[0].self_attn.num_heads
    per = inner.config.L_cycles + 1
    drop_mask = torch.load(str(ic.drop_concept), weights_only=False)["drop_mask"].numpy()
    st = {"on": False, "call": 0, "abl": None, "when": None, "cut": None}
    inner.L_level.register_forward_pre_hook(lambda m, a: st.__setitem__("call", st["call"] + 1) if st["on"] else None)

    def ablate_pre(l):
        def f(module, args):
            if not st["on"] or st["abl"] is None:
                return None
            if st["when"] == "zH" and (st["call"] - 1) % per != per - 1:
                return None
            hs_ = [h for (ll, h) in st["abl"] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return f

    def cut_hook(module, args, kwargs, output):
        if not st["on"] or st["cut"] is None:
            return None
        x = kwargs["hidden_states"]; cos_sin = kwargs.get("cos_sin")
        Bx, S, _ = x.shape
        qkv = module.qkv_proj(x).view(Bx, S, 3 * nH, hd)
        q, k, v = qkv[:, :, :nH], qkv[:, :, nH:2 * nH], qkv[:, :, 2 * nH:]
        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, *cos_sin)
        a = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / hd ** 0.5, -1)
        a = a * st["cut"][:, None]                                                     # (B,1,T,S) mask
        a = a / a.sum(-1, keepdim=True)
        o = torch.einsum("bhqk,bkhd->bqhd", a, v.float()).reshape(Bx, S, nH * hd).to(output.dtype)
        return module.o_proj(o)
    for l, layer in enumerate(layers):
        layer.self_attn.o_proj.register_forward_pre_hook(ablate_pre(l))
        layer.self_attn.register_forward_hook(cut_hook, with_kwargs=True)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    rows, n_board = [], 0
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        idx = np.arange(n_board, n_board + B)
        infos = [tip_info(ms) for ms in mss]
        # token-level cut masks: for each dropped token, its tip-ward / away neighbour token
        cuts = {k: torch.ones(B, G * G, G * G, device=device) for k in ("tip", "away")}
        for i, ms in enumerate(mss):
            cs = 144 / ms.height
            for t in np.nonzero(drop_mask[idx[i]])[0]:
                tr, tc = divmod(int(t), G)
                cid = ms.rc2id.get((int((tr * 12 + 6) // cs), int((tc * 12 + 6) // cs)))
                inf = infos[i].get(cid)
                if not inf or inf[0] is None:
                    continue
                for key, dirs in (("tip", [inf[0]]), ("away", inf[2][:1])):
                    for (dr, dc) in dirs:
                        r2, c2 = tr + dr, tc + dc
                        if 0 <= r2 < G and 0 <= c2 < G:
                            cuts[key][i, t, r2 * G + c2] = 0
        for name, spec, when in RUNS:
            st.update(abl=spec if isinstance(spec, list) else None, when=when,
                      cut=cuts[spec] if spec in ("tip", "away") else None)
            gen = torch.Generator(device=device).manual_seed(b["base_seed"] + 1)
            x = b["x_t"]
            with torch.no_grad():
                for t in ts:
                    s = dataclasses.replace(b["conditions"], x_noisy=x, timesteps=t.expand(B).to(device))
                    st.update(on=True, call=0)
                    pred_cond, _, _ = model.forward_with_carry(s)
                    st["on"] = False
                    pred_uncond = model(model.null_condition_sample(s)).pred
                    pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                    x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
                out = model.decode_for_eval(x).clamp(0, 1)
            out_u8 = (out * 255).byte().cpu()
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                extra = pred - r_["path"]
                ex_dir = {}
                for c in extra:
                    inf = infos[i].get(c)
                    lab = "tip itself" if inf and inf[0] is None else (DIRS.get(inf[0], "?") if inf else "no tip (loop/other)")
                    ex_dir[lab] = ex_dir.get(lab, 0) + 1
                oc = pa.token_labels(out_u8[i], out_u8[i])[0]
                dm = drop_mask[idx[i]]
                inner_c = r_["path"] - {ms.path[0], ms.path[-1]}
                rows.append({"run": name, "board": int(idx[i]), "solved": inner_c <= pred and not extra, "n_extra": len(extra),
                             "n_missing": len(inner_c - pred), "extra_by_tip_dir": ex_dir,
                             "drop_tok_drawn": float((oc[dm] != "off").mean()) if dm.any() else None})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    # all boards (no fitting here); the drop tokens come from the step-1 analysis (all boards)
    base = {}
    for c in mss[0].id2rc if False else []:
        pass
    print("\nall boards; extra = drawn off-path cells, by the direction of their branch's dead-end tip")
    print("  run                                  | solved  missing/board  extra/board | extra cells with tip to the right / left / down / up / (tip itself)")
    for name, spec, when in RUNS:
        sel = [r for r in rows if r["run"] == name]
        tot = {k: sum(r["extra_by_tip_dir"].get(k, 0) for r in sel) for k in ("right", "left", "down", "up", "tip itself")}
        print(f"  {name:36s} | {np.mean([r['solved'] for r in sel]):5.2f}  {np.mean([r['n_missing'] for r in sel]):12.2f}  {np.mean([r['n_extra'] for r in sel]):10.2f} | "
              + " / ".join(f"{tot[k]}" for k in ("right", "left", "down", "up", "tip itself"))
              + f" | dropped tokens drawn {np.mean([r['drop_tok_drawn'] for r in sel if r['drop_tok_drawn'] is not None]):.2f}")
    torch.save({"rows": rows}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
