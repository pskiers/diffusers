"""TEMPORARY diagnostic (Claude, amaze interpretability): deleting concepts from the board with the read-out SAE.

Dictionary: interp_sae_readout.py (final z_H at all 20 denoising steps; template mu_t, scale s_t, mean-difference path
directions d_t). Every edit acts on the final z_H of the conditional pass (call 336) at every denoising step; the SAE
error term is kept, so only the chosen atoms change. Standard generation from pure noise.
Boards: the first 48 parseable test boards are edited; the next parseable test boards (not edited) label the atoms.
Concepts (labelling boards, all steps): share of an atom's activation on tokens of each solution stroke class
(H, V, corner, end = endpoint / marker, off); concept atoms = share > 0.6 (end: > 0.4), active on >= 0.1% of tokens.
"path" atoms = positive vote d_i . u_t (u_t = unit mean-difference path direction at that step).
Runs:
  stretch (path tokens of solution cells 3-6): remove all path atoms; remove the top-k by strength x vote (k = 1, 2, 4, 8);
    controls: remove random non-path atoms until the change has the SAME size per token; random direction, same size;
  whole board: path atoms on all solution tokens (+ size-matched control); H atoms on H tokens; V on V; corner on corner;
    end atoms on end tokens;
  timestep groups (all tokens): atoms active only at t >= 50, only at t <= 20; random shared atoms, same count;
  transplant (duplicated batch, donor = next board, run unedited): all atom codes from the donor (own error kept);
    the donor's whole state (reference).
Per run: drawing metrics, size of the change on edited tokens, and the predicted change of the read-out score in units
of the path - off gap (mean over edited tokens).
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_sae_circuits as sc
import interp_sae_readout as sr
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG, T, N_EDIT, WALL_DARK = 12, 144, 144, 48, 80
CLASSES = {"H": ["H"], "V": ["V"], "corner": ["corner_LT", "corner_LB", "corner_RT", "corner_RB"], "end": ["end"], "off": ["off"]}


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
    ts = model.scheduler.timesteps; tl = [int(t) for t in ts]; J = {t: j for j, t in enumerate(tl)}
    predictor = model.sampling_pipeline.predictor
    inner = model.thinker.inner
    D = inner.config.hidden_size

    P = torch.load(str(ic.sae_readout), weights_only=False)
    assert P["tl"] == tl, "denoising steps differ from the dictionary's"
    sae = sc.TopKSAE(D, P["sae"]["W_enc"].shape[1], sc.K).to(device); sae.load_state_dict(P["sae"]); sae.eval()
    for p_ in sae.parameters():
        p_.requires_grad_(False)
    mu, scl, dirs = P["mu"].float().to(device), P["scl"].float().to(device), P["dirs"].float().to(device)
    M = sae.W_dec.shape[0]
    vote = torch.stack([sae.W_dec @ (dirs[j] / dirs[j].norm()) for j in range(len(tl))])          # (20, M), normalised space
    path_w = vote > 0
    fq = P["freq_train"].to(device)
    hi = [j for j, t in enumerate(tl) if t >= 50]; lo = [j for j, t in enumerate(tl) if t <= 20]
    hi_only = (fq[hi] > 1e-3).any(0) & ~(fq[lo] > 1e-3).any(0)
    lo_only = (fq[lo] > 1e-3).any(0) & ~(fq[hi] > 1e-3).any(0)
    shared = (fq > 1e-3).sum(0) >= 15
    g_ = torch.Generator(device=device).manual_seed(11)
    sh_idx = torch.nonzero(shared).flatten()
    rand_shared = torch.zeros(M, dtype=torch.bool, device=device)
    rand_shared[sh_idx[torch.randperm(len(sh_idx), generator=g_, device=device)[:int(hi_only.sum())]]] = True

    st = {"on": False, "c": 0, "t": None, "fn": None, "rec": None}

    def hook(module, args, output):
        if not st["on"]:
            return None
        c = st["c"]; st["c"] += 1
        if c != 335:
            return None
        if st["rec"] is not None:
            st["rec"][st["t"]] = output.detach().to("cpu", torch.float16)
        return st["fn"](st["t"], output) if st["fn"] is not None else None
    inner.L_level.register_forward_hook(hook)

    def generate(cond, x0, seed, fn=None, record=False):
        B = x0.shape[0]; x = x0; gen = torch.Generator(device=device).manual_seed(seed)
        rec = {} if record else None
        with torch.no_grad():
            for t in ts:
                s_ = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(B).to(device))
                st.update(on=True, c=0, t=int(t), fn=fn, rec=rec)
                pred_cond, _, _ = model.forward_with_carry(s_)
                st["on"] = False
                pred_uncond = model(model.null_condition_sample(s_)).pred
                pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
            out = model.decode_for_eval(x).clamp(0, 1)
        return out, rec

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=IMG, num_channels=3, include_metadata=True,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"))
    kept, mss = [], []
    for i in range(len(ds)):
        ms = mcl.build_maze_sample(ds[i].metadata)
        if ms is not None:
            kept.append(i); mss.append(ms)
    edit_i, lab_i = kept[:N_EDIT], kept[N_EDIT:]
    def prep(idx):
        batch = AmazeDataset.collate_fn([ds[i] for i in idx])
        sol = (batch.images.float().clamp(0, 1) * 255).byte().cpu(); maze = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
        cats = np.stack([pa.token_labels(sol[i], maze[i])[0] for i in range(len(idx))])
        return batch, model._batch_to_sample(batch, device), cats, maze
    def codes(t, z):
        j = J[t]; x = (z.float() - mu[j]) / scl[j]
        f = sae.encode(x); return j, x, f, x - (f @ sae.W_dec + sae.b_dec)
    def back(j, f, err):
        return mu[j] + scl[j] * (f @ sae.W_dec + sae.b_dec + err)

    # ---------------------------------------------------------------- label atoms on boards that are not edited
    batch_l, cond_l, cats_l, _ = prep(lab_i)
    x0_l = torch.randn((len(lab_i), 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(55), device=device)
    _, rec = generate(cond_l, x0_l, 56, record=True)
    mass = {k: torch.zeros(M, device=device) for k in CLASSES}; act_any = torch.zeros(M, device=device)
    with torch.no_grad():
        for t in tl:
            j, x, f, err = codes(t, rec[t].to(device))
            act_any += (f > 0).float().mean((0, 1)) / len(tl)
            for k, names in CLASSES.items():
                m_ = torch.from_numpy(np.isin(cats_l, names)).to(device)
                mass[k] += f[m_].sum(0)
    tot = sum(mass.values()) + 1e-9
    share = {k: v / tot for k, v in mass.items()}
    alive = act_any >= 1e-3
    concept = {k: alive & (share[k] > (0.4 if k == "end" else 0.6)) for k in CLASSES}
    print(f"labelling boards: {len(lab_i)}; concept atoms: " + ", ".join(f"{k} {int(v.sum())}" for k, v in concept.items())
          + f"; path-writing (per step, mean) {float(path_w.float().sum(1).mean()):.0f}; only t>=50: {int(hi_only.sum())}, only t<=20: {int(lo_only.sum())}", flush=True)

    # ---------------------------------------------------------------- edited boards
    batch, cond, cats, maze_u8 = prep(edit_i); B = len(edit_i); mss_e = mss[:N_EDIT]
    cls_mask = {k: torch.from_numpy(np.isin(cats, names)).to(device) for k, names in CLASSES.items()}
    gt = ~cls_mask["off"]
    stretch_cells = []; m_str = np.zeros((B, T), bool)
    for i, ms in enumerate(mss_e):
        ip = ms.path[1:-1]; sc_ = ip[2:6] if len(ip) >= 6 else ip[:4]
        stretch_cells.append(set(sc_)); m_str[i] = sr.tokens_of(ms, sc_, cats[i], True)
    m_str = torch.from_numpy(m_str).to(device)
    x0 = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(77), device=device)
    rnd_dir = torch.randn(D, generator=torch.Generator(device=device).manual_seed(3), device=device); rnd_dir /= rnd_dir.norm()
    log = {"dz": [], "dscore": [], "killed": []}

    def finish(j, z, z2, mask):
        d = (z2 - z.float()); n = d.norm(dim=-1)
        ds = (d @ dirs[j]) / dirs[j].pow(2).sum()                                        # read-out change, in gaps
        log["dz"].append(float(n[mask].mean())); log["dscore"].append(float(ds[mask].mean()))
        return z2.to(z.dtype)

    def delete(atom_sel, mask):
        """atom_sel(j) -> (M,) bool atoms to remove on the tokens in mask."""
        def fn(t, z):
            j, x, f, err = codes(t, z)
            kill = (f > 0) & mask[..., None] & atom_sel(j)[None, None]
            log["killed"].append(float(kill.sum(-1)[mask].float().mean()))
            return finish(j, z, back(j, f * ~kill, err), mask)
        return fn

    def delete_topk(k, mask):
        def fn(t, z):
            j, x, f, err = codes(t, z)
            sc_ = torch.where((f > 0) & path_w[j][None, None], f * vote[j][None, None], torch.zeros_like(f))
            top = sc_.topk(k, dim=-1).indices
            kill = torch.zeros_like(f, dtype=torch.bool).scatter_(-1, top, True) & (sc_ > 0) & mask[..., None]
            log["killed"].append(float(kill.sum(-1)[mask].float().mean()))
            return finish(j, z, back(j, f * ~kill, err), mask)
        return fn

    def size_matched(atom_sel, mask):
        """remove random active atoms NOT in atom_sel, in random order, until the change per token reaches the size of
        removing atom_sel there (or no candidates are left)."""
        def fn(t, z):
            j, x, f, err = codes(t, z)
            sel = atom_sel(j)[None, None]
            target = ((f * ((f > 0) & sel)) @ sae.W_dec).norm(dim=-1)                       # (B, T) size of the concept deletion
            v, idx = f.topk(sc.K, dim=-1)                                                   # the active atoms (<= 16)
            cand = (v > 0) & ~sel.expand_as(f).gather(-1, idx)
            r = torch.rand(v.shape, device=device, generator=st["gen"]) * cand
            order = r.argsort(-1, descending=True)
            v_o, idx_o, cand_o = v.gather(-1, order), idx.gather(-1, order), cand.gather(-1, order)
            contrib = (v_o * cand_o)[..., None] * sae.W_dec[idx_o]                          # (B, T, 16, D)
            norms = contrib.cumsum(-2).norm(dim=-1)                                          # (B, T, 16)
            need = (norms < target[..., None]).sum(-1, keepdim=True) + 1                     # prefix length reaching the target
            need = torch.where(target[..., None] > 0, need, torch.zeros_like(need))         # nothing to match: remove nothing
            take = (torch.arange(sc.K, device=device)[None, None] < need) & cand_o & mask[..., None]
            kill = torch.zeros_like(f, dtype=torch.bool).scatter_(-1, idx_o, take)
            log["killed"].append(float(kill.sum(-1)[mask].float().mean()))
            return finish(j, z, back(j, f * ~kill, err), mask)
        return fn

    def rand_direction(atom_sel, mask):
        def fn(t, z):
            j, x, f, err = codes(t, z)
            size = ((f * ((f > 0) & atom_sel(j)[None, None])) @ sae.W_dec).norm(dim=-1)   # normalised units
            z2 = z.float() + (scl[j] * size * mask.float())[..., None] * rnd_dir
            return finish(j, z, z2, mask)
        return fn

    path_sel = lambda j: path_w[j]
    const = lambda m: (lambda j: m)
    all_tok = torch.ones((B, T), dtype=torch.bool, device=device)
    RUNS = [("normal", None, None),
            ("stretch: remove all path atoms", delete(path_sel, m_str), m_str),
            ("stretch: control, random atoms, same size", size_matched(path_sel, m_str), m_str),
            ("stretch: control, random direction, same size", rand_direction(path_sel, m_str), m_str)]
    RUNS += [(f"stretch: remove top-{k} path atoms (strength x vote)", delete_topk(k, m_str), m_str) for k in (1, 2, 4, 8)]
    RUNS += [("board: path atoms on all solution tokens", delete(path_sel, gt), gt),
             ("board: control, random atoms, same size", size_matched(path_sel, gt), gt),
             ("board: H atoms on H tokens", delete(const(concept["H"]), cls_mask["H"]), cls_mask["H"]),
             ("board: V atoms on V tokens", delete(const(concept["V"]), cls_mask["V"]), cls_mask["V"]),
             ("board: corner atoms on corner tokens", delete(const(concept["corner"]), cls_mask["corner"]), cls_mask["corner"]),
             ("board: end/marker atoms on end tokens", delete(const(concept["end"]), cls_mask["end"]), cls_mask["end"]),
             ("steps: atoms only at t>=50, all tokens", delete(const(hi_only), all_tok), all_tok),
             ("steps: atoms only at t<=20, all tokens", delete(const(lo_only), all_tok), all_tok),
             ("steps: control, random shared atoms, same count", delete(const(rand_shared), all_tok), all_tok)]

    def score(out, mss_, cats_, maze_, normal_cats=None):
        out_u8 = (out * 255).byte().cpu(); rows = []
        oc = np.stack([pa.token_labels(out_u8[i], out_u8[i])[0] for i in range(len(mss_))])
        for i, ms in enumerate(mss_):
            pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
            path = set(ms.path); inner_c = path - {ms.path[0], ms.path[-1]}
            od = out_u8[i].float().mean(0).numpy() < WALL_DARK; md = maze_[i].float().mean(0).numpy() < WALL_DARK
            blue = mcl.blue_mask(out_u8[i].permute(1, 2, 0).numpy())
            r = {"solved": inner_c <= pred and not (pred - path), "path_drawn": len(inner_c & pred) / max(1, len(inner_c)),
                 "extra": len(pred - path), "blue_px": int(blue.sum()), "walls_changed": int((od & ~md & ~blue).sum() + (md & ~od).sum()),
                 "pred_rc": {ms.id2rc[c] for c in pred}}
            if i < len(stretch_cells):
                r["stretch_drawn"] = len(stretch_cells[i] & pred) / max(1, len(stretch_cells[i]))
                r["rest_drawn"] = len((inner_c - stretch_cells[i]) & pred) / max(1, len(inner_c - stretch_cells[i]))
            for k in ("H", "V", "corner", "end"):
                m_ = np.isin(cats_[i], CLASSES[k])
                if m_.any():
                    r[f"{k}_drawn"] = float((oc[i][m_] != "off").mean())
                    if normal_cats is not None:
                        r[f"{k}_same_shape"] = float((oc[i][m_] == normal_cats[i][m_]).mean())
            rows.append(r)
        return rows, oc

    results, normal_oc = [], None
    def report(name, rows, extra=""):
        m = lambda k: np.mean([r[k] for r in rows if k in r]) if any(k in r for r in rows) else float("nan")
        line = (f"  {name:52s} | solved {m('solved'):.2f}  path {m('path_drawn'):.2f}  stretch {m('stretch_drawn'):.2f}  rest {m('rest_drawn'):.2f}  "
                f"extra {m('extra'):5.2f}  blue {m('blue_px'):5.0f}  walls {m('walls_changed'):5.0f} | drawn H/V/corner/end "
                f"{m('H_drawn'):.2f}/{m('V_drawn'):.2f}/{m('corner_drawn'):.2f}/{m('end_drawn'):.2f}  same shape "
                f"{m('H_same_shape'):.2f}/{m('V_same_shape'):.2f}/{m('corner_same_shape'):.2f}/{m('end_same_shape'):.2f}" + extra)
        print(line, flush=True)
    for name, fn, mask in RUNS:
        for k in log:
            log[k].clear()
        st["gen"] = torch.Generator(device=device).manual_seed(5)
        out, _ = generate(cond, x0, 78, fn=fn)
        rows, oc = score(out, mss_e, cats, maze_u8, normal_oc)
        if name == "normal":
            normal_oc = oc; rows, _ = score(out, mss_e, cats, maze_u8, normal_oc)
        ex = (f" | |dz| {np.mean(log['dz']):.2f}, read-out change {np.mean(log['dscore']):+.2f} gaps" if log["dz"] else "") + \
             (f", atoms removed/token {np.mean(log['killed']):.1f}" if log["killed"] else "")
        report(name, rows, ex)
        results.append({"name": name, "rows": [{k: v for k, v in r.items() if k != "pred_rc"} for r in rows], "dz": log["dz"][:],
                        "dscore": log["dscore"][:], "killed": log["killed"][:], "img": (out[:6] * 255).byte().cpu()})

    # ---------------------------------------------------------------- transplant: duplicated batch, donors unedited
    don_i = edit_i[1:] + edit_i[:1]; mss_d = mss_e[1:] + mss_e[:1]
    batch2 = AmazeDataset.collate_fn([ds[i] for i in edit_i + don_i]); cond2 = model._batch_to_sample(batch2, device)
    x02 = torch.cat([x0, torch.roll(x0, -1, 0)])
    def transplant(full):
        def fn(t, z):
            zr, zd = z[:B], z[B:]
            if full:
                z2 = zd.float()
            else:
                j, x, f, err = codes(t, zr); _, _, fd, _ = codes(t, zd)
                z2 = back(j, fd, err)
            log["dz"].append(float((z2 - zr.float()).norm(dim=-1).mean()))
            return torch.cat([z2.to(z.dtype), zd])
        return fn
    for name, full in (("transplant: all atom codes from the next board (own error kept)", False), ("transplant: whole state from the next board", True)):
        for k in log:
            log[k].clear()
        out2, _ = generate(cond2, x02, 78, fn=transplant(full))
        out = out2[:B]
        # donor path drawn on the receiver board (cells compared by grid position)
        donor_drawn, own_drawn = [], []
        for i, ms in enumerate(mss_e):
            pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
            pred_rc = {ms.id2rc[c] for c in pred}
            own_rc = {ms.id2rc[c] for c in ms.path}; don_rc = {mss_d[i].id2rc[c] for c in mss_d[i].path}
            donor_drawn.append(len(pred_rc & (don_rc - own_rc)) / max(1, len(don_rc - own_rc)))
            own_drawn.append(len(pred_rc & (own_rc - don_rc)) / max(1, len(own_rc - don_rc)))
        print(f"  {name:52s} | donor-only path cells drawn {np.mean(donor_drawn):.2f}, own-only path cells drawn {np.mean(own_drawn):.2f}"
              f" | |dz| {np.mean(log['dz']):.2f}", flush=True)
        results.append({"name": name, "donor_drawn": donor_drawn, "own_drawn": own_drawn, "dz": log["dz"][:], "img": (out[:6] * 255).byte().cpu()})

    torch.save({"results": results, "concept": {k: v.cpu() for k, v in concept.items()}, "share": {k: v.cpu() for k, v in share.items()},
                "hi_only": hi_only.cpu(), "lo_only": lo_only.cpu(), "edit_i": edit_i, "lab_i": lab_i}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
