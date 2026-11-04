"""TEMPORARY diagnostic (Claude, amaze interpretability): can SAE atoms of the read-out state control the drawing?

Fixes the weak design of interp_sae_circuits2.py's causal tests:
  - the dictionary is trained on the state the painter actually reads: the FINAL z_H (call 336) of the conditional
    pass, at ALL 20 denoising steps of full generations from pure noise;
  - training boards are square 8x8 training mazes loaded from their ORIGINAL images and resized on load exactly like
    the test boards (no train/test image offset);
  - every edit acts at the final read-out, at every denoising step (the one place where z edits are known to work).
(0) Which atoms are shared across denoising steps (active at >= 15 of 20 steps) vs only at high / low t; FVU per t.
(1) Sanity: z_H replaced by SAE reconstruction + error (must equal normal) and by the reconstruction alone.
(2) The z-edit test of 1.6 with atoms, on a 4-cell stretch of the solution (path tokens covering solution cells 3-6):
      mean-difference push -a*d_t (a = 1, 2; baseline, same regime);
      remove every active path-writing atom (d_i . u_t > 0) on the stretch tokens (SAE error kept);
      replace the stretch tokens' code by the mean code of off tokens of the same token class at that t;
      control: remove the same number of active non-path-writing atoms, at random.
(3) Reverse: on up to 4 off cells next to the solution (side-branch entrances): mean-difference push +2*d_t, and the
    mean code of path tokens of the same token class.
(4) Dose: on every off token, scale all "off" atoms (> 75% of their activation on off tokens) by 1 - b, b = 1, 2.
Outcome = what the painter draws; per edit also the mean size of the change on the edited tokens (||dz||).
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
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_hv_swap as hs
import interp_sae_circuits as sc
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG, T = 12, 144, 144
N_TRAIN, N_TEST, BATCH = 256, 48, 64
WALL_DARK = 80


def train_subset_orig(src_dir, out_path, n):
    """first n square 8x8 boards of maze_dataset_train_original (830x830 originals), images kept at full size."""
    if os.path.exists(out_path):
        return
    f = pq.ParquetFile(f"{src_dir}/maze_dataset_train_original.parquet")
    keep = []
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["id", "instruction", "m_original_img", "sol_img"])
        sizes = [struct.unpack(">II", base64.b64decode(s[:64])[16:24]) for s in t.column("m_original_img").to_pylist()]
        mask = pyar.array([s == (830, 830) for s in sizes])
        keep.append(t.filter(mask))
        if sum(k.num_rows for k in keep) >= n:
            break
    out = pyar.concat_tables(keep).slice(0, n)
    pq.write_table(out, out_path)
    print(f"train subset (originals): {out.num_rows} square 8x8 boards -> {out_path}", flush=True)


def tokens_of(ms, cids, cats, want_path):
    m = np.zeros(G * G, bool)
    for cid in cids:
        y0, y1, x0, x1 = hs.cell_box(ms, cid)
        for tr in range(G):
            for tc in range(G):
                oy = min(y1, (tr + 1) * 12) - max(y0, tr * 12); ox = min(x1, (tc + 1) * 12) - max(x0, tc * 12)
                if oy >= 6 and ox >= 6 and ((cats[tr * G + tc] != "off") == want_path):
                    m[tr * G + tc] = True
    return m


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
    tl = [int(t) for t in ts]
    predictor = model.sampling_pipeline.predictor
    inner = model.thinker.inner
    D = inner.config.hidden_size
    vd = cfg.data.get("val_dataset", {})
    mk = lambda p, split, meta: AmazeDataset(dataset_path=p, split=split, image_size=IMG, num_channels=3, include_metadata=meta,
                                             condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"))
    train_subset_orig(str(ic.train_src), str(ic.train_subset), N_TRAIN)

    st = {"on": False, "c": 0, "n_calls": 336, "t": None, "fn": None, "rec": None}

    def hook(module, args, output):
        if not st["on"]:
            return None
        c = st["c"]; st["c"] += 1
        if c != st["n_calls"] - 1:
            return None
        if st["rec"] is not None:
            st["rec"][st["t"]] = output.detach().to("cpu", torch.float16)
        if st["fn"] is not None:
            return st["fn"](st["t"], output)
        return None
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

    # ---------------------------------------------------------------- training data: final z_H at every t
    ds_tr = mk(str(ic.train_subset), "train", False)
    Z, GT, CAT = {t: [] for t in tl}, [], []
    for i0 in range(0, min(N_TRAIN, len(ds_tr)), BATCH):
        batch = AmazeDataset.collate_fn([ds_tr[i] for i in range(i0, min(i0 + BATCH, N_TRAIN, len(ds_tr)))])
        cond = model._batch_to_sample(batch, device); B = batch.images.shape[0]
        x0 = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(1000 + i0), device=device)
        _, rec = generate(cond, x0, 2000 + i0, record=True)
        for t in tl:
            Z[t].append(rec[t])
        sol = (batch.images.float().clamp(0, 1) * 255).byte().cpu(); maze = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
        c_ = np.stack([pa.token_labels(sol[i], maze[i])[0] for i in range(B)]); CAT.append(c_); GT.append(c_ != "off")
        print(f"  train boards {i0}..{i0 + B}: generated and recorded", flush=True)
    Z = torch.stack([torch.cat(Z[t]) for t in tl], 1)                                 # (N, 20, T, D) fp16
    GT = torch.from_numpy(np.concatenate(GT)); N = len(Z)
    mu = Z.float().mean(0)                                                              # (20, T, D)
    scl = torch.stack([(Z[:, j].float() - mu[j]).std() for j in range(len(tl))])       # (20,)
    X = ((Z.float() - mu) / scl[None, :, None, None]).half().to(device)
    # mean-difference read-out direction per t (solution minus off tokens, training boards)
    dirs = torch.stack([Z[:, j].float()[GT].mean(0) - Z[:, j].float()[~GT].mean(0) for j in range(len(tl))]).to(device)   # (20, D)
    sae = sc.train_sae(X, "z_H read-out (all t)")
    for p_ in sae.parameters():
        p_.requires_grad_(False)
    mu, scl = mu.to(device), scl.to(device)
    M = sae.W_dec.shape[0]
    # per t: which atoms write path (decoder . read-out direction > 0), off atoms, mean code per token class
    tok_cls = torch.tensor([(r % 3) * 3 + (c % 3) for r in range(G) for c in range(G)], device=device)
    path_w = torch.stack([(sae.W_dec @ (dirs[j] / dirs[j].norm())) > 0 for j in range(len(tl))])   # (20, M)
    mean_code = {"path": torch.zeros(len(tl), 9, M, device=device), "off": torch.zeros(len(tl), 9, M, device=device)}
    off_share = torch.zeros(M, device=device); tot = torch.zeros(M, device=device)
    fq = torch.zeros(len(tl), M, device=device)
    with torch.no_grad():
        for j in range(len(tl)):
            f = sae.encode(X[:, j].float())                                            # (N, T, M)
            fq[j] = (f > 0).float().mean((0, 1))
            g_ = GT.to(device)
            off_share += f[~g_].sum(0); tot += f.sum((0, 1))
            for k in range(9):
                sel = (tok_cls == k)[None].expand(N, -1)
                for name, m_ in (("path", sel & g_), ("off", sel & ~g_)):
                    if m_.any():
                        mean_code[name][j, k] = f[m_].mean(0)
    del X
    off_atoms = (off_share / (tot + 1e-9)) > 0.75
    n_act = (fq > 1e-3).sum(0)
    hi = [j for j, t in enumerate(tl) if t >= 50]; lo = [j for j, t in enumerate(tl) if t <= 20]
    shared = int((n_act >= 15).sum()); hi_only = int(((fq[hi] > 1e-3).any(0) & ~(fq[lo] > 1e-3).any(0)).sum())
    lo_only = int(((fq[lo] > 1e-3).any(0) & ~(fq[hi] > 1e-3).any(0)).sum()); alive = int((n_act > 0).sum())
    print(f"\n(0) atoms active (freq > 0.1%) at >= 15 of 20 steps: {shared} of {alive} alive; only at t >= 50: {hi_only}; only at t <= 20: {lo_only}", flush=True)
    print("    path-writing atoms per t: " + " ".join(f"{t}:{int(path_w[j].sum())}" for j, t in enumerate(tl) if j % 4 == 0)
          + f"; off atoms: {int(off_atoms.sum())}; |path-off gap| per t: " + " ".join(f"{t}:{float(dirs[j].norm()):.2f}" for j, t in enumerate(tl) if j % 4 == 0), flush=True)

    # ---------------------------------------------------------------- test boards
    ds = mk(str(ic.data_parquet), "test", True)
    kept, mss = [], []
    for i in range(len(ds)):
        ms = mcl.build_maze_sample(ds[i].metadata)
        if ms is not None:
            kept.append(i); mss.append(ms)
        if len(kept) == N_TEST:
            break
    batch = AmazeDataset.collate_fn([ds[i] for i in kept]); B = len(kept)
    cond = model._batch_to_sample(batch, device)
    sol_u8 = (batch.images.float().clamp(0, 1) * 255).byte().cpu(); maze_u8 = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
    cats = np.stack([pa.token_labels(sol_u8[i], maze_u8[i])[0] for i in range(B)])
    gt_te = torch.from_numpy(cats != "off").to(device)
    stretch_cells, branch_cells = [], []
    m_str, m_br = np.zeros((B, T), bool), np.zeros((B, T), bool)
    for i, ms in enumerate(mss):
        inner_path = [c for c in ms.path[1:-1]]
        sc_ = inner_path[2:6] if len(inner_path) >= 6 else inner_path[:4]
        stretch_cells.append(set(sc_)); m_str[i] = tokens_of(ms, sc_, cats[i], True)
        pset = set(ms.path); br = []
        for cid in ms.path:
            r, c = ms.id2rc[cid]
            for r2, c2 in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                n_ = ms.rc2id.get((r2, c2))
                if n_ is not None and n_ not in pset and n_ not in br and mcl._is_open(ms.morig, r, c, r2, c2, ms.cell_px):
                    br.append(n_)
        br = br[:4]; branch_cells.append(set(br)); m_br[i] = tokens_of(ms, br, cats[i], False)
    m_str, m_br = torch.from_numpy(m_str).to(device), torch.from_numpy(m_br).to(device)
    print(f"\ntest: {B} boards; stretch tokens per board {float(m_str.float().sum(1).mean()):.1f}, branch-entrance tokens {float(m_br.float().sum(1).mean()):.1f}", flush=True)
    x0 = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(77), device=device)

    J = {t: j for j, t in enumerate(tl)}
    log = []
    def codes(t, z):
        j = J[t]; x = (z.float() - mu[j]) / scl[j]
        f = sae.encode(x); xh = f @ sae.W_dec + sae.b_dec
        return j, x, f, x - xh
    def back(j, f, err):
        return mu[j] + scl[j] * (f @ sae.W_dec + sae.b_dec + err)
    def finish(z, z2, mask):
        d = (z2 - z.float()).norm(dim=-1)
        log.append(float(d[mask].mean()) if mask is not None and mask.any() else float(d.mean()))
        return z2.to(z.dtype)
    def recon_err(t, z):
        j, x, f, err = codes(t, z); return finish(z, back(j, f, err), None)
    def recon_only(t, z):
        j, x, f, err = codes(t, z); return finish(z, back(j, f, torch.zeros_like(err)), None)
    def push(sign, a, mask):
        def fn(t, z):
            j = J[t]; return finish(z, z.float() + sign * a * mask[..., None].float() * dirs[j], mask)
        return fn
    def remove_path_atoms(mask, random_ctrl=False):
        def fn(t, z):
            j, x, f, err = codes(t, z)
            sel = (f > 0) & mask[..., None]
            pw = path_w[j][None, None]
            if not random_ctrl:
                kill = sel & pw
            else:                                                       # same count, among active non-path-writing atoms
                n_kill = (sel & pw).sum(-1, keepdim=True)
                cand = sel & ~pw
                r = torch.rand(f.shape, device=device, generator=st["gen"]) * cand
                rank = r.argsort(-1, descending=True).argsort(-1)
                kill = cand & (rank < n_kill)
            st["killed"].append(float(kill.sum(-1)[mask].float().mean()))
            return finish(z, back(j, f * ~kill, err), mask)
        return fn
    def set_code(kind, mask):
        def fn(t, z):
            j, x, f, err = codes(t, z)
            tgt = mean_code[kind][j][tok_cls][None].expand_as(f)
            f2 = torch.where(mask[..., None], tgt, f)
            return finish(z, back(j, f2, err), mask)
        return fn
    def dose(b):
        def fn(t, z):
            j, x, f, err = codes(t, z)
            m_ = (~gt_te)[..., None] & off_atoms[None, None]
            return finish(z, back(j, torch.where(m_, f * (1 - b), f), err), ~gt_te)
        return fn

    RUNS = [("normal", None), ("SAE reconstruction + error (identity check)", recon_err), ("SAE reconstruction only (no error)", recon_only),
            ("stretch: mean-difference push -1 gap", push(-1, 1.0, m_str)), ("stretch: mean-difference push -2 gaps", push(-1, 2.0, m_str)),
            ("stretch: remove path-writing atoms", remove_path_atoms(m_str)), ("stretch: code -> mean off-token code", set_code("off", m_str)),
            ("stretch: control, remove as many non-path atoms", remove_path_atoms(m_str, True)),
            ("branches: mean-difference push +2 gaps", push(+1, 2.0, m_br)), ("branches: code -> mean path-token code", set_code("path", m_br)),
            ("off tokens: off atoms x0 (removed)", dose(1.0)), ("off tokens: off atoms x(-1) (reversed)", dose(2.0))]
    results = []
    for name, fn in RUNS:
        log.clear(); st["killed"] = []; st["gen"] = torch.Generator(device=device).manual_seed(5)
        out, rec = generate(cond, x0, 78, fn=fn, record=(name == "normal"))
        if name == "normal":
            with torch.no_grad():
                fvu = []
                for t in tl:
                    j, x, f, err = codes(t, rec[t].to(device))
                    fvu.append(float(err.pow(2).sum() / (x - x.mean((0, 1))).pow(2).sum()))
                act = torch.stack([(codes(t, rec[t].to(device))[2] > 0).float().mean((0, 1)) for t in tl])
            print("  SAE held-out FVU at the read-out per t: " + " ".join(f"{t}:{v:.2f}" for t, v in zip(tl, fvu)), flush=True)
        out_u8 = (out * 255).byte().cpu()
        rows = []
        for i, ms in enumerate(mss):
            pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
            path = set(ms.path); inner_c = path - {ms.path[0], ms.path[-1]}
            od = out_u8[i].float().mean(0).numpy() < WALL_DARK; md = maze_u8[i].float().mean(0).numpy() < WALL_DARK
            blue = mcl.blue_mask(out_u8[i].permute(1, 2, 0).numpy())
            rows.append({"solved": inner_c <= pred and not (pred - path), "stretch_drawn": len(stretch_cells[i] & pred) / max(1, len(stretch_cells[i])),
                         "branch_drawn": len(branch_cells[i] & pred) / max(1, len(branch_cells[i])),
                         "rest_drawn": len((inner_c - stretch_cells[i]) & pred) / max(1, len(inner_c - stretch_cells[i])),
                         "extra": len(pred - path), "blue_px": int(blue.sum()), "walls_changed": int((od & ~md & ~blue).sum() + (md & ~od).sum())})
        m = lambda k: float(np.mean([r[k] for r in rows]))
        results.append({"name": name, "rows": rows, "dz": float(np.mean(log)) if log else 0.0,
                        "killed": float(np.mean(st["killed"])) if st["killed"] else None, "img": out_u8[:6]})
        print(f"  {name:48s} | stretch drawn {m('stretch_drawn'):.2f}  rest {m('rest_drawn'):.2f}  branches drawn {m('branch_drawn'):.2f}  "
              f"extra {m('extra'):5.2f}  solved {m('solved'):.2f}  blue {m('blue_px'):5.0f}  walls {m('walls_changed'):5.0f} | "
              f"|dz| on edited tokens {results[-1]['dz']:.2f}" + (f", atoms removed per token {results[-1]['killed']:.1f}" if st["killed"] else ""), flush=True)
    torch.save({"tl": tl, "sae": {k: v.cpu() for k, v in sae.state_dict().items()}, "mu": mu.half().cpu(), "scl": scl.cpu(), "dirs": dirs.cpu(),
                "path_w": path_w.cpu(), "off_atoms": off_atoms.cpu(), "freq_train": fq.cpu(), "freq_test": act.cpu(), "fvu": fvu,
                "results": results, "kept": kept, "sol_u8": sol_u8[:6], "maze_u8": maze_u8[:6]}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
