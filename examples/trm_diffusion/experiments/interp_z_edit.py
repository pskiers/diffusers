"""TEMPORARY diagnostic (cluster only): corrected causal tests of the thinker state (review of all z experiments).

Corrections vs the earlier z experiments:
  - edits are applied at EVERY call of the chosen kind (z_L updates and/or z_H updates) of EVERY reasoning step, in
    the conditional pass of EVERY denoising step (earlier: z_L only; az_push "h1": only the first 6 calls; zh_patch:
    one call);
  - z_H is edited too (it is what the logits = lm_head(z_H) are made from); earlier it was never edited at every call;
  - directions are mean differences d_{t,c} = mean(path tokens) − mean(off tokens) of the state at call c of the
    pass at denoising step t (normal runs, even boards), not logistic-probe weights; alpha=1 moves a token by one
    full path−off gap (−d on path tokens: "look like an average off token");
  - activation patching from a donor board without contaminating the donor: the batch is duplicated, the second
    half (donor board i−1, its own x_t and conditions) runs unedited, and the first half's state is overwritten at
    the chosen tokens at every call of the chosen kind. Baseline for these runs: the duplicated batch unpatched.
Outcome = what the painter draws (odd boards = held out from the directions): the target stretch of the remaining
path (4 cells), the thinker's own side branches (drop tokens from drop_concept.pt), the rest of the remaining
path, off-path cells, and for patching how much of the donor's path appears.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_az_push as az
import interp_hv_swap as hs
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

# (name, kind: push|rand|swap|None, edited states, target, sign, alpha)
RUNS = ([("normal", None, (), None, 0, 0.0)]
        + [("push_zL", "push", ("z_L",), "path", -1, a) for a in (1.0, 2.0, 4.0)]
        + [("push_zH", "push", ("z_H",), "path", -1, a) for a in (1.0, 2.0, 4.0)]
        + [("push_both", "push", ("z_L", "z_H"), "path", -1, a) for a in (1.0, 2.0)]
        + [("rand_zL", "rand", ("z_L",), "path", -1, 4.0), ("rand_zH", "rand", ("z_H",), "path", -1, 4.0)]
        + [("push_zL", "push", ("z_L",), "drop", 1, a) for a in (2.0, 4.0)]
        + [("push_zH", "push", ("z_H",), "drop", 1, a) for a in (2.0, 4.0)]
        + [("push_both", "push", ("z_L", "z_H"), "drop", 1, 2.0)]
        + [("normal_dup", "swap", (), None, 0, 0.0)]
        + [("swap_zL", "swap", ("z_L",), "all", 0, 1.0), ("swap_zH", "swap", ("z_H",), "all", 0, 1.0),
           ("swap_both", "swap", ("z_L", "z_H"), "all", 0, 1.0),
           ("swap_zH", "swap", ("z_H",), "path", 0, 1.0), ("swap_both", "swap", ("z_L", "z_H"), "path", 0, 1.0)])


def dup(s, B, perm):
    upd = {}
    for f in dataclasses.fields(s):
        v = getattr(s, f.name)
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
            upd[f.name] = torch.cat([v, v[perm]])
        elif isinstance(v, list) and len(v) == B:
            upd[f.name] = v + [v[j] for j in perm]
    return dataclasses.replace(s, **upd)


@torch.no_grad()
def denoise(model, conditions, x, run_ts, generator, hook_fn, n_out):
    predictor = model.sampling_pipeline.predictor
    state = {"on": False, "c": 0, "t": None}

    def hook(module, inputs, output):
        if not state["on"]:
            return None
        c = state["c"]; state["c"] += 1
        new = hook_fn(state["t"], c, output)
        return None if new is None else new
    h = model.thinker.inner.L_level.register_forward_hook(hook)
    try:
        for t in run_ts:
            s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))
            state.update(on=True, c=0, t=int(t))
            pred_cond, _, _ = model.forward_with_carry(s)
            state["on"] = False
            pred_uncond = model(model.null_condition_sample(s)).pred
            pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
            x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    finally:
        h.remove()
    return model.decode_for_eval(x[:n_out]).clamp(0, 1)


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
    run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
    ts = [int(t) for t in run_ts]
    tc = model.thinker.inner.config
    per = tc.L_cycles + 1
    kind_of = lambda c: "z_H" if c % per == per - 1 else "z_L"
    drop_mask = torch.load(str(ic.drop_concept), weights_only=False)["drop_mask"].numpy()
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    starts = list(range(0, n_total, batch_size))

    # ---- phase 1: per (t, call) mean path / off state on the even boards ----
    S = {}
    n_board = 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        even = torch.from_numpy(np.array([(n_board + i) % 2 == 0 for i in range(b["B"])]))
        gt = torch.from_numpy(b["cats"] != "off")
        mp = (gt & even[:, None]).float().to(device)[..., None]; mo = ((~gt) & even[:, None]).float().to(device)[..., None]

        def acc(t, c, out):
            k = (t, c)
            o = out.float()
            sp, so = (o * mp).sum((0, 1)), (o * mo).sum((0, 1))
            if k in S:
                S[k][0] += sp; S[k][1] += so
            else:
                S[k] = [sp, so]
            return None
        denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), acc, b["B"])
        S.setdefault("n", [0, 0]); S["n"][0] += float(mp.sum()); S["n"][1] += float(mo.sum())
        n_board += b["B"]
    npth, noff = S.pop("n")
    D = {k: (v[0] / npth - v[1] / noff) for k, v in S.items()}            # (t, c) -> (hidden,) on device
    g = torch.Generator(device="cpu").manual_seed(0)
    R = {}
    for k, v in D.items():
        r = torch.randn(v.numel(), generator=g).to(device)
        R[k] = r / r.norm() * v.norm()
    n_calls = max(c for _, c in D) + 1
    for t in (70, 5):
        nz = [float(D[(t, c)].norm()) for c in range(n_calls)]
        print(f"t={t}: |path−off gap| per call: call1 {nz[0]:.2f}, call21 {nz[20]:.2f}, call336 {nz[-1]:.2f}; "
              f"cos(d_call21, d_call336) {float(D[(t, 20)] @ D[(t, n_calls - 1)] / D[(t, 20)].norm() / D[(t, n_calls - 1)].norm()):+.2f}", flush=True)

    # ---- phase 2 ----
    rows, n_board = [], 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        idx = np.arange(n_board, n_board + B)
        cells_p, mp_ = zip(*[az.stretch_tokens(ms, r_, b["cats"][i]) for i, (ms, r_) in enumerate(zip(mss, recs))])
        masks = {"path": torch.from_numpy(np.stack(mp_)), "drop": torch.from_numpy(drop_mask[idx]),
                 "all": torch.ones(B, 144, dtype=torch.bool)}
        perm = torch.roll(torch.arange(B), 1)                                    # donor of board i = board i-1
        cond2, x2 = dup(b["conditions"], B, perm.to(device)), torch.cat([b["x_t"], b["x_t"][perm.to(device)]])
        for name, kind, states, tgt, sign, alpha in RUNS:
            m = None if tgt is None else masks[tgt].to(device)[..., None].float()

            def fn(t, c, out, kind=kind, states=states, m=m, sign=sign, alpha=alpha):
                if kind is None or kind_of(c) not in states:
                    return None
                if kind in ("push", "rand"):
                    vec = (D if kind == "push" else R)[(t, c)]
                    return (out.float() + m * (sign * alpha) * vec).to(out.dtype)
                new = out.clone()                                               # swap: first half <- donor half
                new[:B] = torch.where(m.bool(), out[B:], out[:B])
                return new
            swap = kind == "swap"
            out = denoise(model, cond2 if swap else b["conditions"], x2 if swap else b["x_t"], run_ts,
                          torch.Generator(device=device).manual_seed(b["base_seed"] + 1), fn, B)
            out_u8 = (out * 255).byte().cpu()
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                oc = pa.token_labels(out_u8[i], out_u8[i])[0]
                j = int(perm[i]); dms = mss[j]
                own_rc = {ms.id2rc[c_] for c_ in r_["path"]}
                donor_rc = {dms.id2rc[c_] for c_ in recs[j]["path"]} - own_rc
                pred_rc = {ms.id2rc[c_] for c_ in pred}
                rest = r_["remaining"] - set(cells_p[i])
                dm = masks["drop"][i].numpy()
                rows.append({"run": name, "target": tgt, "alpha": alpha, "held_out": bool(idx[i] % 2 == 1),
                             "stretch_drawn": len(set(cells_p[i]) & pred) / max(1, len(cells_p[i])),
                             "drop_drawn": float((oc[dm] != "off").mean()) if dm.any() else None,
                             "rest_remaining_drawn": len(rest & pred) / len(rest) if rest else None,
                             "visible_kept": len(r_["visible"] & pred) / len(r_["visible"]),
                             "off_path": len(pred - r_["path"]),
                             "donor_path_drawn": len(donor_rc & pred_rc) / len(donor_rc) if donor_rc else None})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    keys = ["stretch_drawn", "drop_drawn", "rest_remaining_drawn", "visible_kept", "off_path", "donor_path_drawn"]
    print("\nheld-out boards (odd). stretch = 4 cells of the remaining path; drop = the thinker's own side-branch tokens")
    for name, kind, states, tgt, sign, alpha in RUNS:
        sel = [r_ for r_ in rows if r_["run"] == name and r_["target"] == tgt and r_["alpha"] == alpha and r_["held_out"]]
        print(f"  {name:10s} target={str(tgt):5s} alpha={alpha:g} n={len(sel)} | " + "  ".join(
            f"{k} {np.mean([r_[k] for r_ in sel if r_[k] is not None]):.3f}" for k in keys))
    torch.save({"rows": rows, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
