"""TEMPORARY diagnostic (cluster only): causal test of the AlphaZero-style "drop" vector v (drop_concept.py).

Clean boards, CFG, same boards / noise / seed as before. In the conditional pass, at every denoising step, during
H-cycle 1 of reasoning step 1 (the first L_cycles z_L updates), add a push to the z_L output of chosen tokens; the
thinker then runs on normally. Pushes are in units of v's score: push = kappa · v/‖v‖² (v·push = kappa; a typical
drop moves v·z by ≈1). Controls have the same norm: a random direction, and the path probe's direction.
  path targets: path tokens of a 4-cell stretch of the remaining (not visible) solution
     +v (drop it?), random, −path-probe direction
  drop targets: the tokens the thinker itself drops in step 1 (strict drops at t=70; side branches)
     −v (keep it?), random, +path-probe direction
Scored on the odd boards: the thinker's path probability of the targets after step 1 at t=70 (z_L of the last z_L
update of step 1, final-update path probe from drop_concept.py), and the painter's output: target cells / tokens
drawn, the rest of the remaining path, off-path cells.
"""
import dataclasses

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

G, P = 12, 12
# (name, direction, sign, target, kappa)
RUNS = ([("normal", None, 0, None, 0.0)]
        + [("v_path", "v", 1, "path", k) for k in (1.0, 3.0, 6.0)]
        + [("rand_path", "rand", 1, "path", 6.0), ("probe_path", "probe", -1, "path", 6.0)]
        + [("v_drop", "v", -1, "drop", k) for k in (3.0, 6.0)]
        + [("rand_drop", "rand", 1, "drop", 6.0), ("probe_drop", "probe", 1, "drop", 6.0)])


def stretch_tokens(ms, rec, cats):
    rem = [c for c in ms.path if c in rec["remaining"]]
    cells = rem[2:6] if len(rem) >= 6 else rem[:4]
    m = np.zeros(G * G, bool)
    for cid in cells:
        y0, y1, x0, x1 = hs.cell_box(ms, cid)
        for tr in range(G):
            for tc in range(G):
                oy = min(y1, (tr + 1) * P) - max(y0, tr * P); ox = min(x1, (tc + 1) * P) - max(x0, tc * P)
                if oy >= 6 and ox >= 6 and cats[tr * G + tc] != "off":
                    m[tr * G + tc] = True
    return cells, m


@torch.no_grad()
def denoise(model, conditions, x, run_ts, generator, delta=None, n_push=6, rec_call=None, record=None):
    """CFG; in the conditional pass, add delta (B, T, D) to the L_level output of the first n_push calls
    (= H-cycle 1's z_L updates of reasoning step 1). record[t] = L_level output of call rec_call at t=70."""
    predictor = model.sampling_pipeline.predictor
    state = {"on": False, "n": 0, "t": None}

    def hook(module, inputs, output):
        if not state["on"]:
            return None
        c = state["n"]; state["n"] += 1
        if record is not None and c == rec_call and state["t"] == 70:
            record[70] = output.detach().float().cpu()
        if delta is not None and c < n_push:
            return output + delta.to(output.device, output.dtype)
        return None
    h = model.thinker.inner.L_level.register_forward_hook(hook)
    try:
        for t in run_ts:
            s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))
            state.update(on=True, n=0, t=int(t))
            pred_cond, _, _ = model.forward_with_carry(s)
            state["on"] = False
            pred_uncond = model(model.null_condition_sample(s)).pred
            pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
            x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    finally:
        h.remove()
    return x


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
    tc = model.thinker.inner.config
    per = tc.L_cycles + 1
    n_push, rec_call = tc.L_cycles, tc.H_cycles * per - 2                  # H-cycle 1's z_L updates; last z_L update of step 1

    dc = torch.load(str(ic.drop_concept), weights_only=False)
    v, w_path, b_path, drop_mask = dc["v"], dc["w_path"], dc["b_path"], dc["drop_mask"].numpy()
    D = v.numel()
    unit = {"v": v / v.norm() ** 2, "probe": w_path / w_path.norm() / v.norm()}   # kappa=1 -> the same norm as v's push
    g = torch.Generator().manual_seed(0)
    r = torch.randn(D, generator=g); unit["rand"] = r / r.norm() / v.norm()
    print(f"‖push‖ per kappa: {float(unit['v'].norm()):.2f} (token z_L norm ≈ 22); cos(v, probe) "
          f"{float(v @ w_path / v.norm() / w_path.norm()):+.2f}", flush=True)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    rows, outs, n_board = [], [], 0
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        idx = np.arange(n_board, n_board + B)
        masks = {"drop": drop_mask[idx]}
        cells_p, mp = zip(*[stretch_tokens(ms, r_, b["cats"][i]) for i, (ms, r_) in enumerate(zip(mss, recs))])
        masks["path"] = np.stack(mp)
        batch_out = {}
        for name, dkey, sign, tgt, kappa in RUNS:
            delta = None
            if dkey is not None:
                m = torch.from_numpy(masks[tgt]).float()[..., None]
                delta = m * (sign * kappa * unit[dkey])[None, None]
            record = {}
            x = denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1),
                        delta, n_push, rec_call, record)
            out = model.decode_for_eval(x).clamp(0, 1)
            out_u8 = (out * 255).byte().cpu()
            batch_out[f"{name}_{kappa:g}"] = out_u8
            pz = torch.sigmoid(record[70] @ w_path + b_path).numpy()            # (B, T) path prob after step 1, t=70
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                oc = pa.token_labels(out_u8[i], out_u8[i])[0]
                rest = r_["remaining"] - set(cells_p[i])
                for tg in ("path", "drop"):
                    mk = masks[tg][i]
                    if not mk.any():
                        continue
                    row = {"run": name, "kappa": kappa, "target": tg, "pushed": tg == tgt, "board": int(idx[i]),
                           "held_out": bool(idx[i] % 2 == 1), "n_tok": int(mk.sum()),
                           "p_step1": float(pz[i][mk].mean()), "tok_drawn": float((oc[mk] != "off").mean()),
                           "rest_remaining_drawn": len(rest & pred) / len(rest) if rest else None,
                           "off_path": len(pred - r_["path"])}
                    if tg == "path":
                        row["cells_drawn"] = len(set(cells_p[i]) & pred) / len(cells_p[i])
                    rows.append(row)
        outs.append({"n": B, "sol": b["sol_u8"], "clean": b["clean_u8"], "out": batch_out,
                     "masks": {k: torch.from_numpy(v_) for k, v_ in masks.items()}})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    print("\nheld-out boards (odd); p_step1 = thinker's path prob of the targets after step 1 (t=70); tok_drawn = targets "
          "with blue in the output")
    for tg in ("path", "drop"):
        for name, dkey, sign, tgt, kappa in RUNS:
            if tgt not in (None, tg):
                continue
            sel = [r_ for r_ in rows if r_["run"] == name and r_["kappa"] == kappa and r_["target"] == tg and r_["held_out"]]
            m = lambda k: np.mean([r_[k] for r_ in sel if r_.get(k) is not None])
            extra = f" cells drawn {m('cells_drawn'):.2f}" if tg == "path" else ""
            print(f"  [{tg:4s}] {name:10s} kappa={kappa:g} n={len(sel):3d} | p_step1 {m('p_step1'):.3f} | tokens drawn {m('tok_drawn'):.2f}"
                  f"{extra} | rest of remaining drawn {m('rest_remaining_drawn'):.2f}  off-path cells {m('off_path'):.1f}")
    torch.save({"rows": rows, "batches": outs, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
