"""TEMPORARY diagnostic (cluster only): does a z_H edit act through the thinker, or only at the final read-out?

Same edits as interp_z_edit.py (per-call mean-difference path direction, conditional pass, every denoising step),
but applied only in a WINDOW of the 336 calls of each pass:
  all          every z_H update (16 steps x 3)                       — reproduces interp_z_edit.py
  no_last      all but the very last z_H update (the one lm_head reads)
  steps1_15    the z_H updates of reasoning steps 1–15 (none in step 16)
  steps1_8     reasoning steps 1–8
  step1        reasoning step 1 only (3 z_H updates)
  step16       reasoning step 16 only (3 z_H updates)
  last_only    only the final z_H update (= a read-out edit, like pushing the logits)
plus z_L edits in steps 1–15. If the effect survives without the last update(s), the thinker carries the edit
forward through its own later calls; if it needs the last update, z_H matters only at the read-out.
"""
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_az_push as az
import interp_hv_swap as hs
import interp_z_edit as ze
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

# (name, states, target, sign, alpha, window)
RUNS = ([("normal", (), None, 0, 0.0, None)]
        + [("zH", ("z_H",), "path", -1, 2.0, w) for w in ("all", "no_last", "steps1_15", "steps1_8", "step1", "step16", "last_only")]
        + [("zL", ("z_L",), "path", -1, 2.0, "steps1_15")]
        + [("zH", ("z_H",), "drop", 1, 4.0, w) for w in ("all", "steps1_15", "last_only")])


def in_window(c, w, per_step, n_calls):
    s = c // per_step + 1                                                      # reasoning step 1..16
    return {"all": True, "no_last": c != n_calls - 1, "steps1_15": s <= 15, "steps1_8": s <= 8, "step1": s == 1,
            "step16": s == 16, "last_only": c == n_calls - 1}[w]


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
    per_step = tc.H_cycles * per
    kind_of = lambda c: "z_H" if c % per == per - 1 else "z_L"
    drop_mask = torch.load(str(ic.drop_concept), weights_only=False)["drop_mask"].numpy()
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    starts = list(range(0, n_total, batch_size))

    # phase 1 (as interp_z_edit.py): per (t, call) mean path − mean off state on the even boards
    S, cnt, n_board = {}, [0.0, 0.0], 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        even = torch.from_numpy(np.array([(n_board + i) % 2 == 0 for i in range(b["B"])]))
        gt = torch.from_numpy(b["cats"] != "off")
        mp = (gt & even[:, None]).float().to(device)[..., None]; mo = ((~gt) & even[:, None]).float().to(device)[..., None]

        def acc(t, c, out):
            o = out.float(); sp, so = (o * mp).sum((0, 1)), (o * mo).sum((0, 1))
            if (t, c) in S:
                S[(t, c)][0] += sp; S[(t, c)][1] += so
            else:
                S[(t, c)] = [sp, so]
        ze.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), acc, b["B"])
        cnt[0] += float(mp.sum()); cnt[1] += float(mo.sum())
        n_board += b["B"]
    D = {k: v[0] / cnt[0] - v[1] / cnt[1] for k, v in S.items()}
    n_calls = max(c for _, c in D) + 1
    print(f"{n_calls} calls per pass, {per_step} per reasoning step; last z_H update = call {n_calls}", flush=True)

    rows, n_board = [], 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        idx = np.arange(n_board, n_board + B)
        cells_p, mp_ = zip(*[az.stretch_tokens(ms, r_, b["cats"][i]) for i, (ms, r_) in enumerate(zip(mss, recs))])
        masks = {"path": torch.from_numpy(np.stack(mp_)), "drop": torch.from_numpy(drop_mask[idx])}
        for name, states, tgt, sign, alpha, w in RUNS:
            m = None if tgt is None else masks[tgt].to(device)[..., None].float()

            def fn(t, c, out, states=states, m=m, sign=sign, alpha=alpha, w=w):
                if not states or kind_of(c) not in states or not in_window(c, w, per_step, n_calls):
                    return None
                return (out.float() + m * (sign * alpha) * D[(t, c)]).to(out.dtype)
            out = ze.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), fn, B)
            out_u8 = (out * 255).byte().cpu()
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                oc = pa.token_labels(out_u8[i], out_u8[i])[0]
                rest = r_["remaining"] - set(cells_p[i])
                dm = masks["drop"][i].numpy()
                rows.append({"run": name, "target": tgt, "alpha": alpha, "window": w, "held_out": bool(idx[i] % 2 == 1),
                             "stretch_drawn": len(set(cells_p[i]) & pred) / max(1, len(cells_p[i])),
                             "drop_drawn": float((oc[dm] != "off").mean()) if dm.any() else None,
                             "rest_remaining_drawn": len(rest & pred) / len(rest) if rest else None,
                             "off_path": len(pred - r_["path"])})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    keys = ["stretch_drawn", "drop_drawn", "rest_remaining_drawn", "off_path"]
    print("\nheld-out boards (odd)")
    for name, states, tgt, sign, alpha, w in RUNS:
        sel = [r_ for r_ in rows if r_["run"] == name and r_["target"] == tgt and r_["window"] == w and r_["held_out"]]
        print(f"  {name:6s} target={str(tgt):5s} alpha={alpha:g} window={str(w):9s} n={len(sel)} | " + "  ".join(
            f"{k} {np.mean([r_[k] for r_ in sel if r_[k] is not None]):.3f}" for k in keys))
    torch.save({"rows": rows, "runs": RUNS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
