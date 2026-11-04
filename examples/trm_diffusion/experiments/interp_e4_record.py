"""TEMPORARY diagnostic (cluster only): E4a — successes vs failures of the same maze in a regime where the thinker
sometimes fails: budget (n_sup=1, H_cycles=2, L_cycles=6) = 14 block calls per pass (59% solved with the prefix).

Same 98 clean boards (50% prefix, t=70 start); SEEDS noise seeds per board (noise at t=70 + sampler). Per run:
which cells the painter draws (solved / missing path cells / extra off-path cells), and the thinker's state after
each of the 14 calls of the conditional pass at t=70 (fp16, for the success-vs-failure contrast).
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
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

BUDGET = (1, 2, 6)
SEEDS = 8


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
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    rec = {"on": False, "states": []}
    model.thinker.inner.L_level.register_forward_hook(lambda m, i, o: rec["states"].append(o.detach().half().cpu()) if rec["on"] else None)
    rows, states, meta, n_board = [], [], [], 0
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        kt = b["clean_u8"].float().to(device) / 255.0
        per_seed = []
        for k in range(SEEDS):
            if k == 0:
                x = b["x_t"]
            else:
                g = torch.Generator(device=device).manual_seed(b["base_seed"] + 5000 + k)
                x = model.scheduler.add_noise(kt, torch.randn(kt.shape, generator=g, device=device), torch.full((B,), 70, device=device, dtype=torch.long))
            gen = torch.Generator(device=device).manual_seed(b["base_seed"] + 1 + 10 * k)
            with torch.no_grad():
                for t in ts:
                    s = dataclasses.replace(b["conditions"], x_noisy=x, timesteps=t.expand(B).to(device))
                    rec.update(on=int(t) == 70, states=[] if int(t) == 70 else rec["states"])
                    pred_cond, _, _ = model.forward_with_carry(s, n_sup=BUDGET[0], H_cycles=BUDGET[1], L_cycles=BUDGET[2])
                    if int(t) == 70:
                        per_seed.append(torch.stack(rec["states"], 1))            # (B, 14, T, D)
                    rec["on"] = False
                    pred_uncond = model(model.null_condition_sample(s)).pred
                    pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                    x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
                out = model.decode_for_eval(x).clamp(0, 1)
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred_c = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                inner = r_["path"] - {ms.path[0], ms.path[-1]}
                rows.append({"board": n_board + i, "seed": k, "solved": inner <= pred_c and not (pred_c - r_["path"]),
                             "missing": sorted(inner - pred_c), "extra": sorted(pred_c - r_["path"]),
                             "remaining_drawn": len(r_["remaining"] & pred_c) / max(1, len(r_["remaining"]))})
        states.append(torch.stack(per_seed, 1))                                    # (B, SEEDS, 14, T, D)
        for i, (ms, r_) in enumerate(zip(mss, recs)):
            meta.append({"board": n_board + i, "path": ms.path, "id2rc": ms.id2rc, "visible": sorted(r_["visible"]),
                         "height": ms.height, "width": ms.width,
                         "open": {cid: [n for n in ms.rc2id and [] or []] for cid in []}})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)
    sol = np.array([[r["solved"] for r in rows if r["board"] == i] for i in range(n_board)])   # (N, SEEDS)
    rate = sol.mean(1)
    print(f"\nbudget {BUDGET}: solved {sol.mean():.2f} over {n_board} boards x {SEEDS} seeds")
    print(f"  boards always solved {np.mean(rate == 1):.2f}, never solved {np.mean(rate == 0):.2f}, mixed {np.mean((rate > 0) & (rate < 1)):.2f}")
    print(f"  per-board solve-rate histogram (0, 1/8, ..., 1): " + " ".join(str(int(((rate * SEEDS).round() == j).sum())) for j in range(SEEDS + 1)))
    fails = [r for r in rows if not r["solved"]]
    print(f"  failures: {len(fails)}; with missing path cells {np.mean([len(r['missing']) > 0 for r in fails]):.2f}, "
          f"with extra off-path cells {np.mean([len(r['extra']) > 0 for r in fails]):.2f}; mean missing {np.mean([len(r['missing']) for r in fails]):.2f}, "
          f"mean extra {np.mean([len(r['extra']) for r in fails]):.2f}")
    torch.save({"rows": rows, "meta": meta, "budget": BUDGET, "seeds": SEEDS}, f"{ic.out_dir}/e4_runs.pt")
    torch.save(torch.cat(states), f"{ic.out_dir}/e4_states_t70.pt")
    print("saved", ic.out_dir)


if __name__ == "__main__":
    main()
