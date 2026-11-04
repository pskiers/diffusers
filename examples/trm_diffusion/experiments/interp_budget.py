"""TEMPORARY diagnostic (cluster only): E3 — where does the thinker start to fail? Inference-only budget sweep.

The conditional pass's thinker runs with a reduced budget (n_sup reasoning steps x H_cycles x (L_cycles z_L updates
+ 1 z_H update)); the unconditional pass is unchanged. Two settings, same 98 clean boards:
  prefix   50% of the solution drawn, noised to t=70, denoised from there (the setting of all earlier experiments)
  noprefix empty board (start marker only), pure noise, all denoising steps (standard generation)
Per board: solved = every solution cell except start/goal is drawn and no off-path cell is drawn; remaining path drawn,
off-path cells. Saves the per-board outcome for each (setting, budget) for the success-vs-failure contrast (E4).
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import amaze_interpretability as ai
import interp_hv_swap as hs
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

BUDGETS = [(16, 3, 6), (4, 3, 6), (1, 3, 6), (1, 2, 6), (1, 1, 6), (1, 1, 4), (1, 1, 3), (1, 1, 2), (1, 1, 1),
           (2, 1, 2), (4, 1, 1), (1, 3, 1), (1, 2, 2)]


@torch.no_grad()
def denoise(model, conditions, x, ts, generator, budget):
    predictor = model.sampling_pipeline.predictor
    n, h, l = budget
    for t in ts:
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))
        pred_cond, _, _ = model.forward_with_carry(s, n_sup=n, H_cycles=h, L_cycles=l)
        pred_uncond = model(model.null_condition_sample(s)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
        x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    return model.decode_for_eval(x).clamp(0, 1)


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
    all_ts = model.scheduler.timesteps
    ts70 = all_ts[all_ts <= 70]
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
        g = torch.Generator(device=device).manual_seed(b["base_seed"] + 7)
        x_pure = torch.randn(b["x_t"].shape, generator=g, device=device)
        for setting in ("prefix", "noprefix"):
            for budget in BUDGETS:
                if setting == "prefix":
                    out = denoise(model, b["conditions"], b["x_t"], ts70, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), budget)
                else:
                    out = denoise(model, b["conditions"], x_pure, all_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 3), budget)
                for i, (ms, r_) in enumerate(zip(mss, recs)):
                    pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                    inner = r_["path"] - {ms.path[0], ms.path[-1]}
                    target = r_["remaining"] if setting == "prefix" else r_["path"]
                    rows.append({"setting": setting, "budget": budget, "board": n_board + i,
                                 "solved": inner <= pred and not (pred - r_["path"]),
                                 "path_drawn": len(target & pred) / max(1, len(target)), "off_path": len(pred - r_["path"])})
            print(f"batch start={start}: {B} boards, setting {setting} done", flush=True)
        n_board += B
    print(f"\n{n_board} boards. budget = (n_sup, H_cycles, L_cycles); thinker block calls per pass = n_sup*H*(L+1)")
    for setting in ("prefix", "noprefix"):
        for budget in BUDGETS:
            sel = [r for r in rows if r["setting"] == setting and r["budget"] == budget]
            n, h, l = budget
            print(f"  {setting:8s} budget {str(budget):11s} calls {n * h * (l + 1):4d} | solved {np.mean([r['solved'] for r in sel]):.2f}  "
                  f"path drawn {np.mean([r['path_drawn'] for r in sel]):.3f}  off-path cells {np.mean([r['off_path'] for r in sel]):.2f}")
    torch.save({"rows": rows, "budgets": BUDGETS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
