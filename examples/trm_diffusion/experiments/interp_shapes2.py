"""TEMPORARY diagnostic (cluster only): (a) solve rate on maze size 10, which was held out from training (all shapes);
(b) saved outputs for visual checks of the failure cases (circle n5/n8, hexagon n16, square n16, square n10).
Same construction and scoring as interp_shapes.py; full budget; prefix (t=70) and noprefix (pure noise).
"""
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_budget as ib
import interp_hv_swap as hs
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

SETS = [("square", 10), ("hexagon", 10), ("triangle", 10), ("circle", 10), ("circle", 5), ("circle", 8), ("hexagon", 16), ("square", 16)]
SAVE = {("circle", 5), ("circle", 8), ("hexagon", 16), ("square", 16), ("square", 10)}
N_BOARDS, N_SAVE = 48, 8


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
    rows, saved = [], {}
    for shape, size in SETS:
        ds = AmazeDataset(dataset_path=f"{ic.data_root}/{shape}/n{size}_test.parquet", split="test", image_size=hs.IMG,
                          condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                          num_channels=3, include_metadata=True)
        kept = 0
        for start in range(0, len(ds), 16):
            if kept >= N_BOARDS:
                break
            b = hs.prepare(ds, model, device, start, len(ds), 16, 0)
            if b is None:
                continue
            g = torch.Generator(device=device).manual_seed(b["base_seed"] + 7)
            x_pure = torch.randn(b["x_t"].shape, generator=g, device=device)
            for setting in ("prefix", "noprefix"):
                if setting == "prefix":
                    out = ib.denoise(model, b["conditions"], b["x_t"], ts70, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), (None, None, None))
                else:
                    out = ib.denoise(model, b["conditions"], x_pure, all_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 3), (None, None, None))
                for i, (ms, r_) in enumerate(zip(b["mss"], b["recs"])):
                    pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                    inner = r_["path"] - {ms.path[0], ms.path[-1]}
                    target = r_["remaining"] if setting == "prefix" else r_["path"]
                    rows.append({"shape": shape, "size": size, "setting": setting, "solved": inner <= pred and not (pred - r_["path"]),
                                 "path_drawn": len(target & pred) / max(1, len(target)), "off_path": len(pred - r_["path"]),
                                 "n_cells": len(ms.id2rc), "path_len": len(ms.path)})
                key = (shape, size)
                if key in SAVE and start == 0:
                    sv = saved.setdefault(f"{shape}_n{size}", {"maze": b["maze_u8"][:N_SAVE], "sol": b["sol_u8"][:N_SAVE], "clean": b["clean_u8"][:N_SAVE]})
                    sv[f"out_{setting}"] = (out[:N_SAVE] * 255).byte().cpu()
                    sv[f"pred_{setting}"] = [sorted(mcl.predicted_cells(ai._to_native(out[i], ms), ms)) for i, ms in enumerate(b["mss"][:N_SAVE])]
                    sv["path"] = [ms.path for ms in b["mss"][:N_SAVE]]
                    sv["n_cells"] = [len(ms.id2rc) for ms in b["mss"][:N_SAVE]]
            kept += b["B"]
        for setting in ("prefix", "noprefix"):
            s2 = [r for r in rows if r["shape"] == shape and r["size"] == size and r["setting"] == setting]
            print(f"  {shape:8s} n{size:<2d} {setting:8s} n={len(s2):3d} | solved {np.mean([r['solved'] for r in s2]):.2f}  "
                  f"path drawn {np.mean([r['path_drawn'] for r in s2]):.3f}  off-path cells {np.mean([r['off_path'] for r in s2]):.2f}  "
                  f"(cells per maze {np.mean([r['n_cells'] for r in s2]):.0f}, solution length {np.mean([r['path_len'] for r in s2]):.0f})", flush=True)
    torch.save({"rows": rows, "saved": saved}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
