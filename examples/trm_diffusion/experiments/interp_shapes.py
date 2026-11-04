"""TEMPORARY diagnostic (cluster only): E3b — where does PaTh fail naturally? Solve rate over maze shapes and sizes.

Full budget (as trained). Test parquets of the main data directory: shapes square / hexagon / triangle / circle,
sizes in SIZES (trained on 5, 7, 8, 9, 11, 13, 16; 10 held out), up to N_BOARDS boards each (same construction as
before: boards where apply_add finds a dead-end; 50% prefix). Settings: prefix (t=70 start) and noprefix (pure
noise, all steps). Scoring is shape-agnostic (cells from the dataset's cell map): solved = every solution cell except
start/goal drawn and no off-path cell drawn. Per-board outcomes saved for the success-vs-failure contrast (E4).
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

SHAPES = ("square", "hexagon", "triangle", "circle")
SIZES = (5, 8, 11, 13, 16)
N_BOARDS = 48


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
    rows = []
    for shape in SHAPES:
        for size in SIZES:
            path = f"{ic.data_root}/{shape}/n{size}_test.parquet"
            ds = AmazeDataset(dataset_path=path, split="test", image_size=hs.IMG,
                              condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                              num_channels=3, include_metadata=True)
            n_total, kept = len(ds), 0
            for start in range(0, n_total, 16):
                if kept >= N_BOARDS:
                    break
                try:
                    b = hs.prepare(ds, model, device, start, n_total, 16, 0)
                except Exception as e:                                   # a shape the corruption code cannot handle
                    print(f"  {shape} n{size} batch {start}: prepare failed ({type(e).__name__}: {e})", flush=True)
                    b = None
                if b is None:
                    continue
                mss, recs = b["mss"], b["recs"]
                g = torch.Generator(device=device).manual_seed(b["base_seed"] + 7)
                x_pure = torch.randn(b["x_t"].shape, generator=g, device=device)
                for setting in ("prefix", "noprefix"):
                    if setting == "prefix":
                        out = ib.denoise(model, b["conditions"], b["x_t"], ts70, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), (None, None, None))
                    else:
                        out = ib.denoise(model, b["conditions"], x_pure, all_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 3), (None, None, None))
                    for i, (ms, r_) in enumerate(zip(mss, recs)):
                        pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                        inner = r_["path"] - {ms.path[0], ms.path[-1]}
                        target = r_["remaining"] if setting == "prefix" else r_["path"]
                        rows.append({"shape": shape, "size": size, "setting": setting, "idx": start + i,
                                     "solved": inner <= pred and not (pred - r_["path"]),
                                     "path_drawn": len(target & pred) / max(1, len(target)), "off_path": len(pred - r_["path"]),
                                     "path_len": len(ms.path), "cell_px": 144 / max(ms.height, ms.width)})
                kept += b["B"]
            sel = [r for r in rows if r["shape"] == shape and r["size"] == size]
            for setting in ("prefix", "noprefix"):
                s2 = [r for r in sel if r["setting"] == setting]
                if s2:
                    print(f"  {shape:8s} n{size:<2d} {setting:8s} n={len(s2):3d} | solved {np.mean([r['solved'] for r in s2]):.2f}  "
                          f"path drawn {np.mean([r['path_drawn'] for r in s2]):.3f}  off-path cells {np.mean([r['off_path'] for r in s2]):.2f}  "
                          f"(cell ≈ {np.mean([r['cell_px'] for r in s2]):.1f} px at 144)", flush=True)
    torch.save({"rows": rows}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
