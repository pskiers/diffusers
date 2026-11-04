"""TEMPORARY diagnostic (cluster only): how much of the plan code depends on the particular noise the thinker sees?

Clean boards, normal CFG runs with the original noise / sampler seed and with two other seeds (noise at t=70 and the
sampler). Records the painter's conditional logits per t. Offline: per t, the share of the board-specific variance
(logits − per-position mean) that changes across seeds of the same board (= depends on the noise), and how well
the known features explain the seed-averaged code.
"""
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import interp_hv_swap as hs
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

SEEDS = (0, 1000, 2000)


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
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    saved = []
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        kt = b["clean_u8"].float().to(device) / 255.0
        logits = []
        for s in SEEDS:
            if s == 0:
                x_t, gs = b["x_t"], b["base_seed"] + 1
            else:
                g = torch.Generator(device=device).manual_seed(b["base_seed"] + s)
                noise = torch.randn(kt.shape, generator=g, device=device)
                x_t = model.scheduler.add_noise(kt, noise, torch.full((b["B"],), 70, device=device, dtype=torch.long))
                gs = b["base_seed"] + s + 1
            rec = []
            hs.denoise(model, b["conditions"], x_t, run_ts, torch.Generator(device=device).manual_seed(gs), record=rec)
            logits.append(torch.stack(rec, 1))                                       # (B, n_t, 144, 11)
        saved.append({"n": b["B"], "logits": torch.stack(logits, 1).half(),             # (B, n_seeds, n_t, 144, 11)
                      "sol": b["sol_u8"], "clean": b["clean_u8"], "maze": b["maze_u8"]})
        print(f"batch start={start}: {b['B']} boards", flush=True)
    torch.save({"batches": saved, "ts": ts, "seeds": SEEDS}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
