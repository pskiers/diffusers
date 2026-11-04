"""TEMPORARY diagnostic (cluster only): what happens inside the thinker's first reasoning step?

Normal CFG denoising of the clean boards (same boards / noise / seed as amaze_interpretability.py). At t=70, in the
conditional pass, a hook on L_level keeps the per-token output of the first n_calls block calls (= the whole first
reasoning step: H_cycles x (L_cycles z_L updates + 1 z_H update)), plus the final z_H / z_L of the pass (after all
n_sup steps) to train the read-out probes on.
"""
import dataclasses
import random

import cv2
import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import amaze_interpretability as ai
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

AT_T = 70


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
    n_calls = tc.H_cycles * (tc.L_cycles + 1)
    print(f"first reasoning step = {n_calls} L_level calls ({tc.H_cycles} H-cycles x ({tc.L_cycles} z_L + 1 z_H))")

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    saved = []
    for start in range(0, n_total, batch_size):
        base_seed = seed + start * 977
        gen = torch.Generator(device=device).manual_seed(base_seed)
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, ms) for s, ms in ((s, mcl.build_maze_sample(s.metadata)) for s in raw) if ms is not None]
        kept, clean_imgs = [], []
        for i, (s, ms) in enumerate(pairs):
            rng = random.Random(base_seed + i * 131)
            board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
            _, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)  # same rng use as the main script
            if not wrong:
                continue
            img_k = board.copy()
            full = [shown[-1]] + gt_walked
            for x, y in zip(full[:-1], full[1:]):
                cv2.line(img_k, mcl._center(ms.id2rc[x], ms.cell_px), mcl._center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
            kept.append(s); clean_imgs.append(img_k)
        if not kept:
            continue
        B = len(kept)
        batch = AmazeDataset.collate_fn(kept)
        conditions = model._batch_to_sample(batch, device)
        kt = torch.stack([ds.transform(Image.fromarray(im)) for im in clean_imgs]).to(device)
        noise = torch.randn(kt.shape, generator=gen, device=device)
        x = model.scheduler.add_noise(kt, noise, torch.full((B,), 70, device=device, dtype=torch.long))
        generator = torch.Generator(device=device).manual_seed(base_seed + 1)
        predictor = model.sampling_pipeline.predictor
        entry = {"n": B, "clean_img": (kt.clamp(0, 1) * 255).byte().cpu(),
                 "solution_img": (batch.images.float().clamp(0, 1) * 255).byte().cpu(),
                 "maze_img": (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()}
        with torch.no_grad():
            for t in run_ts:
                s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(B).to(device))
                calls, handle = [], None
                if int(t) == AT_T:
                    def hook(module, inputs, output):
                        if len(calls) < n_calls:
                            calls.append(output.detach().half().cpu())
                    handle = model.thinker.inner.L_level.register_forward_hook(hook)
                pred_cond, z_H, z_L = model.forward_with_carry(s)
                if handle is not None:
                    handle.remove()
                    entry["calls"] = torch.stack(calls, 1)                 # (B, n_calls, T, D): first reasoning step
                    entry["z_H_final"], entry["z_L_final"] = z_H.half().cpu(), z_L.half().cpu()
                pred_uncond = model(model.null_condition_sample(s)).pred
                pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
        saved.append(entry)
        print(f"batch start={start}: {B} boards", flush=True)
    torch.save({"batches": saved, "n_calls": n_calls, "L_cycles": tc.L_cycles}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
