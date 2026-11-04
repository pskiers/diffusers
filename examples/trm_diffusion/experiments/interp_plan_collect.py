"""TEMPORARY diagnostic (cluster only): collect the thinker's output to the painter for every board.

For each board (corrupted + clean twin, the same boards, noise and sampler seed as amaze_interpretability.py),
denoise from t=70 and keep, at t in RECORD, the conditional pass's final thinker logits (B, 144, vocab) and
final z_H (B, 144, hidden), plus the images needed to label tokens: maze, both boards, GT solution.
Everything is analysed offline (CPU).
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

RECORD = {70, 50, 35, 20, 5}


def to_u8(x):  # (B, C, H, W) in [0, 1] -> uint8 on CPU
    return (x.detach().float().clamp(0, 1) * 255).byte().cpu()


@torch.no_grad()
def denoise_recording(model, conditions, x, run_ts, generator, out: dict):
    predictor = model.sampling_pipeline.predictor
    for t in run_ts:
        tb = t.expand(x.shape[0]).to(x.device)
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=tb)
        pred_cond, z_H, _ = model.forward_with_carry(s)
        if int(t) in RECORD:
            out[int(t)] = {"logits": pred_cond.logits.float().cpu(), "z_H": z_H.half().cpu()}
        pred_uncond = model(model.null_condition_sample(s)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
        x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
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

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    batches = []
    for start in range(0, n_total, batch_size):
        base_seed = seed + start * 977
        gen = torch.Generator(device=device).manual_seed(base_seed)
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, ms) for s, ms in ((s, mcl.build_maze_sample(s.metadata)) for s in raw) if ms is not None]
        kept, corr_imgs, clean_imgs = [], [], []
        for i, (s, ms) in enumerate(pairs):
            rng = random.Random(base_seed + i * 131)
            board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
            img_c, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)
            if not wrong:
                continue
            img_k = board.copy()
            full = [shown[-1]] + gt_walked
            for x, y in zip(full[:-1], full[1:]):
                cv2.line(img_k, mcl._center(ms.id2rc[x], ms.cell_px), mcl._center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
            kept.append(s); corr_imgs.append(img_c); clean_imgs.append(img_k)
        if not kept:
            continue
        B = len(kept)
        batch = AmazeDataset.collate_fn(kept)
        conditions = model._batch_to_sample(batch, device)
        ct = torch.stack([ds.transform(Image.fromarray(im)) for im in corr_imgs]).to(device)
        kt = torch.stack([ds.transform(Image.fromarray(im)) for im in clean_imgs]).to(device)
        noise = torch.randn(ct.shape, generator=gen, device=device)
        tb = torch.full((B,), 70, device=device, dtype=torch.long)
        entry = {"n": B, "corrupted_img": to_u8(ct), "clean_img": to_u8(kt),
                 "solution_img": to_u8(batch.images), "maze_img": to_u8(batch.spatial_conditions)}
        for name, img in (("corrupted", ct), ("clean", kt)):
            rec = {}
            x = denoise_recording(model, conditions, model.scheduler.add_noise(img, noise, tb), run_ts,
                                  torch.Generator(device=device).manual_seed(base_seed + 1), rec)
            entry[name] = rec
            entry[f"{name}_out"] = to_u8(model.decode_for_eval(x))
        batches.append(entry)
        print(f"batch start={start}: {B} boards", flush=True)
    torch.save(batches, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
