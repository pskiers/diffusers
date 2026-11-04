"""TEMPORARY diagnostic (cluster only, not part of the repo): does the SteerZL hook's returned value reach
the thinker's output, and how much does each intervention change the logits / the final image?

One batch of corrupted boards, as amaze_interpretability.py builds them. Compares against the normal run:
  - one thinker pass at t=70: relative change of the logits and of z_H
  - the full denoise from t=70: mean |pixel difference| of the final image, and the drawn-cell metrics
for: a destructive hook (z_L := 0 at every update), the local ablation x5, and the random-direction x5.
"""
import dataclasses
import random

import cv2
import hydra
import numpy as np
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


class ZeroZL:
    """Destructive control: replace every z_L update with zeros (z_H updates untouched)."""
    def __init__(self, per_cycle):
        self.per_cycle, self.call = per_cycle, 0

    def start(self, t):
        self.call = 0

    def __call__(self, module, inputs, output):
        call, self.call = self.call, self.call + 1
        if call % self.per_cycle == self.per_cycle - 1:
            return None
        return torch.zeros_like(output)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    print("L_level type:", type(model.thinker.inner.L_level))
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)

    per_cycle = model.thinker.inner.config.L_cycles + 1
    bm = torch.load(str(ic.states))
    v, gap = ai.local_dead_end_direction(bm[0::2], 70, [int(t) for t in run_ts])
    u = torch.randn(v.shape, generator=torch.Generator().manual_seed(0)); u = u / u.norm()
    v, u = v.to(device), u.to(device)
    alphas = (5, 10, 20, 50)
    agg = {}
    for start in (16, 48, 80):  # half-B batches (odd batch index): v_local came from the other half
        seed = 0
        base_seed = seed + start * 977
        gen = torch.Generator(device=device).manual_seed(base_seed)
        raw = [ds[i] for i in range(start, start + 16)]
        pairs = [(s_, ms) for s_, ms in ((s_, mcl.build_maze_sample(s_.metadata)) for s_ in raw) if ms is not None]
        kept, corr_imgs, clean_imgs, mss, recs = [], [], [], [], []
        for i, (s_, ms) in enumerate(pairs):
            rng = random.Random(base_seed + i * 131)
            board, shown = mcl.render_partial(ms, ai.PREFIX_FRAC)
            img_c, wrong, gt_walked, _ = mcl.apply_add(board, ms, shown, rng)
            if not wrong:
                continue
            img_k = board.copy()
            full = [shown[-1]] + gt_walked
            for x, y in zip(full[:-1], full[1:]):
                cv2.line(img_k, mcl._center(ms.id2rc[x], ms.cell_px), mcl._center(ms.id2rc[y], ms.cell_px), ms.blue_color, ms.stroke)
            kept.append(s_); corr_imgs.append(img_c); clean_imgs.append(img_k); mss.append(ms)
            recs.append({"wrong": set(wrong), "shown": set(shown) | set(gt_walked), "path": set(ms.path)})
        B = len(kept)
        conditions = model._batch_to_sample(AmazeDataset.collate_fn(kept), device)
        ct = torch.stack([ds.transform(Image.fromarray(im)) for im in corr_imgs]).to(device)
        kt = torch.stack([ds.transform(Image.fromarray(im)) for im in clean_imgs]).to(device)
        noise = torch.randn(ct.shape, generator=gen, device=device)
        x_corr = model.scheduler.add_noise(ct, noise, torch.full((B,), 70, device=device, dtype=torch.long))
        dead_end = (ai._token_fraction((ct - kt).abs().amax(1) > 0.05, 12) > 0).float().to(device)
        # the same number of tokens per board, but away from the dead-end (>= 2 tokens from it): is it about *where*?
        grid = 12
        rc = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), -1).reshape(-1, 2).to(device)
        cheb = (rc[:, None] - rc[None]).abs().amax(-1)
        dist = torch.where(dead_end.bool()[:, None, :], cheb[None], grid).amin(-1)
        elsewhere = torch.zeros_like(dead_end)
        g = torch.Generator().manual_seed(base_seed)
        for i in range(B):
            cand = (dist[i] >= 2).nonzero().flatten().cpu()
            pick = cand[torch.randperm(len(cand), generator=g)[: int(dead_end[i].sum())]]
            elsewhere[i, pick.to(device)] = 1.0
        hooks = {"normal": None, "zero_z_L": ZeroZL(per_cycle)}
        for a in alphas:
            hooks[f"v_local_x{a}"] = ai.SteerZL(v, per_cycle, 288, mode="ablate", amount=gap, alpha=a, t_min=35, mask=dead_end)
            hooks[f"random_x{a}"] = ai.SteerZL(u, per_cycle, 288, mode="ablate", amount=gap, alpha=a, t_min=35, mask=dead_end)
        hooks["v_local_x50_elsewhere"] = ai.SteerZL(v, per_cycle, 288, mode="ablate", amount=gap, alpha=50, t_min=35, mask=elsewhere)

        outs = {}
        for name, h in hooks.items():
            x = ai._denoise_steered(model, conditions, x_corr, run_ts, torch.Generator(device=device).manual_seed(base_seed + 1), h)
            outs[name] = model.decode_for_eval(x).clamp(0, 1)
        for name, out in outs.items():
            a = agg.setdefault(name, {"de": [], "sh": [], "off": [], "px": []})
            for i, (ms, r) in enumerate(zip(mss, recs)):
                pc = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                a["de"].append(len(r["wrong"] & pc) / len(r["wrong"]))
                a["sh"].append(len(r["shown"] & pc) / len(r["shown"]))
                a["off"].append(len(pc - r["path"] - r["wrong"]))
                a["px"].append(float((out[i] - outs["normal"][i]).abs().mean()))
    for name, a in agg.items():
        print(f"[full denoise, n={len(a['de'])}] {name:24s} dead_end_kept {np.mean(a['de']):.3f}  shown_kept {np.mean(a['sh']):.3f}  "
              f"other_off_path {np.mean(a['off']):.2f}  mean|Δpixel| {np.mean(a['px']):.5f}")


if __name__ == "__main__":
    main()
