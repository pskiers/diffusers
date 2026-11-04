"""TEMPORARY diagnostic (Claude, amaze interpretability): is the size-10 failure the painter's own (it never saw 10x10 mazes)?

Square mazes of size 8, 10 (untrained) and 11, generated from pure noise, first N_BOARDS parseable boards, two runs each:
  normal         thinker + painter (CFG as in the eval)
  painter alone  null_steering in both CFG passes: the thinker's logits are replaced by zeros, the painter gets only its own
                 conditioning (the maze image)
Per board: walls faithful to the maze? (dark pixels added / erased vs the maze image, at 144 px); path: solved, solution
cells drawn, extra off-path cells. In the normal run, the final z_H of the conditional pass is kept at t in REC_T, to check
offline whether the thinker's plan on size 10 stays correct at later denoising steps.
If the painter alone already distorts the walls on size 10 but not on 8 / 11, its own image prior is size-specific.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

IMG, N_BOARDS, WALL_DARK = 144, 48, 80
SIZES = (8, 10, 11)
REC_T = (70, 35, 5)


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
    ts = model.scheduler.timesteps
    predictor = model.sampling_pipeline.predictor
    vd = cfg.data.get("val_dataset", {})
    results = {}
    for n in SIZES:
        ds = AmazeDataset(dataset_path=f"{ic.data_root}/square/n{n}_test.parquet", split="test", image_size=IMG,
                          condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                          num_channels=3, include_metadata=True)
        res = {"normal": [], "painter alone": []}
        zH = {t: [] for t in REC_T}; gts, saved = [], {}
        i0 = 0
        while len(gts) < N_BOARDS and i0 < len(ds):
            raw = [ds[i] for i in range(i0, min(i0 + 16, len(ds)))]
            i0 += 16
            pairs = [(s, mcl.build_maze_sample(s.metadata)) for s in raw]
            pairs = [(s, ms) for s, ms in pairs if ms is not None]
            if not pairs:
                continue
            samples, mss = [p[0] for p in pairs], [p[1] for p in pairs]
            B = len(samples)
            batch = AmazeDataset.collate_fn(samples)
            cond = model._batch_to_sample(batch, device)
            maze_u8 = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
            sol_u8 = (batch.images.float().clamp(0, 1) * 255).byte().cpu()
            x0 = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(1000 + i0), device=device)
            for run in ("normal", "painter alone"):
                null = run == "painter alone"
                x = x0
                gen = torch.Generator(device=device).manual_seed(2000 + i0)
                with torch.no_grad():
                    for t in ts:
                        s = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(B).to(device))
                        pred_cond, z_H, _ = model.forward_with_carry(s, null_steering=null)
                        if not null and int(t) in REC_T:
                            zH[int(t)].append(z_H.half().cpu())
                        pred_uncond = model(model.null_condition_sample(s), null_steering=null).pred
                        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                        x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
                    out = model.decode_for_eval(x).clamp(0, 1)
                out_u8 = (out * 255).byte().cpu()
                for i, ms in enumerate(mss):
                    od = out_u8[i].float().mean(0).numpy() < WALL_DARK
                    md = maze_u8[i].float().mean(0).numpy() < WALL_DARK
                    blue = mcl.blue_mask(out_u8[i].permute(1, 2, 0).numpy())
                    pred_c = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                    path = set(ms.path); inner = path - {ms.path[0], ms.path[-1]}
                    res[run].append({"walls_added": int((od & ~md & ~blue).sum()), "walls_erased": int((md & ~od).sum()),
                                     "maze_wall_px": int(md.sum()), "solved": inner <= pred_c and not (pred_c - path),
                                     "path_drawn": len(path & pred_c) / len(path), "extra": len(pred_c - path),
                                     "blue_px": int(blue.sum())})
                saved.setdefault(run, (out_u8[:6]))
            gts += [pa.token_labels(sol_u8[i], maze_u8[i])[0] != "off" for i in range(B)]
            saved.setdefault("maze", maze_u8[:6]); saved.setdefault("sol", sol_u8[:6])
        results[n] = {"rows": res, "zH": {t: torch.cat(v) for t, v in zH.items()}, "gt": np.stack(gts), "saved": saved}
        for run, rows in res.items():
            m = lambda k: np.mean([r[k] for r in rows])
            print(f"  n{n:<2d} {run:13s} n={len(rows):3d} | walls: added {m('walls_added'):6.1f} px, erased {m('walls_erased'):6.1f} px "
                  f"(of {m('maze_wall_px'):.0f}) | path: solved {m('solved'):.2f}, drawn {m('path_drawn'):.3f}, extra cells {m('extra'):.2f}, "
                  f"blue px {m('blue_px'):.0f}", flush=True)
    torch.save(results, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
