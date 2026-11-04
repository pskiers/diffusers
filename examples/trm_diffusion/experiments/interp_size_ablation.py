"""TEMPORARY diagnostic (Claude, amaze interpretability): why does PaTh fail on maze size 10 (held out from training)
while solving the trained sizes around it (8, 11)?

All runs: square mazes, full budget, standard generation from pure noise (no drawn prefix), first N_BOARDS boards.
(1) Size map (H1, the model learned each trained size separately): native square sizes 3, 5, 7, 8, 9, 10, 11 (3 and 10 untrained).
(2) Rescaling (H2, what matters is the pixel size of a cell): the maze image, the solution image and the cell map are shrunk
    together so a cell has the pixel size of another size at 144 px, and padded back to the full image (black margin, as in the
    training images of hexagons/circles; one white-margin variant). Scoring uses the transformed cell map.
      10x10 at the cell size of 11, 13, 16 (13.1, 11.1, 9.0 px; native 14.4 px)
      8x8   at the cell size of 10 (14.4 px, untrained) and of 11 (13.1 px)       -- 8x8 natively 18 px
      11x11 at the cell size of 13 (11.1 px)                                       -- padding control, trained cell size
(3) Thinker vs painter (H3) and convergence (H4): at the denoising step t=70 of each run, the final z_H of the conditional pass
    and the relative state change at every block call; held-out linear probes (path vs off per token) on z_H, trained on the
    same set (even boards) and transferred from native size 11 and 8.
Solved = every solution cell except start/goal drawn and no off-path cell drawn.
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
import maze_corruption_lib as mcl
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG = 12, 144
N_BOARDS = 48
CELL_PX = {n: IMG / n for n in (3, 5, 7, 8, 9, 10, 11, 13, 16)}
# (name, size, target cell size in px or None = native, margin colour)
SETS = ([(f"n{n} native", n, None, "black") for n in (3, 5, 7, 8, 9, 10, 11)]
        + [("n10 @ n11 cells", 10, CELL_PX[11], "black"), ("n10 @ n11 cells, white margin", 10, CELL_PX[11], "white"),
           ("n10 @ n13 cells", 10, CELL_PX[13], "black"), ("n10 @ n16 cells", 10, CELL_PX[16], "black"),
           ("n8 @ n10 cells", 8, CELL_PX[10], "black"), ("n8 @ n11 cells", 8, CELL_PX[11], "black"),
           ("n11 @ n13 cells", 11, CELL_PX[13], "black")])


def rescale(s, ds, n, target_px, margin):
    """Shrink the sample's images so a cell is target_px wide at 144 px, centred on a canvas of the original size."""
    if target_px is None:
        return s
    frac = target_px * n / IMG
    md = dict(s.metadata)
    bg = (0, 0, 0) if margin == "black" else (255, 255, 255)
    for key in ("m_original_img", "sol_img", "original_img", "mask_img", "cell_map"):
        img = md.get(key)
        if img is None:
            continue
        W, H = img.size
        nw, nh = max(1, round(W * frac)), max(1, round(H * frac))
        small = img.resize((nw, nh), Image.NEAREST if key == "cell_map" else Image.BICUBIC)
        canvas = Image.new("RGB", (W, H), (0, 0, 0) if key == "cell_map" else bg)
        canvas.paste(small, ((W - nw) // 2, (H - nh) // 2))
        md[key] = canvas
    return dataclasses.replace(s, images=ds._image_to_tensor(md["sol_img"]), spatial_conditions=ds._image_to_tensor(md["m_original_img"]),
                               metadata=md)


def fit_probe(X, y, train):
    m, sd = X[train].mean(0), X[train].std(0) + 1e-6
    w = torch.zeros(X.shape[1] + 1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)
    Xa = (X[train] - m) / sd
    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w[:-1] + w[-1], y[train]) + 1e-3 * w[:-1].pow(2).sum()
        l.backward(); return l
    opt.step(cl)
    w = w.detach()
    return lambda Z: ((Z - m) / sd) @ w[:-1] + w[-1]


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
    inner = model.thinker.inner
    per = inner.config.L_cycles + 1
    st = {"on": False, "c": 0, "last": {}, "rel": []}

    def hook(module, inputs, output):
        if not st["on"]:
            return None
        c = st["c"]; st["c"] += 1
        k = "z_H" if c % per == per - 1 else "z_L"
        o = output.float()
        if k in st["last"]:
            st["rel"].append(((o - st["last"][k]).flatten(1).norm(dim=1) / o.flatten(1).norm(dim=1)).cpu())
        else:
            st["rel"].append(torch.full((o.shape[0],), float("nan")))
        st["last"][k] = o
        return None
    inner.L_level.register_forward_hook(hook)

    vd = cfg.data.get("val_dataset", {})
    results = {}
    for name, n, target_px, margin in SETS:
        ds = AmazeDataset(dataset_path=f"{ic.data_root}/square/n{n}_test.parquet", split="test", image_size=IMG,
                          condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                          num_channels=3, include_metadata=True)
        rows, zH, gts, rels, saved = [], [], [], [], None
        i0 = 0
        while len(rows) < N_BOARDS and i0 < len(ds):
            raw = [ds[i] for i in range(i0, min(i0 + 16, len(ds)))]
            i0 += 16
            samples, mss = [], []
            for s in raw:
                s2 = rescale(s, ds, n, target_px, margin)
                ms = mcl.build_maze_sample(s2.metadata)
                if ms is not None:
                    samples.append(s2); mss.append(ms)
            if not samples:
                continue
            B = len(samples)
            batch = AmazeDataset.collate_fn(samples)
            cond = model._batch_to_sample(batch, device)
            g = torch.Generator(device=device).manual_seed(1000 + i0)
            x = torch.randn((B, 3, IMG, IMG), generator=g, device=device)
            gen = torch.Generator(device=device).manual_seed(2000 + i0)
            with torch.no_grad():
                for t in ts:
                    s = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(B).to(device))
                    rec = int(t) == 70
                    st.update(on=rec, c=0, last={}, rel=[])
                    pred_cond, z_H, _ = model.forward_with_carry(s)
                    st["on"] = False
                    if rec:
                        zH.append(z_H.half().cpu()); rels.append(torch.stack(st["rel"], 1))           # (B, n_calls)
                    pred_uncond = model(model.null_condition_sample(s)).pred
                    pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                    x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
                out = model.decode_for_eval(x).clamp(0, 1)
            sol_u8 = (batch.images.float().clamp(0, 1) * 255).byte().cpu()
            maze_u8 = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
            for i, ms in enumerate(mss):
                pred_c = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                path = set(ms.path)
                inner_c = path - {ms.path[0], ms.path[-1]}
                rows.append({"solved": inner_c <= pred_c and not (pred_c - path), "missing": len(inner_c - pred_c),
                             "extra": len(pred_c - path), "path_drawn": len(path & pred_c) / len(path)})
                gts.append(pa.token_labels(sol_u8[i], maze_u8[i])[0] != "off")
            if saved is None:
                saved = {"sol": sol_u8[:6], "maze": maze_u8[:6], "out": (out[:6] * 255).byte().cpu()}
        zH = torch.cat(zH)[:len(gts)]; rel = torch.cat(rels)[:len(gts)]; gt = np.stack(gts)
        results[name] = {"rows": rows, "zH": zH, "gt": gt, "rel": rel, "saved": saved}
        r_ = rows
        conv = rel.nanmedian(0).values
        print(f"  {name:32s} n={len(r_):3d} | solved {np.mean([r['solved'] for r in r_]):.2f}  missing/board {np.mean([r['missing'] for r in r_]):5.2f}  "
              f"extra/board {np.mean([r['extra'] for r in r_]):5.2f}  path drawn {np.mean([r['path_drawn'] for r in r_]):.3f} | state change at call "
              f"21/63/336: {float(conv[20]):.4f} {float(conv[62]):.4f} {float(conv[-1]):.4f}", flush=True)

    # (3) is the path linearly readable from z_H (t=70)? probe on the same set (even boards -> odd boards), and transferred
    print("\nlinear probe path vs off from the final z_H at t=70 (held-out accuracy / share of boards whose path tokens are all right):")
    probes = {}
    for name, r in results.items():
        N = len(r["gt"]); X = r["zH"].float().reshape(N * G * G, -1).to(device); y = torch.from_numpy(r["gt"].reshape(-1).astype(np.float32)).to(device)
        A = torch.from_numpy(np.repeat(np.arange(N) % 2 == 0, G * G)).to(device)
        p = fit_probe(X, y, A); probes[name] = p
        dec = (p(X) > 0).reshape(N, -1).cpu().numpy()
        acc = float(((p(X) > 0).float() == y)[~A].float().mean())
        exact = np.mean([(dec[i] == r["gt"][i]).all() for i in range(1, N, 2)])
        r["probe_acc"], r["probe_exact"] = acc, exact
    for name, r in results.items():
        N = len(r["gt"]); X = r["zH"].float().reshape(N * G * G, -1).to(device); y = torch.from_numpy(r["gt"].reshape(-1).astype(np.float32)).to(device)
        B_ = torch.from_numpy(np.repeat(np.arange(N) % 2 == 1, G * G)).to(device)
        tr = {k: float(((probes[k](X) > 0).float() == y)[B_].float().mean()) for k in ("n11 native", "n8 native")}
        print(f"  {name:32s} own probe: acc {r['probe_acc']:.3f}, all tokens right on {r['probe_exact']:.2f} of boards | "
              f"probe from n11 native: {tr['n11 native']:.3f}, from n8 native: {tr['n8 native']:.3f}")
    torch.save(results, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
