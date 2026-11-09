"""Per-timestep diffusion loss: frozen painter alone vs painter steered by the thinker.

The condition fully determines the solution, so a thinker that actually uses it
should pull the x0-MSE far below the unconditional painter's, most of all at high
noise where the painter alone can only guess. This measures exactly that, on the
SAME validation images, noise and timesteps for both, in eval mode (no CFG logit
dropout -- train_trm's val/diff_loss runs with it on).

Usage (same overrides as the thinker run):
    python experiments/thinker_vs_painter_loss.py experiment=amaze_latent_thinker_xattn \
        painter.checkpoint=<painter.pt> painter.latent_shift=[...] painter.latent_scale=[...] \
        vae.pretrained_model_name_or_path=<vae dir> data.amaze_root=data/amaze/maze \
        +checkpoint=<thinker.pt> [+n_batches=4] [+timesteps=[5,15,...]] [+out_json=...]
    +checkpoint=none measures the painter only.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eval.checkpoint_utils import load_checkpoint
from factory import build_datasets, build_model


def _mse(pred, target):
    return (pred.float() - target.float()).pow(2).flatten(1).mean(1)   # per sample


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = str(cfg.get("checkpoint", "none"))
    n_batches = int(cfg.get("n_batches", 4))
    T = int(cfg.diffusion.num_train_timesteps)
    ts = list(cfg.get("timesteps", list(range(5, T, 10))))

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    step = None
    if ckpt.lower() != "none":
        step = load_checkpoint(model, ckpt, use_ema=cfg.get("use_ema", True), device="cpu")
    model.to(device).eval()

    _, val_ds = build_datasets(cfg)
    dl = DataLoader(val_ds, batch_size=int(cfg.eval.batch_size), shuffle=False, num_workers=4,
                    collate_fn=getattr(type(val_ds), "collate_fn", None))
    painter = model.painter if hasattr(model, "painter") else model

    sums = {t: {"painter": 0.0, "thinker": 0.0} for t in ts}
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= n_batches:
                break
            base = painter._prepare_training_sample(batch, device)
            z0 = base.target                     # prediction_type=sample -> clean latent
            noise = torch.randn_like(z0)
            B = z0.shape[0]
            for t in ts:
                tt = torch.full((B,), t, device=device, dtype=torch.long)
                s = dataclasses.replace(base, x_noisy=scheduler.add_noise(z0, noise, tt), timesteps=tt, target=z0)
                sums[t]["painter"] += _mse(painter(s).pred, z0).sum().item()
                if painter is not model:
                    sums[t]["thinker"] += _mse(model(s).pred, z0).sum().item()
            n += B
            print(f"batch {bi + 1}/{n_batches} done", flush=True)

    rows = []
    print(f"\ncheckpoint={ckpt} step={step} images={n}")
    print(f"{'t':>4} {'alpha_bar':>9} {'painter':>9} {'thinker':>9} {'thinker/painter':>15}")
    for t in ts:
        p, q = sums[t]["painter"] / n, sums[t]["thinker"] / n
        ab = float(scheduler.alphas_cumprod[t])
        ratio = q / p if p > 0 and painter is not model else float("nan")
        rows.append({"t": t, "alpha_bar": ab, "painter": p, "thinker": q if painter is not model else None,
                     "ratio": ratio})
        print(f"{t:>4} {ab:>9.4f} {p:>9.4f} {q:>9.4f} {ratio:>15.3f}")
    mp = sum(r["painter"] for r in rows) / len(rows)
    mq = sum((r["thinker"] or 0.0) for r in rows) / len(rows)
    print(f"mean {'':>9} {mp:>9.4f} {mq:>9.4f} {mq / mp if mp > 0 else float('nan'):>15.3f}")

    out = cfg.get("out_json", None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"checkpoint": ckpt, "step": step, "images": n, "rows": rows,
                       "mean_painter": mp, "mean_thinker": mq}, f, indent=2)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
