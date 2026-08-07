"""
trajectory_viz.py — visualize a model's x0 prediction at intermediate
denoising timesteps, for a loaded checkpoint.

eval.py's callbacks only ever see the *final* generated image. This
diagnostic instead saves the scheduler's pred_original_sample (the model's
current x0 estimate) at several chosen timesteps along one DDIM/DDPM
sampling trajectory — the same "x0 predictions across denoising steps"
view the paper's own Figure 3 uses. Distinguishes two different failure
modes for a model that produces good val loss but bad final samples:
  - already garbage at the first step (t close to num_train_timesteps,
    denoising from pure noise) -> something wrong independent of iteration
  - looks fine early, degrades over the trajectory -> error compounding
    over repeated self-referential calls (exposure bias), which teacher-
    forced val loss (always given a *real* noisy image, never the model's
    own prior output) can't expose.

Usage:
    python trajectory_viz.py experiment=steiner_unet_concat_painter backbone=steiner_unet_paper \
        checkpoint=runs/steiner_paper_unet/checkpoint_final.pt \
        +use_ema=false +num_samples=4

    # Custom timesteps to snapshot (default: 99,90,70,50,30,10,0):
    python trajectory_viz.py ... +log_timesteps=99,80,60,40,20,0

    # Output path (default: <checkpoint_dir>/trajectory.png):
    python trajectory_viz.py ... +output=results/traj.png
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from factory import build_datasets, build_model
from models.utility_models import load_checkpoint as _load_checkpoint


def _to_img(t: torch.Tensor) -> torch.Tensor:
    """(C, H, W) -> (H, W) or (H, W, 3) numpy-ready tensor in [0, 1], for imshow.

    decode_for_eval/images_to_log return each painter's own native range
    (e.g. [-1, 1] for steiner's pixel_tanh convention, [0, 1] for others) —
    min-max normalizing per-image here is display-only and avoids having to
    know which convention applies.
    """
    t = t.float()
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo).clamp(min=1e-8)
    return t[0] if t.shape[0] == 1 else t.permute(1, 2, 0)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        print("ERROR: checkpoint=<path> required", file=sys.stderr)
        sys.exit(1)

    use_ema: bool = cfg.get("use_ema", True)
    num_samples: int = int(cfg.get("num_samples", 4))
    seed: int = int(cfg.get("seed", 0))
    log_ts_raw = cfg.get("log_timesteps", None)
    target_ts = (
        [int(t) for t in str(log_ts_raw).split(",")] if log_ts_raw is not None else [99, 90, 70, 50, 30, 10, 0]
    )
    output = cfg.get("output", None) or str(Path(checkpoint).parent / "trajectory.png")

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    _, eval_ds = build_datasets(cfg)
    collate_fn = getattr(type(eval_ds), "collate_fn", None)
    dl = DataLoader(eval_ds, batch_size=num_samples, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    step = _load_checkpoint(model, str(checkpoint), use_ema=use_ema, device="cpu")
    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    conditions = unwrapped._batch_to_sample(batch, device)
    pipeline = unwrapped.sampling_pipeline
    B = num_samples
    shape = (B, *unwrapped.noise_shape)
    T = unwrapped.scheduler.config.num_train_timesteps

    unwrapped.scheduler.set_timesteps(pipeline.num_inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, generator=generator)

    snapshots: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for t in unwrapped.scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
            noise_pred = pipeline.predictor.predict(unwrapped, step_sample, int(t.item()), T)
            step_out = unwrapped.scheduler.step(noise_pred, t, x)
            x = step_out.prev_sample
            t_int = int(t.item())
            if t_int in target_ts:
                x0_est = getattr(step_out, "pred_original_sample", None)
                snapshots[t_int] = unwrapped.decode_for_eval(x0_est if x0_est is not None else x).cpu()
    if 0 not in snapshots:
        snapshots[0] = unwrapped.decode_for_eval(x).cpu()

    gt = unwrapped.images_to_log(batch["images"][:B]).cpu()
    cond = batch.get("spatial_conditions") if hasattr(batch, "get") else None
    cond_imgs = unwrapped.images_to_log(cond[:B]).cpu() if cond is not None else None

    ts_sorted = sorted(snapshots.keys(), reverse=True)
    col_labels = (["condition"] if cond_imgs is not None else []) + ["GT"] + [f"t={t}" for t in ts_sorted]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(B, n_cols, figsize=(n_cols * 2.2, B * 2.2), squeeze=False)
    for row in range(B):
        col = 0
        if cond_imgs is not None:
            axes[row, col].imshow(_to_img(cond_imgs[row]), cmap="gray", vmin=0, vmax=1)
            col += 1
        axes[row, col].imshow(_to_img(gt[row]), cmap="gray", vmin=0, vmax=1)
        col += 1
        for t in ts_sorted:
            axes[row, col].imshow(_to_img(snapshots[t][row]), cmap="gray", vmin=0, vmax=1)
            col += 1
        for c in range(n_cols):
            axes[row, c].axis("off")
            if row == 0:
                axes[row, c].set_title(col_labels[c], fontsize=10)

    fig.suptitle(f"{checkpoint}  (step={step}, use_ema={use_ema})", fontsize=9)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved trajectory grid -> {output}")


if __name__ == "__main__":
    main()
