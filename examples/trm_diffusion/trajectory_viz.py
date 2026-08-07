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

    # Match train_trm.py's environment more closely (it compiles self.unet;
    # this script doesn't by default):
    python trajectory_viz.py ... +compile=true

    # Also re-check teacher-forced loss (train_dl and eval_dl, eval() mode)
    # against what was logged live during training — isolates "the reloaded
    # model itself denoises worse" from "something about sampling/iteration
    # specifically is broken":
    python trajectory_viz.py ... +loss_check_batches=20
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


@torch.no_grad()
def _quick_loss(model, dataloader, device, max_batches: int) -> tuple[float, int]:
    """Teacher-forced diff_loss over up to max_batches, model in eval() mode
    (matching how train_trm.py's eval_step computes val/diff_loss) — but
    without running eval_callbacks, so this stays fast."""
    total, n = 0.0, 0
    dl_iter = iter(dataloader)
    for _ in range(max_batches):
        try:
            batch = next(dl_iter)
        except StopIteration:
            break
        sample = model._prepare_training_sample(batch, device)
        result = model(sample)
        _, components = model.loss_fn(result.pred, result.logits, sample)
        total += components.get("diff_loss", 0.0)
        n += 1
    return (total / n if n else float("nan")), n


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

    train_ds, eval_ds = build_datasets(cfg)
    collate_fn = getattr(type(eval_ds), "collate_fn", None)
    dl = DataLoader(eval_ds, batch_size=num_samples, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    step = _load_checkpoint(model, str(checkpoint), use_ema=use_ema, device="cpu")
    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    if cfg.get("compile", False):
        unwrapped.compile_submodules()
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

    # Cross-check against the pipeline's own (already-tested) sampling call,
    # same conditions/seed — if this disagrees with the manual loop's t=0
    # snapshot above, the manual loop (used only for the trajectory columns)
    # has its own bug; if it agrees, the manual loop is trustworthy and any
    # discrepancy vs. eval.py's callback lies elsewhere (e.g. compile, or
    # genuine sample-to-sample variance from a different noise draw).
    generator2 = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        official_x0 = pipeline.sample_one_batch(unwrapped, conditions, device, generator=generator2)
    official_img = unwrapped.decode_for_eval(official_x0).cpu()

    gt = unwrapped.images_to_log(batch["images"][:B]).cpu()
    cond = batch.get("spatial_conditions") if hasattr(batch, "get") else None
    cond_imgs = unwrapped.images_to_log(cond[:B]).cpu() if cond is not None else None

    ts_sorted = sorted(snapshots.keys(), reverse=True)
    col_labels = (
        (["condition"] if cond_imgs is not None else [])
        + ["GT"]
        + [f"t={t}" for t in ts_sorted]
        + ["pipeline\nsample"]
    )
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
        axes[row, col].imshow(_to_img(official_img[row]), cmap="gray", vmin=0, vmax=1)
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

    loss_check_batches = int(cfg.get("loss_check_batches", 0))
    if loss_check_batches > 0:
        loss_bs = cfg.eval.get("batch_size", cfg.train.batch_size)
        train_collate_fn = getattr(type(train_ds), "collate_fn", None)
        train_loss_dl = DataLoader(train_ds, batch_size=loss_bs, shuffle=True, collate_fn=train_collate_fn)
        val_loss_dl = DataLoader(eval_ds, batch_size=loss_bs, shuffle=False, collate_fn=collate_fn)
        train_loss, n_train = _quick_loss(unwrapped, train_loss_dl, device, loss_check_batches)
        val_loss, n_val = _quick_loss(unwrapped, val_loss_dl, device, loss_check_batches)
        print(f"train_dl diff_loss (eval() mode, {n_train} batches of {loss_bs}): {train_loss:.6f}")
        print(f"val_dl   diff_loss (eval() mode, {n_val} batches of {loss_bs}): {val_loss:.6f}")
        print("Compare both against the train/diff_loss and val/diff_loss curves logged during training.")


if __name__ == "__main__":
    main()
