"""One-off diagnostic: (a) classifier accuracy on ground-truth cell crops for
easy vs hard, (b) puzzle_acc vs violating-units-count cross-check on the same
PT trajectory, run separately per experiment=... via hydra."""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from eval.mnist_eval import evaluate_grids, _check_sudoku_constraints
from factory import build_datasets, build_model
from perturbation_recovery_probe import _decode_cellwise


@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from accelerate import Accelerator
    Accelerator(mixed_precision=cfg.precision.mixed_precision)

    checkpoint = cfg.get("checkpoint")
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    sudoku_cb = next(c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None)
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    batch = next(iter(eval_dl))
    conditions = model._batch_to_sample(batch, device)
    images = conditions.images
    solutions = batch["solution"].to(device)

    print(f"images shape={tuple(images.shape)} dtype={images.dtype} min={images.min().item():.4f} max={images.max().item():.4f} mean={images.mean().item():.4f}")

    # (a) classifier accuracy on GROUND TRUTH crops (no diffusion at all)
    gt_preds = _decode_cellwise(images.clamp(0.0, 1.0), classifier, cell_size)
    gt_correct = (gt_preds == solutions)
    print(f"GT classifier cell_acc = {gt_correct.float().mean().item():.4f}  (should be ~1.0)")
    gt_valid = _check_sudoku_constraints(gt_preds)
    print(f"GT classifier valid-grid rate = {gt_valid.float().mean().item():.4f}  (should be ~1.0)")

    # (b) t=0-noised + 1-step painter denoise (matches painter_only t_start=0), check cell_acc directly
    num_inference_steps = model.sampling_pipeline.num_inference_steps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    full_ts = model.scheduler.timesteps
    t0 = full_ts[full_ts <= 0]
    torch.manual_seed(0)
    noise = torch.randn_like(images)
    t_batch = torch.full((images.shape[0],), 0, device=device, dtype=torch.long)
    x_t0 = model.scheduler.add_noise(images, noise, t_batch)
    x = x_t0
    for t in t0:
        tb = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=tb)
        noise_pred = model.painter(step_sample, steering=None).pred
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    preds_t0 = _decode_cellwise(x.clamp(0.0, 1.0), classifier, cell_size)
    print(f"painter-only t_start=0 cell_acc = {(preds_t0 == solutions).float().mean().item():.4f}")

    # (c) PT full trajectory from pure noise, check puzzle_acc + violations on final step
    x_init = torch.randn_like(conditions.images)
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x2 = x_init.clone()
    cfg_scale = model.sampling_pipeline.cfg_scale
    for t in model.scheduler.timesteps:
        tb = t.expand(x2.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x2, timesteps=tb)
        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u = model(null_sample).pred
            noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        x2 = model.scheduler.step(noise_pred, t, x2).prev_sample
    final_preds = _decode_cellwise(x2.clamp(0.0, 1.0), classifier, cell_size)
    given_mask = batch.get("solution_mask")
    given_mask = given_mask.to(device) if given_mask is not None else None
    res = evaluate_grids(x2.clamp(0.0, 1.0), solutions, classifier, cell_size, given_masks=given_mask)
    n_bad_units = torch.zeros(final_preds.shape[0], dtype=torch.int64, device=device)
    grid = final_preds.reshape(-1, 9, 9)
    expected = torch.arange(9, device=device)
    for i in range(9):
        row_bad = ~grid[:, i, :].sort(dim=1).values.eq(expected).all(dim=1)
        col_bad = ~grid[:, :, i].sort(dim=1).values.eq(expected).all(dim=1)
        br, bc = (i // 3) * 3, (i % 3) * 3
        box = grid[:, br:br+3, bc:bc+3].reshape(-1, 9)
        box_bad = ~box.sort(dim=1).values.eq(expected).all(dim=1)
        n_bad_units += row_bad.int() + col_bad.int() + box_bad.int()
    print(f"PT full-traj puzzle_acc={res['puzzle_acc']:.4f}  cell_acc={res['cell_acc']:.4f}  mean_violating_units={n_bad_units.float().mean().item():.3f}")


if __name__ == "__main__":
    main()
