import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _load_checkpoint
from eval.mnist_eval import evaluate_grids
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
    eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)

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
    solutions = batch["solution"].to(device)
    given_mask = batch.get("solution_mask")
    given_mask = given_mask.to(device) if given_mask is not None else None

    num_inference_steps = model.sampling_pipeline.num_inference_steps
    cfg_scale = model.sampling_pipeline.cfg_scale
    print(f"n_sup(trained)={model.n_sup}  cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  eval_cfg.use_halt_head={model.eval_cfg.use_halt_head}")

    torch.manual_seed(123)
    x_init = torch.randn_like(conditions.images)
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    z_H_c = z_L_c = None
    z_H_u = z_L_u = None
    for t in model.scheduler.timesteps:
        tb = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=tb)
        pred_c, z_H_c, z_L_c = model.forward_with_carry(step_sample, z_H_c, z_L_c, n_sup=16, use_halt_head=False)
        noise_pred = pred_c.pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, z_H_u, z_L_u = model.forward_with_carry(null_sample, z_H_u, z_L_u, n_sup=16, use_halt_head=False)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)
        x = model.scheduler.step(noise_pred, t, x).prev_sample

    res = evaluate_grids(x.clamp(0.0, 1.0), solutions, classifier, cell_size, given_masks=given_mask)
    print(f"[explicit forward_with_carry, reset every step] puzzle_acc={res['puzzle_acc']:.4f}  cell_acc={res['cell_acc']:.4f}")

    # now via bare model(sample) call for comparison
    torch.manual_seed(123)
    x_init2 = torch.randn_like(conditions.images)
    x2 = x_init2.clone()
    for t in model.scheduler.timesteps:
        tb = t.expand(x2.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x2, timesteps=tb)
        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u = model(null_sample).pred
            noise_pred = pred_u + cfg_scale * (noise_pred - pred_u)
        x2 = model.scheduler.step(noise_pred, t, x2).prev_sample
    res2 = evaluate_grids(x2.clamp(0.0, 1.0), solutions, classifier, cell_size, given_masks=given_mask)
    print(f"[bare model(sample) call]                  puzzle_acc={res2['puzzle_acc']:.4f}  cell_acc={res2['cell_acc']:.4f}")
    print(f"identical outputs: {torch.equal(x, x2)}")


if __name__ == "__main__":
    main()
