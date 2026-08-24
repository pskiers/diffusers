"""
experiments/diag_halt_noshrink.py — isolates whether the halt axis's
accuracy cliff comes from the per-sample halting DECISIONS themselves, or
from the dynamic re-batching mechanism that PHYSICALLY SHRINKS the active
batch as samples halt (forward_with_carry's use_halt_head=True path).

Reconstructs the exact same per-sample halting decisions (same
predict_halt_value trace, same threshold) but WITHOUT ever changing tensor
shapes: runs the full nominal n_sup trajectory for every sample (via
forward_with_carry's use_halt_head=False path + halt_preds_out/logits_out,
which record every iteration's values for the FULL, unshrunk batch, exactly
matching training's own batch-shape-stable computation), then in Python
picks out each sample's own logits from whichever iteration it would have
halted at. If this "same decisions, no shrinking" version doesn't show the
accuracy cliff that the real dynamic-shrinking mechanism does, that proves
the cliff comes from GPU floating-point non-associativity across different
batch shapes cascading through the recursive reasoning loop, not from the
halting decisions being bad.

Usage:
    python experiments/diag_halt_noshrink.py experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter.pt \\
      condition_encoder=x0_hint_v1 condition_encoder.threshold=80 \\
      +condition_encoder.enabled=false condition_encoder.inner.with_timestep_emb=false \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      thinker.with_halt_head=true +checkpoint=runs/sudoku_painter_thinker_with_halt_head.pt +use_ema=false \\
      +diag.thresholds=[-0.001,0.0002,0.001] +diag.num_samples=2048
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _build_cached_batches, _load_checkpoint
from eval.mnist_eval import evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate

logger = get_logger(__name__, log_level="INFO")


def _apply_nominal_halt(model, sample, halt_preds_list, logits_list, halt_threshold):
    """Same per-sample halting decision as the real dynamic mechanism
    (first iteration where predict_halt_value <= threshold, else the last
    iteration), but picked out of a FULL, never-shrunk trajectory -- every
    sample's every iteration ran in the same, full-batch-size tensor shape
    the whole time, exactly like training and like the use_halt_head=False
    path. Returns (noise_pred, calls_per_sample)."""
    n_sup = len(halt_preds_list)
    preds_stack = torch.stack(halt_preds_list, dim=0)  # (n_sup, B)
    halted = preds_stack <= halt_threshold  # (n_sup, B) bool
    any_halt = halted.any(dim=0)  # (B,)
    first_halt_idx = halted.float().argmax(dim=0)  # first True index (argmax returns first max on ties)
    halt_step = torch.where(any_halt, first_halt_idx, torch.full_like(first_halt_idx, n_sup - 1))  # (B,)

    logits_stack = torch.stack(logits_list, dim=0)  # (n_sup, B, seq_len, vocab)
    idx = halt_step.view(1, -1, 1, 1).expand(1, *logits_stack.shape[1:])
    combined_logits = logits_stack.gather(0, idx).squeeze(0)  # (B, seq_len, vocab)

    noise_pred = model.run_painter(sample, combined_logits)
    calls = halt_step.float() + 1.0  # 1-indexed steps used per sample
    return noise_pred, calls


@torch.no_grad()
def _run_noshrink_halt_trajectory(model, conditions, x_init, num_inference_steps, cfg_scale, halt_threshold, n_sup, total_calls):
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    step_calls = 0.0

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        halt_preds_c, logits_out_c = [], []
        model.forward_with_carry(
            step_sample, None, None, n_sup=n_sup, use_halt_head=False,
            halt_preds_out=halt_preds_c, logits_out=logits_out_c,
        )
        noise_pred, calls_c = _apply_nominal_halt(model, step_sample, halt_preds_c, logits_out_c, halt_threshold)
        step_calls += calls_c.mean().item()

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            halt_preds_u, logits_out_u = [], []
            model.forward_with_carry(
                null_sample, None, None, n_sup=n_sup, use_halt_head=False,
                halt_preds_out=halt_preds_u, logits_out=logits_out_u,
            )
            noise_pred_u, calls_u = _apply_nominal_halt(model, null_sample, halt_preds_u, logits_out_u, halt_threshold)
            noise_pred = noise_pred_u + cfg_scale * (noise_pred - noise_pred_u)
            step_calls += calls_u.mean().item()

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    total_calls.append(step_calls)
    return x


def _run_noshrink_config(model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale, halt_threshold, n_sup):
    import numpy as np
    all_cell, all_puzzle, all_constraint, all_given_consistent = [], [], [], []
    total_calls: list[float] = []

    for cb in cached_batches:
        x = _run_noshrink_halt_trajectory(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale, halt_threshold, n_sup, total_calls,
        )
        generated = model.decode_for_eval(x)
        acc = evaluate_grids(generated, cb["solutions"], classifier, cell_size, given_masks=cb["given_masks"])
        all_cell.append(acc["cell_acc"])
        all_puzzle.append(acc["puzzle_acc"])
        all_constraint.append(acc.get("constraint_puzzle_acc", 0.0))
        if acc.get("given_consistent_puzzle_acc") is not None:
            all_given_consistent.append(acc["given_consistent_puzzle_acc"])

    result = {
        "cell_acc": float(np.mean(all_cell)),
        "puzzle_acc": float(np.mean(all_puzzle)),
        "constraint_puzzle_acc": float(np.mean(all_constraint)),
        "total_sup_calls": float(np.sum(total_calls)),
    }
    if all_given_consistent:
        result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent))
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit("ERROR: No checkpoint specified.")

    dg = cfg.get("diag", {})
    num_samples: int = dg.get("num_samples", 2048)
    seed: int = dg.get("seed", 0)
    thresholds: list[float] = list(dg.get("thresholds", [-0.001, 0.0002, 0.001]))
    out_path: str = dg.get("out", str(Path(checkpoint).parent / "diag_halt_noshrink.json"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    _, eval_ds = build_datasets(cfg)
    eval_collate_fn = getattr(type(eval_ds), "collate_fn", None)
    eval_dl = DataLoader(
        eval_ds, batch_size=cfg.eval.get("batch_size", cfg.train.batch_size), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    if not getattr(model.thinker, "with_halt_head", False):
        raise SystemExit("Model was built without a halt head — add thinker.with_halt_head=true.")
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    cfg_scale: float = pipeline.cfg_scale
    num_inference_steps: int = pipeline.num_inference_steps
    n_sup = model.n_sup

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier found.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  n_sup={n_sup}  thresholds={thresholds}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results = {}
    for th in thresholds:
        key = f"noshrink/threshold={th}"
        logger.info(f"Running {key} ...")
        r = _run_noshrink_config(model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale, th, n_sup)
        results[key] = r
        logger.info(f"  → {r}")

    if accelerator.is_main_process:
        print("\n" + "=" * 100)
        print(f"{'config':<30}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}{'given_cons':>12}{'sup_calls':>10}")
        print("=" * 100)
        for key, r in results.items():
            gc = r.get("given_consistent_puzzle_acc")
            print(f"{key:<30}{r['cell_acc']:>10.4f}{r['puzzle_acc']:>12.4f}{r['constraint_puzzle_acc']:>12.4f}{(gc if gc is not None else 0.0):>12.4f}{r['total_sup_calls']:>10.1f}")
        print("=" * 100)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
