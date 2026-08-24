"""
experiments/state_shuffle_probe.py — Ablation C: state-shuffling probe on a
trained TRMDiffusionBackbone checkpoint.

Test-time-only intervention on an already-trained, frozen model: at every
n_sup reasoning iteration during sampling (including the last, right before
the final readout), permute the carried state (z_H, z_L) across the batch
dimension — same permutation applied to both, resampled fresh each
iteration — so sample i's next reasoning step sees sample j's carried state
instead of its own. tokens (the per-sample image/condition input) are never
shuffled, so the model is still conditioned on the right puzzle every step;
only whose prior reasoning state it's handed changes. Identical FLOPs,
identical parameters, same aggregate "warmed-up" state statistics.

If accuracy collapses relative to the unshuffled baseline, the carried state
is doing genuine sample-specific refinement, not acting as generic effective
depth / regularization. Both configs run against the exact same cached
validation batches and exact same initial noise (seeded once up front) — a
paired comparison, so any difference is attributable to shuffling alone.

Usage:
    python experiments/state_shuffle_probe.py \\
      experiment=mnist_trm_diffusion_backbone \\
      checkpoint=runs/mnist_trm_diffusion_backbone/checkpoint_final.pt \\
      +probe.num_samples=256
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from eval.mnist_eval import evaluate_grids
from factory import build_datasets, build_model
from hydra.utils import instantiate
from models.utility_models import strip_compiled_prefix

logger = get_logger(__name__, log_level="INFO")


def _load_checkpoint(model, ckpt_path: str, use_ema: bool = True, device="cpu") -> int | None:
    """Duplicated from eval.py — see ablate_trm_loop_budget.py's identical
    helper for why (the eval/ package in this same directory shadows the
    top-level eval.py module, so `from eval import ...` resolves to the
    package, not the script)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        step = ckpt.get("step", None)
        sd = strip_compiled_prefix(ckpt["model_state"])
        model.load_state_dict(sd, strict=False)

        if use_ema and ckpt.get("ema_state") is not None:
            ema_state = ckpt["ema_state"]
            if isinstance(ema_state, dict) and ema_state:
                model.load_state_dict(strip_compiled_prefix(ema_state), strict=False)
                logger.info(f"Loaded EMA weights on top of model_state (step={step})")
                return step
            logger.warning("EMA state is empty — using raw model_state")
        logger.info(f"Loaded model_state (step={step}, use_ema={use_ema})")
        return step

    model.load_state_dict(strip_compiled_prefix(ckpt), strict=False)
    logger.info("Loaded raw state_dict")
    return None


@torch.no_grad()
def _sample_with_shuffle(
    model, conditions, x_init: torch.Tensor, num_inference_steps: int, cfg_scale: float, shuffle_state: bool
) -> torch.Tensor:
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        pred_cond = model(step_sample, shuffle_state=shuffle_state).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_uncond = model(null_sample, shuffle_state=shuffle_state).pred
            noise_pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = pred_cond

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    return x


def _build_cached_batches(model, dataloader, device, num_samples: int, seed: int) -> list[dict]:
    """Cache a fixed set of (conditions, solutions, given_masks, x_init) once,
    reused identically for both the baseline and shuffled configs — a
    paired comparison."""
    torch.manual_seed(seed)
    cached = []
    n_done = 0
    for batch in dataloader:
        if n_done >= num_samples:
            break
        conditions = model._batch_to_sample(batch, device)
        solutions = batch["solution"]
        given_masks = batch.get("solution_mask")
        bsz = solutions.shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)
        cached.append({
            "conditions": conditions,
            "solutions": solutions,
            "given_masks": given_masks,
            "x_init": x_init,
        })
        n_done += bsz
    logger.info(f"Cached {n_done} samples across {len(cached)} batches.")
    return cached


def _run_config(
    model, classifier, cell_size: int, cached_batches: list[dict],
    num_inference_steps: int, cfg_scale: float, shuffle_state: bool,
) -> dict:
    all_cell, all_puzzle, all_constraint, all_given_consistent = [], [], [], []
    t0 = time.time()

    for cb in cached_batches:
        x = _sample_with_shuffle(
            model, cb["conditions"], cb["x_init"], num_inference_steps, cfg_scale, shuffle_state
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
        "wall_time_sec": time.time() - t0,
    }
    if all_given_consistent:
        result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent))
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/state_shuffle_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+probe.num_samples=256]"
        )

    probe = cfg.get("probe", {})
    num_samples: int = probe.get("num_samples", 256)
    seed: int = probe.get("seed", 0)
    out_path: str = probe.get("out", str(Path(checkpoint).parent / "state_shuffle_probe.json"))

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
        eval_ds,
        batch_size=cfg.eval.get("batch_size", cfg.train.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=eval_collate_fn,
    )

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline_cfg = cfg.get("sampling")
    num_inference_steps = int(pipeline_cfg.num_inference_steps) if pipeline_cfg is not None else 20
    cfg_scale = 1.0
    if pipeline_cfg is not None and pipeline_cfg.get("predictor", {}).get("scale") is not None:
        cfg_scale = float(pipeline_cfg.predictor.scale)

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    logger.info(f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  n_sup={model.n_sup}")

    cached_batches = _build_cached_batches(model, eval_dl, device, num_samples, seed)

    results: dict[str, dict] = {}
    for shuffle_state, name in [(False, "baseline"), (True, "shuffled")]:
        logger.info(f"Running {name} (shuffle_state={shuffle_state}) ...")
        results[name] = _run_config(
            model, classifier, cell_size, cached_batches, num_inference_steps, cfg_scale, shuffle_state
        )
        logger.info(f"  → {results[name]}")

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print(f"{'config':<12}{'cell_acc':>10}{'puzzle_acc':>12}{'constr_acc':>12}{'given_cons':>12}")
        print("=" * 80)
        for name, r in results.items():
            print(
                f"{name:<12}{r['cell_acc']:>10.4f}{r['puzzle_acc']:>12.4f}{r['constraint_puzzle_acc']:>12.4f}"
                f"{r.get('given_consistent_puzzle_acc', float('nan')):>12.4f}"
            )
        print("=" * 80)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
