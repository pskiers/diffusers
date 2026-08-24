"""
experiments/train_digit_probe.py — Train a post-hoc linear probe that
reads a per-cell digit prediction directly off a TRM's z_H state, from an
already-trained, FROZEN checkpoint (TRMDiffusionBackbone or
ThinkerFrozenPainterBase — anything exposing forward_with_carry). Only the
probe's own parameters are ever updated; the checkpoint is loaded and
never written to (same spirit as experiments/train_halt_head.py).

Motivation: experiments/candidate_trajectory_probe.py (D2) originally
decoded "what digit does the TRM want to paint" by rendering each
reasoning iteration's output through run_painter and reclassifying the
pixels with the MNIST CNN. At early denoising steps that reading is
dominated by ordinary diffusion x0-blur (ANY diffusion model's own x0
estimate is inherently blurry/averaged at high noise, regardless of
reasoning quality), which swamps whatever the TRM's internal state has
actually figured out. A probe on z_H itself sidesteps that rendering
bottleneck.

Label choice — the part worth being careful about:
  NOT the puzzle's ground-truth solution: that would assume the TRM always
  reasons correctly, which it doesn't (final puzzle accuracy is often
  <10%) — training against it would conflate "z_H doesn't yet know the
  answer" (an interesting, real finding) with "the model was never going
  to get this right anyway" (irrelevant to what the probe should measure).

  NOT the CNN's reading of THIS iteration's own (possibly still noisy)
  partial render: that just re-imports the exact rendering-bottleneck
  confound this probe exists to avoid.

  Instead: (1) run a REAL, FULL sampling trajectory — actual CFG-guided
  generation, start to finish, the model's own real num_inference_steps /
  n_sup / CFG scale — from random noise; (2) classify ONLY the one
  genuinely clean final image at the end (the same CNN used for every
  other accuracy number in this project); (3) that single per-cell digit
  becomes the training label for EVERY z_H captured anywhere in that same
  trajectory — every reasoning iteration of every denoising step, via
  forward_with_carry's zH_out hook (conditional branch only — the CFG null
  branch reasons about a zeroed condition and isn't meaningful here).

This measures "does z_H at this point already predict what the model is
about to commit to" — not "is the TRM objectively correct." A wrong early
guess that gets corrected later should score LOW against the eventual
label before the correction and HIGH after — that's the repair signal
experiment D3 wants, not noise to fix.

Two safeguards against the probe learning a shortcut solver rather than
genuinely reading TRM state (see models.digit_probe.DigitProbe's
docstring and the "control task" critique of probing classifiers):
  - The probe is kept strictly linear.
  - Every eval computes a control-task FLOOR: probe accuracy on z_H taken
    from the model's INITIAL state (get_initial_states(), before a single
    reasoning iteration has run) — this is the SAME constant tensor for
    every puzzle by construction, so it carries zero puzzle-specific
    information; if the probe scores meaningfully above the majority-class
    baseline there, something is leaking and the whole curve should be
    read relative to this floor, not from zero.
  - Eval is run on a held-out split (build_datasets' val_dataset),
    disjoint from the puzzles the probe trains on.

Usage:
    python experiments/train_digit_probe.py \\
      experiment=mnist_trm_diffusion_backbone \\
      checkpoint=runs/mnist_trm_diffusion_backbone/checkpoint_final.pt \\
      eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
      +digit_probe.num_steps=500

    python experiments/train_digit_probe.py \\
      experiment=mnist_thinker_v1_controlnet \\
      painter.checkpoint=runs/mnist_unet_painter/checkpoint_final.pt \\
      checkpoint=runs/mnist_thinker_x0hint_v1_80/checkpoint_final.pt \\
      +digit_probe.num_steps=500

    # Options (all under +digit_probe.*):
    #   num_steps        — probe training iterations, one full sampling
    #                       trajectory per step (default 500)
    #   batch_size       — puzzles per trajectory (default 8 — full-
    #                       trajectory z_H capture is memory-heavy: every
    #                       reasoning iteration of every denoising step is
    #                       held in memory at once for that step's update)
    #   lr               — probe optimizer lr (default 1e-3)
    #   log_every        — default 20
    #   eval_every       — run held-out + floor eval every N steps (default 100)
    #   num_eval_batches — held-out batches per eval (default 4)
    #   out              — output path (default: alongside checkpoint,
    #                       named digit_probe.pt)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ablate_trm_loop_budget import _load_checkpoint
from factory import build_datasets, build_model
from models.digit_probe import DigitProbe
from perturbation_recovery_probe import _decode_cellwise

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _sample_trajectory_capture_zH(
    model, conditions, x_init: torch.Tensor, num_inference_steps: int, cfg_scale: float, classifier, cell_size: int
) -> tuple[torch.Tensor, list]:
    """Runs one real, full CFG-guided generation trajectory, capturing the
    CONDITIONAL branch's z_H after every reasoning iteration of every
    denoising step. Returns (final_digits, zH_list):
      final_digits — (B, 81) long, the CNN's classification of the ONE
        genuinely clean final image (the training label for every entry
        in zH_list).
      zH_list — list of (B, seq_len, hidden_size) tensors, one per
        reasoning iteration across the whole trajectory (length =
        num_inference_steps * n_sup).
    """
    device = x_init.device
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    x = x_init.clone()
    zH_list: list = []

    for t in model.scheduler.timesteps:
        t_batch = t.expand(x.shape[0]).to(device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)

        # z_H/z_L intentionally NOT threaded across denoising steps: the
        # real, deployed model.forward() always calls forward_with_carry
        # with z_H=z_L=None, resetting the carry fresh at every denoising
        # step (see ThinkerFrozenPainterBase.forward). Passing the
        # previous step's returned carry back in here would silently
        # simulate ablate_trm_loop_budget.py's "carry" axis instead of
        # real inference — out-of-training-distribution and not what the
        # probe should be trained on.
        step_zH: list = []
        pred_c, _, _ = model.forward_with_carry(step_sample, zH_out=step_zH)
        zH_list.extend(step_zH)
        noise_pred = pred_c.pred

        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            pred_u, _, _ = model.forward_with_carry(null_sample)
            noise_pred = pred_u.pred + cfg_scale * (noise_pred - pred_u.pred)

        x = model.scheduler.step(noise_pred, t, x).prev_sample

    final_image = model.decode_for_eval(x)
    final_digits = _decode_cellwise(final_image, classifier, cell_size)
    return final_digits, zH_list


def _probe_loss(probe: DigitProbe, zH_list: list, final_digits: torch.Tensor, blank: torch.Tensor, puzzle_emb_len: int) -> torch.Tensor:
    total = 0.0
    for z_H in zH_list:
        logits = probe(z_H[:, puzzle_emb_len:])  # (B, 81, 9)
        total = total + F.cross_entropy(logits[blank], final_digits[blank])
    return total / len(zH_list)


@torch.no_grad()
def _evaluate_probe(
    model, probe: DigitProbe, classifier, cell_size: int, eval_batches: list, num_inference_steps: int,
    cfg_scale: float, puzzle_emb_len: int, device,
) -> dict:
    """Held-out accuracy (pooled across every captured reasoning
    iteration, plus separately the very first and very last iteration of
    each trajectory as a cheap "does it rise" check) and the control-task
    floor (probe accuracy on the model's constant, pre-reasoning initial
    z_H — see module docstring)."""
    reasoner = getattr(model, "thinker", model)
    total_correct = total_n = 0
    first_correct = first_n = 0
    last_correct = last_n = 0
    floor_correct = floor_n = 0

    for batch in eval_batches:
        conditions = model._batch_to_sample(batch, device)
        bsz = batch["solution"].shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)
        final_digits, zH_list = _sample_trajectory_capture_zH(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size
        )
        given_mask = conditions.solution_mask
        blank = (~given_mask.to(device)) if given_mask is not None else torch.ones_like(final_digits, dtype=torch.bool)

        for i, z_H in enumerate(zH_list):
            preds = probe(z_H[:, puzzle_emb_len:]).argmax(-1)
            correct = (preds == final_digits) & blank
            total_correct += int(correct.sum().item())
            total_n += int(blank.sum().item())
            if i == 0:
                first_correct += int(correct.sum().item())
                first_n += int(blank.sum().item())
            if i == len(zH_list) - 1:
                last_correct += int(correct.sum().item())
                last_n += int(blank.sum().item())

        z_H_init, _ = reasoner.get_initial_states(bsz)
        z_H_init = z_H_init.to(device)
        floor_preds = probe(z_H_init[:, puzzle_emb_len:]).argmax(-1)
        floor_c = (floor_preds == final_digits) & blank
        floor_correct += int(floor_c.sum().item())
        floor_n += int(blank.sum().item())

    return {
        "pooled_acc": total_correct / total_n if total_n else None,
        "first_iter_acc": first_correct / first_n if first_n else None,
        "last_iter_acc": last_correct / last_n if last_n else None,
        "floor_acc": floor_correct / floor_n if floor_n else None,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/train_digit_probe.py experiment=<name> "
            "checkpoint=<path/to/checkpoint.pt> [+digit_probe.xxx=...]"
        )

    dp = cfg.get("digit_probe", {})
    num_steps: int = dp.get("num_steps", 500)
    batch_size: int = dp.get("batch_size", 8)
    lr: float = dp.get("lr", 1e-3)
    log_every: int = dp.get("log_every", 20)
    eval_every: int = dp.get("eval_every", 100)
    num_eval_batches: int = dp.get("num_eval_batches", 4)
    out_path: str = dp.get("out", str(Path(checkpoint).parent / "digit_probe.pt"))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    if not hasattr(model, "forward_with_carry"):
        raise SystemExit(
            "This model has no forward_with_carry — the digit probe requires a TRM-based "
            "model (TRMDiffusionBackbone or ThinkerFrozenPainterBase)."
        )

    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    reasoner = getattr(model, "thinker", model)
    hidden_size = reasoner.inner.config.hidden_size
    puzzle_emb_len = reasoner.inner.puzzle_emb_len

    sudoku_cb = next((c for c in model.eval_callbacks if getattr(c, "eval_clf", None) is not None), None)
    if sudoku_cb is None:
        raise SystemExit("No eval callback with a loaded classifier (eval_clf) found on the model.")
    classifier = sudoku_cb.eval_clf
    cell_size = sudoku_cb.cell_size

    pipeline = model.sampling_pipeline
    cfg_scale: float = dp.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = dp.get("num_inference_steps", pipeline.num_inference_steps)

    train_ds, val_ds = build_datasets(cfg)
    collate_fn = getattr(type(train_ds), "collate_fn", None)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True, collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=True, collate_fn=collate_fn,
    )

    logger.info(
        f"Training digit probe for {num_steps} steps (hidden_size={hidden_size}, "
        f"num_inference_steps={num_inference_steps}, n_sup={model.n_sup}, cfg_scale={cfg_scale}) — "
        "everything else in the checkpoint is loaded and never updated."
    )

    probe = DigitProbe(hidden_size).to(device)
    probe_optim = torch.optim.Adam(probe.parameters(), lr=lr)

    train_iter = iter(train_dl)

    def _next_train_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            return next(train_iter)

    recent_losses: list[float] = []
    for step in range(num_steps):
        batch = _next_train_batch()
        conditions = model._batch_to_sample(batch, device)
        bsz = batch["solution"].shape[0]
        x_init = torch.randn(bsz, *model.noise_shape, device=device)

        final_digits, zH_list = _sample_trajectory_capture_zH(
            model, conditions, x_init, num_inference_steps, cfg_scale, classifier, cell_size
        )
        given_mask = conditions.solution_mask
        blank = (~given_mask.to(device)) if given_mask is not None else torch.ones_like(final_digits, dtype=torch.bool)

        loss = _probe_loss(probe, zH_list, final_digits, blank, puzzle_emb_len)
        probe_optim.zero_grad()
        loss.backward()
        probe_optim.step()

        recent_losses.append(loss.item())
        if step % log_every == 0:
            window = recent_losses[-log_every:]
            logger.info(f"step={step}  probe_loss={loss.item():.4f}  (last {len(window)} avg={sum(window)/len(window):.4f})")

        if step % eval_every == 0 and step > 0:
            val_iter = iter(val_dl)
            eval_batches = [next(val_iter) for _ in range(min(num_eval_batches, len(val_dl)))]
            stats = _evaluate_probe(
                model, probe, classifier, cell_size, eval_batches, num_inference_steps, cfg_scale, puzzle_emb_len, device
            )
            logger.info(f"step={step}  held-out eval: {stats}")

    val_iter = iter(val_dl)
    eval_batches = [next(val_iter) for _ in range(min(num_eval_batches, len(val_dl)))]
    final_stats = _evaluate_probe(
        model, probe, classifier, cell_size, eval_batches, num_inference_steps, cfg_scale, puzzle_emb_len, device
    )
    logger.info(f"Final held-out eval: {final_stats}")

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "probe_state": probe.state_dict(),
                "hidden_size": hidden_size,
                "puzzle_emb_len": puzzle_emb_len,
                "checkpoint": str(checkpoint),
                "final_eval": final_stats,
            },
            out_path,
        )
        logger.info(f"Saved → {out_path}")

    return final_stats


if __name__ == "__main__":
    main()
