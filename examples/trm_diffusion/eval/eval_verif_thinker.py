"""
eval_verif_thinker.py — Evaluation of ThinkerWithFrozenPainterV1Verif with
classifier guidance and/or adaptive re-noising using the verifier head.

Techniques (combinable — both can be active simultaneously):

  Classifier guidance
    At each DDIM step: ∇_{x_t} log σ(verif(z_H)) is subtracted from the
    noise prediction to steer the trajectory toward consistent solutions.
    Two modes (sweepable via --guidance_grad both):
      detach : gradient flows only through the linear verif_head (fast)
      full   : gradient flows through all n_sup thinker steps (expensive)

  Adaptive re-noising
    After each DDIM step: if σ(verif(z_H)) < threshold, x is re-noised back.
    Two modes (sweepable via --renoise_mode both):
      fixed_k   : re-noise back k steps at once (--renoise_k K)
      iterative : re-noise 1 step at a time, re-check, repeat until
                  score ≥ threshold or --max_retries hit

Usage examples:
  # Guidance-only sweep
  python eval/eval_verif_thinker.py \\
      --checkpoint runs/thinker_frozen_painter_v1_verif/checkpoint_final.pt \\
      --guidance_scale 0.0 0.5 1.0 2.0 --guidance_grad both

  # Re-noising sweep with both modes
  python eval/eval_verif_thinker.py \\
      --checkpoint ... \\
      --renoise_threshold 0.0 0.3 0.5 0.7 --renoise_mode both --renoise_k 1 3 5

  # Both combined
  python eval/eval_verif_thinker.py \\
      --checkpoint ... \\
      --guidance_scale 0.0 1.0 --renoise_threshold 0.0 0.5 --renoise_mode fixed_k --renoise_k 3
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.schemas import (
    EvalConfig, ImageEncoderConfig, PainterConfig, PainterOptimConfig,
    PainterThinkerConfig, ThinkerModelConfig, ThinkerOptimConfig,
    TimestepCondConfig, TrainConfig,
)
from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.mnist_eval import evaluate_grids, load_or_train_classifier
from models.painter_thinkers import ThinkerWithFrozenPainterV1Verif
from models.painters import StandalonePainter
from models.utility_models import strip_compiled_prefix


# ── Model construction ────────────────────────────────────────────────────────

def build_model(args, scheduler) -> ThinkerWithFrozenPainterV1Verif:
    cell_size  = args.cell_size
    painter_size = 9 * cell_size
    dummy_optim  = PainterOptimConfig(lr=1e-4, weight_decay=1.0, warmup_steps=0)
    thinker_optim = ThinkerOptimConfig(
        lr=1e-4, weight_decay=1.0, beta1=0.9, beta2=0.95, warmup_steps=0
    )
    train_cfg = TrainConfig(seed=0, batch_size=args.batch_size, num_steps=1)
    eval_cfg  = EvalConfig(
        eval_every=1, save_every=1, log_every=1, cfg_scale=1.0,
        num_samples=args.num_samples, batch_size=args.batch_size,
        num_ddim_steps=args.num_steps, classifier_path=args.classifier_path,
    )
    painter_cfg = PainterConfig(
        vocab_size=11, painter_size=painter_size, cell_size=cell_size,
        bridge_channels=args.bridge_channels,
        painter_channels=tuple(args.painter_channels),
        painter_layers_per_block=args.painter_layers_per_block,
        painter_dtype=args.painter_dtype,
    )
    dummy_painter = StandalonePainter(
        model_cfg=painter_cfg, optim_cfg=dummy_optim,
        train_cfg=train_cfg, eval_cfg=eval_cfg, scheduler=scheduler,
    )
    model_cfg = PainterThinkerConfig(
        painter_size=painter_size, cell_size=cell_size,
        bridge_channels=args.bridge_channels,
        painter_channels=tuple(args.painter_channels),
        painter_layers_per_block=args.painter_layers_per_block,
        thinker_bridge_mode=args.thinker_bridge_mode,
        painter_dtype=args.painter_dtype,
    )
    thinker_cfg = ThinkerModelConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, hidden_size=args.hidden_size,
        n_heads=args.n_heads, L_layers=args.L_layers, L_cycles=args.L_cycles,
        H_cycles=args.H_cycles, n_sup=args.n_sup, batch_size=args.batch_size,
        forward_dtype=args.forward_dtype, expansion=args.expansion,
        puzzle_emb_ndim=0, puzzle_emb_len=16, num_puzzle_identifiers=1,
    )
    encoder_cfg = ImageEncoderConfig(
        enc_channels=args.enc_channels,
        enc_hidden_channels=tuple(args.enc_hidden_channels),
    )
    return ThinkerWithFrozenPainterV1Verif(
        painter=dummy_painter,
        thinker_cfg=thinker_cfg,
        encoder_cfg=encoder_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        thinker_optim_cfg=thinker_optim,
        painter_optim_cfg=dummy_optim,
        scheduler=scheduler,
        verif_weight=args.verif_weight,
        verif_max_corruptions=args.verif_max_corruptions,
        timestep_cfg=TimestepCondConfig(
            enc_timestep_cond=args.enc_timestep_cond,
            thinker_timestep_cond=args.thinker_timestep_cond,
            decoder_timestep_cond=args.decoder_timestep_cond,
        ),
    )


def load_checkpoint(model, path, use_ema, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=False)
    if use_ema and ckpt.get("ema_state"):
        shadow = strip_compiled_prefix(ckpt["ema_state"])
        params = dict(model.named_parameters())
        n = sum(1 for name, t in shadow.items()
                if name in params and not params[name].data.copy_(t) is None)
        print(f"EMA: applied {n}/{len(shadow)} shadow params")
    model.to(device)
    return model


# ── Core sampling helper ──────────────────────────────────────────────────────

def _verif_only(model, x, ts, condition):
    """Run encoder+thinker, return verifier score without painter. For re-noising checks."""
    B = x.shape[0]
    z_H, z_L = model.get_initial_states(B)
    z_H, z_L = z_H.to(x.device), z_L.to(x.device)
    enc_emb = model._get_enc_emb(condition, x, timesteps=ts)
    for _ in range(model.n_sup):
        _, z_H, z_L = model.thinker.reasoning_step(enc_emb, z_H, z_L)
    seq_len = model.thinker.inner.config.seq_len
    feats = z_H[:, :seq_len, :].float().mean(dim=1).detach()
    return torch.sigmoid(model.verif_head(feats).squeeze(-1))   # (B,)


def sample_with_verif(
    model,
    condition,           # (B, 1, H, W) pixel condition image
    device,
    scheduler_train,
    num_ddim_steps=20,
    painter_size=144,
    # Classifier guidance
    guidance_scale=0.0,
    guidance_grad="detach",     # "detach" | "full"
    # Re-noising
    renoise_threshold=0.0,      # 0 = disabled
    renoise_mode="fixed_k",     # "fixed_k" | "iterative"
    renoise_k=3,
    max_retries=10,
    # Painter CFG
    cfg_scale=1.0,
):
    """
    DDIM loop with optional classifier guidance and/or adaptive re-noising.
    Both can be active simultaneously.
    Returns (B, 1, painter_size, painter_size) images in [0, 1].
    """
    B = condition.shape[0]
    condition = condition.to(device)

    ddim = DDIMScheduler(
        num_train_timesteps=scheduler_train.config.num_train_timesteps,
        beta_schedule=scheduler_train.config.beta_schedule,
        prediction_type=scheduler_train.config.prediction_type,
    )
    ddim.set_timesteps(num_ddim_steps)
    timesteps = list(ddim.timesteps)
    alphas_cumprod = ddim.alphas_cumprod.to(device)

    x = torch.randn(B, 1, painter_size, painter_size, device=device)
    model.eval_cfg.cfg_scale = cfg_scale

    i = 0
    while i < len(timesteps):
        t   = int(timesteps[i])
        ts  = torch.full((B,), t, device=device, dtype=torch.long)

        # ── Forward pass (with or without guidance gradient) ──────────────────
        if guidance_scale > 0:
            # "detach" = cheap: only last thinker step in grad graph (n_sup_grad=1)
            # "full"   = accurate: all n_sup steps in grad graph (n_sup_grad=-1)
            n_sup_grad = 1 if guidance_grad == "detach" else -1
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                noise_pred, logits, verif_score = model.forward_with_verif(
                    x_in, ts, condition, n_sup_grad=n_sup_grad
                )
                # ∇_{x_t} log σ(verif) — positive gradient steers toward consistency
                grad = torch.autograd.grad(verif_score.log().sum(), x_in)[0]
            sigma_t = (1.0 - alphas_cumprod[t]).sqrt()
            # Subtract from noise pred (standard classifier-guidance convention)
            noise_pred = (noise_pred - guidance_scale * sigma_t * grad.detach()).detach()
            verif_score = verif_score.detach()
        else:
            with torch.no_grad():
                noise_pred, logits, verif_score = model.forward_with_verif(
                    x, ts, condition, n_sup_grad=1  # no grad needed, run 1 step for score
                )

        # Standard DDIM step
        x_next = ddim.step(noise_pred, t, x).prev_sample

        # ── Adaptive re-noising ───────────────────────────────────────────────
        if renoise_threshold > 0:
            t_next_i = i + 1
            t_next   = int(timesteps[t_next_i]) if t_next_i < len(timesteps) else 0
            ts_next  = torch.full((B,), t_next, device=device, dtype=torch.long)
            # x0 estimate: for "sample" prediction, noise_pred IS x0_pred
            x0_est   = noise_pred.clamp(-1.0, 1.0)

            if renoise_mode == "fixed_k":
                with torch.no_grad():
                    score = _verif_only(model, x_next, ts_next, condition)
                if score.mean().item() < renoise_threshold:
                    back_i  = min(i + renoise_k, len(timesteps) - 1)
                    t_back  = int(timesteps[back_i])
                    ab_back = alphas_cumprod[t_back]
                    noise   = torch.randn_like(x_next)
                    x_next  = ab_back.sqrt() * x0_est + (1 - ab_back).sqrt() * noise
                    i = back_i - 1   # will be incremented below

            elif renoise_mode == "iterative":
                # Step back 1 at a time until score ≥ threshold or max_retries
                retries = 0
                j = i   # current position in schedule (will increase as we go back)
                x_cur = x_next
                while retries < max_retries:
                    j_check = j + 1
                    if j_check >= len(timesteps):
                        break
                    t_check = int(timesteps[j_check])
                    ts_check = torch.full((B,), t_check, device=device, dtype=torch.long)
                    with torch.no_grad():
                        score = _verif_only(model, x_cur, ts_check, condition)
                    if score.mean().item() >= renoise_threshold:
                        break
                    # Re-noise 1 step back
                    ab_back = alphas_cumprod[t_check]
                    noise   = torch.randn_like(x_cur)
                    x_cur   = ab_back.sqrt() * x0_est + (1 - ab_back).sqrt() * noise
                    j       = j_check
                    retries += 1
                x_next = x_cur
                i = j - 1   # will be incremented below

        x = x_next
        i += 1

    return x.clamp(0.0, 1.0)


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(model, loader, classifier, args, device, cfg):
    painter_size = 9 * args.cell_size
    all_cell, all_puzzle = [], []
    n_done = 0
    model.eval()

    desc = (f"gs={cfg['guidance_scale']:.1f}/{cfg['guidance_grad']}  "
            f"th={cfg['renoise_threshold']:.2f}/{cfg['renoise_mode']}"
            + (f"/k={cfg['renoise_k']}" if cfg['renoise_mode'] == 'fixed_k' else ""))

    for batch in tqdm(loader, desc=desc, leave=False):
        if n_done >= args.num_samples:
            break
        solutions   = batch["solution"]
        condition   = batch["conditions"].to(device)
        given_masks = batch.get("given_mask")

        generated = sample_with_verif(
            model, condition, device,
            scheduler_train=model.scheduler,
            num_ddim_steps=args.num_steps,
            painter_size=painter_size,
            guidance_scale=cfg["guidance_scale"],
            guidance_grad=cfg["guidance_grad"],
            renoise_threshold=cfg["renoise_threshold"],
            renoise_mode=cfg["renoise_mode"],
            renoise_k=cfg["renoise_k"],
            max_retries=args.max_retries,
            cfg_scale=cfg["cfg_scale"],
        )

        acc = evaluate_grids(generated.cpu(), solutions, classifier,
                             args.cell_size, given_masks=given_masks)
        all_cell.append(acc["cell_acc"])
        all_puzzle.append(acc["puzzle_acc"])
        n_done += solutions.shape[0]

    return {
        **{k: v for k, v in cfg.items()},
        "cell_acc":   float(np.mean(all_cell)),
        "puzzle_acc": float(np.mean(all_puzzle)),
        "n_samples":  n_done,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--classifier_path", default="runs/mnist_classifier_cell16.pt")

    # Data
    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample")

    # Sampling
    p.add_argument("--num_steps",   type=int, default=20)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--no_ema",  action="store_true")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Guidance sweep ────────────────────────────────────────────────────────
    p.add_argument("--guidance_scale", type=float, nargs="+", default=[0.0],
                   help="Guidance scale(s) to sweep. 0 = disabled.")
    p.add_argument("--guidance_grad",  choices=["detach", "full", "both"],
                   default="detach",
                   help="'detach': grad only through verif_head (cheap). "
                        "'full': grad through all thinker steps. 'both': sweep both.")
    p.add_argument("--cfg_scale", type=float, nargs="+", default=[1.0],
                   help="Painter CFG scale(s) to sweep.")

    # ── Re-noising sweep ──────────────────────────────────────────────────────
    p.add_argument("--renoise_threshold", type=float, nargs="+", default=[0.0],
                   help="Verifier score threshold(s). 0 = disabled.")
    p.add_argument("--renoise_mode", choices=["fixed_k", "iterative", "both"],
                   default="fixed_k",
                   help="Re-noising strategy. 'both' sweeps both modes.")
    p.add_argument("--renoise_k", type=int, nargs="+", default=[3],
                   help="Steps to re-noise back (fixed_k mode); sweepable.")
    p.add_argument("--max_retries", type=int, default=10,
                   help="Max re-noise iterations per step (iterative mode).")

    # ── Model arch (must match checkpoint) ────────────────────────────────────
    p.add_argument("--vocab_size",   type=int,   default=11,
                   help="Thinker vocabulary size — 11 for v1_verif (data.vocab_size), "
                        "9 for digit-class-only thinkers.")
    p.add_argument("--seq_len",      type=int,   default=81)
    p.add_argument("--hidden_size",  type=int,   default=512)
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--L_layers",     type=int,   default=2)
    p.add_argument("--L_cycles",     type=int,   default=6)
    p.add_argument("--H_cycles",     type=int,   default=3)
    p.add_argument("--n_sup",        type=int,   default=16)
    p.add_argument("--expansion",    type=float, default=4.0)
    p.add_argument("--forward_dtype",            default="bfloat16")
    p.add_argument("--bridge_channels",          type=int, default=16)
    p.add_argument("--painter_channels",         type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block", type=int, default=2)
    p.add_argument("--thinker_bridge_mode",      default="softmax")
    p.add_argument("--painter_dtype",            default="bfloat16")
    p.add_argument("--enc_channels",             type=int, default=128)
    p.add_argument("--enc_hidden_channels",      type=int, nargs="+", default=[128, 256, 256])
    p.add_argument("--enc_timestep_cond",        action="store_true")
    p.add_argument("--thinker_timestep_cond",    action="store_true")
    p.add_argument("--decoder_timestep_cond",    action="store_true")
    p.add_argument("--verif_weight",             type=float, default=0.1)
    p.add_argument("--verif_max_corruptions",    type=int,   default=5)

    # Wandb
    p.add_argument("--wandb_project",  default=None)
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


def _build_sweep(args):
    """Build list of config dicts covering all requested parameter combinations."""
    guidance_grads  = (["detach", "full"] if args.guidance_grad == "both"
                       else [args.guidance_grad])
    renoise_modes   = (["fixed_k", "iterative"] if args.renoise_mode == "both"
                       else [args.renoise_mode])

    seen, cfgs = set(), []
    for gs, gg, cs, th, rm, rk in itertools.product(
        args.guidance_scale,
        guidance_grads,
        args.cfg_scale,
        args.renoise_threshold,
        renoise_modes,
        args.renoise_k,
    ):
        # Collapse redundant combinations
        gg_eff = gg if gs > 0 else "detach"   # grad mode irrelevant when gs=0
        rm_eff = rm if th > 0 else "fixed_k"  # renoise mode irrelevant when th=0
        rk_eff = rk if (th > 0 and rm_eff == "fixed_k") else 0

        key = (gs, gg_eff, cs, th, rm_eff, rk_eff)
        if key in seen:
            continue
        seen.add(key)
        cfgs.append({
            "guidance_scale":    gs,
            "guidance_grad":     gg_eff,
            "cfg_scale":         cs,
            "renoise_threshold": th,
            "renoise_mode":      rm_eff,
            "renoise_k":         rk_eff if th > 0 and rm_eff == "fixed_k" else args.renoise_k[0],
        })
    return cfgs


def main():
    args = parse_args()
    if args.painter_dtype == "none":
        args.painter_dtype = None
    device = torch.device(args.device)

    # Dataset
    test_dir = os.path.join(args.sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=eval_dir, mnist_root=args.mnist_root,
        cell_size=args.cell_size, mnist_split="test", mask_given=True,
    )
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Scheduler + model
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    model = build_model(args, scheduler)
    model = load_checkpoint(model, args.checkpoint, use_ema=not args.no_ema, device=device)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}  total={n_all:,}  trainable={n_tr:,}")

    # Classifier
    classifier = load_or_train_classifier(args.classifier_path, args.mnist_root,
                                          args.cell_size, device)

    # Wandb
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=args.wandb_run_name, config=vars(args))
        except ImportError:
            print("wandb not installed; skipping")

    # Sweep
    sweep = _build_sweep(args)
    print(f"\nRunning {len(sweep)} configurations …")
    all_results = []
    for cfg in sweep:
        print(f"\n{'='*65}\n{cfg}\n{'='*65}")
        metrics = run_eval(model, loader, classifier, args, device, cfg)
        all_results.append(metrics)
        print(f"  cell_acc={metrics['cell_acc']:.4f}  puzzle_acc={metrics['puzzle_acc']:.4f}")
        if wandb_run:
            wandb_run.log({f"eval/{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float))})

    # Summary table
    print(f"\n{'='*65}")
    print("Sweep summary:")
    cols = ["guidance_scale", "guidance_grad", "cfg_scale",
            "renoise_threshold", "renoise_mode", "renoise_k",
            "cell_acc", "puzzle_acc"]
    print("  " + "  ".join(f"{c:>22}" for c in cols))
    for r in all_results:
        print("  " + "  ".join(
            f"{r.get(c, ''):>22.4f}" if isinstance(r.get(c), float)
            else f"{str(r.get(c, '-')):>22}"
            for c in cols
        ))

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
