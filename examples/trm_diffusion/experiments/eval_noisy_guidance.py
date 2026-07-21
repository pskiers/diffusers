"""
eval_noisy_guidance.py — Sweep noisy-guidance schedules for V1 models.

At each denoising step the noise prediction is blended:
    pred = pred_noisy + s(t,T) * (pred_clean - pred_noisy)

where pred_noisy uses x_t as the encoder's noisy channel (standard V1),
and pred_clean uses zeros instead (V0-equivalent behaviour).

s=0  → standard V1 baseline
s=1  → full suppression of x_noisy (V0-equivalent)
s>1  → CFG-style extrapolation (amplifies noisy suppression beyond V0)

Schedules parameterised as  s(t, T) = a + b*(t/T).
Positive b → more guidance at high t (noisy end of trajectory).
Negative b → more guidance at low  t (clean end, shortcut regime).

Requires:
  - A V1 checkpoint (mode=painter --painter_variant v1).
  - The model must have been trained with noisy_dropout_p_max > 0, otherwise
    pred_clean is out-of-distribution and guidance will be incoherent.

Usage:
    python eval/eval_noisy_guidance.py \\
        --checkpoint runs/thinker_frozen_painter_v1_minsnr5_dropout05/checkpoint_final.pt \\
        --painter_variant v1 \\
        --classifier_path runs/mnist_classifier_cell16.pt \\
        --num_samples 512 --batch_size 64
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── Local imports (run from trm_diffusion/) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.eval_painter import build_model, load_checkpoint, _get_condition
from eval.mnist_eval import evaluate_grids, load_or_train_classifier, sample_grids


# ── Schedule definitions ──────────────────────────────────────────────────────
#
# Each entry: (label, fn(t, T) -> s)
#   t ∈ [0, T-1] decreasing during denoising.
#   a + b*(t/T): b<0 means MORE guidance as t approaches 0 (clean end).
#
def _lin(a, b):
    return lambda t, T: a + b * (t / T)


SCHEDULES = [
    # Constant baselines
    ("const_0.0  [baseline V1]", _lin(0.0, 0.0)),
    ("const_0.3", _lin(0.3, 0.0)),
    ("const_0.5", _lin(0.5, 0.0)),
    ("const_0.7", _lin(0.7, 0.0)),
    ("const_1.0  [V0-equiv]", _lin(1.0, 0.0)),
    # Linear: more guidance toward clean end (t→0), negative b
    ("lin  a=0.0 b=-0.5  [0→0.5]", _lin(0.5, -0.5)),  # s goes 0.5→0 over traj
    ("lin  a=0.0 b=-1.0  [0→1.0]", _lin(1.0, -1.0)),  # s goes 1.0→0
    ("lin  a=0.5 b=-0.5  [0.5→1]", _lin(1.0, -0.5)),  # s goes 1.0→0.5
    # Linear: more guidance toward noisy end (t→T), positive b
    ("lin  a=0.5 b=+0.5  [0→1,v]", _lin(0.0, 0.5)),  # s goes 0→0.5
    ("lin  a=1.0 b=-0.5  [0.5→1]", _lin(0.5, 0.5)),  # s goes 0.5→1.0
    # CFG-style extrapolation (s > 1 at some steps)
    ("const_1.5  [CFG extrap]", _lin(1.5, 0.0)),
]


def run_schedule(model, loader, classifier, args, guidance_fn, device):
    """Run one full sampling eval with the given guidance schedule."""
    painter_size = 9 * args.cell_size
    sample_steps = args.num_train_timesteps if args.sampler == "ddpm" else args.num_steps

    all_cell_acc = []
    all_puzzle_acc = []
    all_constraint_acc = []
    all_given_consistent_acc = []
    n_done = 0

    for batch in tqdm(
        loader, desc="sampling", leave=False, total=(args.num_samples + args.batch_size - 1) // args.batch_size
    ):
        if n_done >= args.num_samples:
            break
        cond = _get_condition(batch, model, device="cpu")
        sols = batch["solution"]
        pids = batch.get("puzzle_id")
        given_masks = batch.get("given_mask")

        with torch.no_grad():
            sr = sample_grids(
                model,
                cond,
                num_train_timesteps=args.num_train_timesteps,
                beta_schedule=args.beta_schedule,
                prediction_type=args.prediction_type,
                num_steps=sample_steps,
                device=device,
                puzzle_ids=pids,
                solutions=sols,
                painter_size=painter_size,
                given_masks=given_masks,
                noisy_guidance_fn=guidance_fn,
            )

        acc = evaluate_grids(sr["generated"], sols, classifier, args.cell_size, given_masks=given_masks)
        all_cell_acc.append(acc["cell_acc"])
        all_puzzle_acc.append(acc["puzzle_acc"])
        all_constraint_acc.append(acc.get("constraint_puzzle_acc", 0.0))
        if acc.get("given_consistent_puzzle_acc") is not None:
            all_given_consistent_acc.append(acc["given_consistent_puzzle_acc"])
        n_done += sols.shape[0]

    result = {
        "cell_acc": float(np.mean(all_cell_acc)),
        "puzzle_acc": float(np.mean(all_puzzle_acc)),
        "constraint_puzzle_acc": float(np.mean(all_constraint_acc)),
    }
    if all_given_consistent_acc:
        result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent_acc))
    return result


def main():
    p = argparse.ArgumentParser(description="Sweep noisy-guidance schedules for V1 models.")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--classifier_path", default="runs/mnist_classifier_cell16.pt")
    p.add_argument("--sudoku_dir", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root", default="data/mnist")
    p.add_argument("--cell_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule", default="squaredcos_cap_v2")
    p.add_argument("--prediction_type", default="sample", choices=["sample", "epsilon"])
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=2.0,
                   help="CFG scale applied during sampling (must match eval.cfg_scale from training)")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture (must match the checkpoint)
    p.add_argument("--mode", default="thinker_frozen_painter", choices=["painter", "thinker_frozen_painter"])
    p.add_argument("--painter_variant", default="v1", choices=["v0", "v1", "v2", "v3", "v4", "v0tok"])
    p.add_argument("--vocab_size", type=int, default=11)
    p.add_argument("--seq_len", type=int, default=81)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--L_layers", type=int, default=2)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--n_sup", type=int, default=16)
    p.add_argument("--expansion", type=float, default=4.0)
    p.add_argument("--forward_dtype", default="bfloat16")
    p.add_argument("--mlp_t", action="store_true")
    p.add_argument("--pos_encodings", default="rope")
    p.add_argument("--puzzle_emb_ndim", type=int, default=0)
    p.add_argument("--puzzle_emb_len", type=int, default=16)
    p.add_argument("--num_puzzle_identifiers", type=int, default=1000)
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--thinker_out_channels", type=int, default=None)
    p.add_argument("--enc_channels", type=int, default=32)
    p.add_argument("--enc_hidden_channels", type=int, nargs="+", default=[16, 32])
    p.add_argument("--bridge_channels", type=int, default=16)
    p.add_argument("--painter_channels", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block", type=int, default=2)
    p.add_argument("--thinker_bridge_mode", default="logits", choices=["logits", "onehot", "softmax"])
    p.add_argument("--adapter_in_channels", type=int, default=0)
    p.add_argument("--painter_dtype", default="bfloat16", choices=["bfloat16", "float16", "none"])

    # Timestep conditioning (must match checkpoint training flags)
    p.add_argument("--enc_timestep_cond", action="store_true")
    p.add_argument("--thinker_timestep_cond", action="store_true")
    p.add_argument("--decoder_timestep_cond", action="store_true")
    p.add_argument("--temb_dim", type=int, default=256)

    args = p.parse_args()
    if args.painter_dtype == "none":
        args.painter_dtype = None

    device = torch.device(args.device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dir = os.path.join(args.sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=eval_dir,
        mnist_root=args.mnist_root,
        cell_size=args.cell_size,
        mnist_split="test",
        mask_given=True,
    )
    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"Dataset: {len(eval_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args)
    if hasattr(model, "eval_cfg"):
        model.eval_cfg.cfg_scale = args.cfg_scale
    model = load_checkpoint(model, args.checkpoint, use_ema=not args.no_ema, device=device)
    model.eval()

    classifier = load_or_train_classifier(
        args.classifier_path,
        mnist_root=args.mnist_root,
        cell_size=args.cell_size,
        device=device,
    )

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(
        f"\nSweeping {len(SCHEDULES)} guidance schedules "
        f"({args.num_samples} samples, {args.num_steps} DDIM steps)\n"
    )
    print(f"{'Schedule':<40}  {'cell_acc':>9}  {'puzzle_acc':>10}")
    print("-" * 64)

    results = []
    for label, fn in SCHEDULES:
        m = run_schedule(model, loader, classifier, args, fn, device)
        results.append((label, m))
        print(f"{label:<40}  {m['cell_acc']:>9.4f}  {m['puzzle_acc']:>10.4f}")

    # Summary: best schedule
    best = max(results, key=lambda r: r[1]["puzzle_acc"])
    print(f"\nBest puzzle_acc: {best[0].strip()}")


if __name__ == "__main__":
    main()
