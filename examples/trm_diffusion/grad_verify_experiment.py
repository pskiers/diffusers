"""
grad_verify_experiment.py — Gradient signal verification through the frozen painter.

For a trained StandalonePainter, checks whether backpropagating loss through the
bridge + UNet provides meaningful gradient signal on the spatial conditioning logits
(i.e., whether gradients push the logits toward the correct digit).

Experiment design:
  For each (conditioning type, timestep, loss type):
    1. Build soft conditioning logits peaked at the given digit / uniform / correct.
    2. Forward through bridge → painter → x0_pred.
    3. Compute loss: MSE(x0_pred, clean image) or CE on classifier(x_{t-1}) cells.
    4. Backprop to the logit tensor; L2-normalise per-cell gradient vector.
    5. Stratify by true digit and average across cells and batches.

Conditioning types (11 total):
  "d=1" .. "d=9" — all cells set to that digit (one-hot-ish via large logit)
  "uniform"       — all logits zero → uniform softmax
  "correct"       — per-cell one-hot at the ground-truth digit

Output:
  For each (loss type, timestep): a figure with 9 heatmaps (one per true digit),
  each showing rows=cond_type, cols=grad_dim_for_digit, value=mean_norm_grad.

Usage:
    python grad_verify_experiment.py \\
        --checkpoint runs/standalone_painter/checkpoint_final.pt \\
        --classifier_path runs/noisy_classifier/classifier_noisy.pt \\
        --noisy_classifier \\
        --output_dir results/grad_verify
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from mnist_sudoku_dataset import MNISTSudokuDataset
from mnist_eval import load_or_train_classifier
from models_pt import strip_compiled_prefix
from trm_wrappers import StandalonePainter

VOCAB_SIZE    = 11
DIGIT_OFFSET  = 2        # token for digit d (1-indexed) = d + DIGIT_OFFSET  [2..10]
IGNORE_IDX    = -100
LOGIT_PEAK    = 5.0      # logit for "active" digit; rest=0 → softmax ≈ one-hot

N_COND        = 11       # 0..8 = digit 1..9, 9 = uniform, 10 = correct
COND_NAMES    = [f"d={d}" for d in range(1, 10)] + ["uniform", "correct"]
GRAD_LABELS   = [f"d={d}" for d in range(1, 10)]  # columns in the heatmap


# ── Model loading ──────────────────────────────────────────────────────────────

def build_painter(args, device):
    model = StandalonePainter(
        painter_size=9 * args.cell_size,
        cell_size=args.cell_size,
        vocab_size=VOCAB_SIZE,
        bridge_channels=args.bridge_channels,
        painter_channels=tuple(args.painter_channels),
        painter_layers_per_block=args.painter_layers_per_block,
        cfg_prob=0.0,
        cfg_scale=1.0,
        painter_dtype=None if args.painter_dtype == "none" else args.painter_dtype,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Prefer EMA weights if present.
    raw = ckpt.get("ema_state") or ckpt["model_state"]
    model.load_state_dict(strip_compiled_prefix(raw))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded painter from {args.checkpoint}")
    return model


def build_classifier(args, device):
    if args.noisy_classifier:
        from train_noisy_classifier import load_noisy_classifier
        clf = load_noisy_classifier(args.classifier_path, device)
    else:
        clf = load_or_train_classifier(
            args.classifier_path,
            mnist_root=args.mnist_root,
            cell_size=args.cell_size,
            device=device,
        )
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf.eval()


# ── Core primitives ────────────────────────────────────────────────────────────

def make_logits(cond_type, sol_2d, B, grid, device):
    """Build (B, VOCAB_SIZE, grid, grid) float logit tensor.

    sol_2d:    (B, grid, grid) int64, 0-indexed digit 0..8 or IGNORE_IDX
    cond_type: 0..8 = all cells peaked at digit cond_type+1
               9    = uniform (all zeros)
               10   = per-cell peaked at the true digit
    """
    logits = torch.zeros(B, VOCAB_SIZE, grid, grid, device=device)
    if cond_type < 9:
        logits[:, cond_type + DIGIT_OFFSET, :, :] = LOGIT_PEAK
    elif cond_type == 9:
        pass  # uniform
    else:
        # correct: scatter peak at true-digit token per cell
        sol_clamped = sol_2d.clamp(min=0)                      # (B, 9, 9)
        token_idx = (sol_clamped + DIGIT_OFFSET).unsqueeze(1)  # (B, 1, 9, 9)
        logits.scatter_(1, token_idx, LOGIT_PEAK)
    return logits


def x0_from_pred(noise_pred, noisy, timesteps, scheduler):
    pt = scheduler.config.prediction_type
    if pt == "epsilon":
        ab   = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
        sa   = ab.sqrt().view(-1, 1, 1, 1)
        s1a  = (1 - ab).sqrt().view(-1, 1, 1, 1)
        x0   = (noisy - s1a * noise_pred.float()) / sa
    else:  # "sample"
        x0 = noise_pred.float()
    return x0.clamp(0, 1)


def ddim_prev_sample(x0_pred, noisy, timesteps, scheduler):
    """Deterministic DDIM x_{t-1}. Differentiable w.r.t. x0_pred."""
    dev    = noisy.device
    ab_t   = scheduler.alphas_cumprod.to(dev)[timesteps]
    ab_tm1 = scheduler.alphas_cumprod.to(dev)[(timesteps - 1).clamp(min=0)]
    sa_t   = ab_t.sqrt().view(-1, 1, 1, 1)
    s1a_t  = (1 - ab_t).sqrt().view(-1, 1, 1, 1)
    sa_tm1 = ab_tm1.sqrt().view(-1, 1, 1, 1)
    s1a_tm1 = (1 - ab_tm1).sqrt().view(-1, 1, 1, 1)
    eps    = (noisy - sa_t * x0_pred) / s1a_t
    return (sa_tm1 * x0_pred + s1a_tm1 * eps).clamp(0, 1)


def extract_cells(imgs, cell_size):
    B = imgs.shape[0]
    cells = imgs.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    return cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)


def compute_grad(logits_init, images, noisy, timesteps, solution_flat,
                 bridge, painter_module, classifier, scheduler,
                 cell_size, painter_dtype, loss_type):
    """One forward + backward pass. Returns (B, VOCAB_SIZE, 9, 9) gradient tensor."""
    logits = logits_init.detach().clone().requires_grad_(True)
    soft   = logits.softmax(dim=1)

    ctx = (torch.autocast(device_type=images.device.type, dtype=painter_dtype)
           if painter_dtype is not None
           else torch.autocast(device_type=images.device.type, enabled=False))

    with ctx:
        bridge_feat = bridge(soft)
        noise_pred  = painter_module(
            torch.cat([noisy, bridge_feat], dim=1), timesteps
        ).sample

    x0_pred = x0_from_pred(noise_pred, noisy, timesteps, scheduler)

    if loss_type == "mse":
        loss = F.mse_loss(x0_pred, images.float())
    else:  # "classifier"
        x_tm1     = ddim_prev_sample(x0_pred, noisy, timesteps, scheduler)
        cells     = extract_cells(x_tm1, cell_size)
        clf_out   = classifier(cells)
        labels    = solution_flat.reshape(-1).to(images.device)
        loss      = F.cross_entropy(clf_out, labels, ignore_index=IGNORE_IDX)

    loss.backward()
    return logits.grad.detach()


# ── Accumulation ──────────────────────────────────────────────────────────────

def accumulate(grads, solution_flat, store):
    """Vectorised accumulation of L2-normalised per-cell gradients.

    grads:         (B, VOCAB_SIZE, 9, 9)
    solution_flat: (B, 81) int64 0-8 or IGNORE_IDX
    store:         dict  true_digit (0-8) -> list of (N_cells, 9) tensors
    """
    B    = grads.shape[0]
    grid = 9
    # Digit-only dimensions: (B, 9, 9, 9) → reshape to (B*81, 9)
    grad_digits = grads[:, DIGIT_OFFSET:DIGIT_OFFSET + 9, :, :]
    grad_flat   = grad_digits.permute(0, 2, 3, 1).reshape(B * 81, 9).cpu().float()
    sol_flat    = solution_flat.reshape(B * 81).cpu()

    norms    = grad_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    grad_norm = grad_flat / norms  # (B*81, 9)

    valid = sol_flat >= 0
    for d in range(9):
        mask = valid & (sol_flat == d)
        if mask.any():
            store[d].append(grad_norm[mask])


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(painter, loader, scheduler, classifier, args, device):
    bridge        = painter.bridge
    painter_mod   = painter.painter
    painter_dtype = painter._painter_dtype
    grid          = 9
    timesteps_list = args.timesteps

    # buckets[(loss_type, cond_type, t)] -> {true_digit: [tensors]}
    buckets = defaultdict(lambda: defaultdict(list))

    for bi, batch in enumerate(tqdm(loader, desc="batches", total=args.num_batches)):
        if bi >= args.num_batches:
            break

        images   = batch["images"].to(device)    # (B, 1, H, W)
        solution = batch["solution"].to(device)  # (B, 81) 0-8 or -100
        B        = images.shape[0]
        sol_2d   = solution.reshape(B, grid, grid)

        for t_val in timesteps_list:
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise    = torch.randn_like(images)
            with torch.no_grad():
                noisy = scheduler.add_noise(images, noise, t_tensor)

            for ct in tqdm(range(N_COND), desc=f"  t={t_val} cond", leave=False):
                logits_init = make_logits(ct, sol_2d, B, grid, device)

                for lt in ("mse", "classifier"):
                    grads = compute_grad(
                        logits_init, images, noisy, t_tensor, solution,
                        bridge, painter_mod, classifier, scheduler,
                        args.cell_size, painter_dtype, lt,
                    )
                    key = (lt, ct, t_val)
                    accumulate(grads, solution.cpu(), buckets[key])

    # Aggregate mean normalised gradient per (loss_type, cond_type, t, true_digit)
    results = {}
    for (lt, ct, t_val), store in buckets.items():
        for d in range(9):
            vecs = store[d]
            mean_grad = (torch.cat(vecs, dim=0).mean(0).numpy()
                         if vecs else np.zeros(9))
            results[(lt, ct, t_val, d)] = mean_grad
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, timesteps_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for lt in ("mse", "classifier"):
        for t_val in timesteps_list:
            fig, axes = plt.subplots(
                3, 3,
                figsize=(13, 11),
                constrained_layout=True,
            )
            fig.suptitle(
                f"Mean L2-normalised gradient on digit logits\n"
                f"loss={lt}  |  timestep t={t_val}\n"
                f"rows=conditioning type, cols=grad dim (digit 1-9)\n"
                f"blue=negative (loss↓ if logit↑), red=positive",
                fontsize=10,
            )

            for d in range(9):
                ax = axes[d // 3][d % 3]

                # Build (N_COND, 9) heatmap matrix
                mat = np.stack(
                    [results.get((lt, ct, t_val, d), np.zeros(9))
                     for ct in range(N_COND)],
                    axis=0,
                )  # (11, 9)

                vmax = max(np.abs(mat).max(), 1e-6)
                im = ax.imshow(mat, vmin=-vmax, vmax=vmax,
                               cmap="RdBu_r", aspect="auto")
                ax.set_title(f"true digit = {d + 1}", fontsize=9)
                ax.set_yticks(range(N_COND))
                ax.set_yticklabels(COND_NAMES, fontsize=7)
                ax.set_xticks(range(9))
                ax.set_xticklabels(GRAD_LABELS, fontsize=7, rotation=45)
                ax.set_xlabel("grad dim", fontsize=7)
                ax.set_ylabel("cond type", fontsize=7)
                # Mark the "correct" column for this true digit
                ax.axvline(x=d - 0.5, color="gold", linewidth=1.5, alpha=0.7)
                ax.axvline(x=d + 0.5, color="gold", linewidth=1.5, alpha=0.7)
                plt.colorbar(im, ax=ax, shrink=0.8)

            path = os.path.join(output_dir, f"grad_{lt}_t{t_val:03d}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved {path}")

    # Summary: for each (loss_type, t), compute "correct-sign fraction"
    # = fraction of (cond_type, true_digit) pairs where gradient on correct dim is negative
    summary = {}
    for lt in ("mse", "classifier"):
        summary[lt] = {}
        for t_val in timesteps_list:
            correct_signs = []
            for ct in range(N_COND - 1):  # exclude "correct" cond (trivially 0 gradient)
                for d in range(9):
                    g = results.get((lt, ct, t_val, d), np.zeros(9))
                    correct_signs.append(g[d] < 0)  # gradient on correct dim is negative
            frac = float(np.mean(correct_signs)) if correct_signs else 0.0
            summary[lt][t_val] = frac
            print(f"  {lt}  t={t_val:3d}  correct-sign fraction: {frac:.3f}")

    # Summary line plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, lt in zip(axes, ("mse", "classifier")):
        ts  = sorted(summary[lt])
        frs = [summary[lt][t] for t in ts]
        ax.plot(ts, frs, "o-", color="steelblue")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
        ax.set_xlabel("timestep t")
        ax.set_ylabel("fraction correct-sign")
        ax.set_title(f"{lt} loss — fraction of cells with\ncorrect-direction gradient")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Gradient sign diagnostic: does backprop point toward the correct digit?")
    path = os.path.join(output_dir, "summary_correct_sign.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Gradient signal verification through frozen painter.")

    p.add_argument("--checkpoint",      required=True,
                   help="Path to StandalonePainter checkpoint")
    p.add_argument("--classifier_path", default="runs/mnist_classifier_cell16.pt")
    p.add_argument("--noisy_classifier", action="store_true",
                   help="Load classifier via load_noisy_classifier() (for noisy-augmented model)")
    p.add_argument("--output_dir",      default="results/grad_verify")

    # Data
    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)

    # Experiment scale
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--batch_size",  type=int, default=8,
                   help="Smaller batches are fine; we're computing gradients, not training")
    p.add_argument("--timesteps",   type=int, nargs="+", default=[10, 25, 50, 75, 90],
                   help="Timesteps to evaluate (e.g. 10 25 50 75 90)")

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int,    default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample",
                   choices=["sample", "epsilon"])

    # Architecture (must match checkpoint)
    p.add_argument("--bridge_channels",           type=int,       default=16)
    p.add_argument("--painter_channels",          type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block",  type=int,       default=2)
    p.add_argument("--painter_dtype",             default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers",               type=int,       default=4)
    p.add_argument("--seed",                      type=int,       default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    test_dir = os.path.join(args.sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    ds = MNISTSudokuDataset(
        sudoku_dir=eval_dir, mnist_root=args.mnist_root,
        cell_size=args.cell_size, mnist_split="test", mask_given=True,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    painter    = build_painter(args, device)
    classifier = build_classifier(args, device)

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    print(f"\nTimesteps to test: {args.timesteps}")
    print(f"Conditioning types: {COND_NAMES}")
    print(f"Batches: {args.num_batches} × {args.batch_size} samples\n")

    results = run_experiment(painter, loader, scheduler, classifier, args, device)
    plot_results(results, args.timesteps, args.output_dir)
    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
