"""
grad_verify_experiment.py — Gradient signal verification through the frozen painter.

For a trained StandalonePainter, checks whether backpropagating loss through the
bridge + UNet provides meaningful gradient signal on the conditioning logits.

Experiment design (local perturbation):
  For each true digit d and conditioning type c:
    1. Start from fully correct conditioning for ALL cells.
    2. Override only the cells where true_digit == d with conditioning for digit c
       (or uniform).  Rest of the grid remains correctly conditioned.
    3. Forward through bridge → painter → x0_pred.
    4. Compute loss: MSE(x0_pred, clean image) or CE on classifier(x_{t-1}) cells.
    5. Backprop; L2-normalise per-cell gradient; collect only cells where true_digit==d.

Expected heatmap pattern for true digit d, row c:
  row c == d (same digit = no perturbation):  all gray  (correct cond, ~zero gradient)
  row c != d (wrong digit):  col d positive (push toward true),
                              col c negative (push away from wrong), rest gray

Conditioning types per heatmap (10 rows):
  "d=1" .. "d=9"  — d-cells set to that digit; the row c==d is the "no-op" baseline
  "uniform"        — d-cells set to zero logits (uniform softmax)

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

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.mnist_eval import load_or_train_classifier
from models.utility_models import strip_compiled_prefix
from models.painters import StandalonePainter

VOCAB_SIZE   = 11
DIGIT_OFFSET = 2       # token for digit d (1-indexed) = d + DIGIT_OFFSET  [2..10]
IGNORE_IDX   = -100
LOGIT_PEAK   = 5.0     # logit for "active" digit; rest=0 → softmax ≈ one-hot

# 10 conditioning types per heatmap: digits 1-9 (indices 0-8) + uniform (index 9)
N_COND     = 10
COND_NAMES = [f"d={d}" for d in range(1, 10)] + ["uniform"]
GRAD_LABELS = [f"d={d}" for d in range(1, 10)]  # heatmap columns


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

def make_logits_for_digit(true_d, cond_c, sol_2d, B, grid, device):
    """Build (B, VOCAB_SIZE, grid, grid) logit tensor with local perturbation.

    Starts from correct per-cell conditioning for ALL cells, then overrides
    only the cells where true_digit == true_d.

    true_d: 0-indexed true digit (0..8) — which cells to perturb
    cond_c: 0..8 → peak at that digit (cond_c==true_d means no perturbation)
            9    → zero logits on those cells (uniform softmax)
    sol_2d: (B, grid, grid) int64, 0-indexed digit 0..8 or IGNORE_IDX
    """
    # Start: correct conditioning for every cell.
    sol_clamped = sol_2d.clamp(min=0)                      # (B, 9, 9)
    token_idx   = (sol_clamped + DIGIT_OFFSET).unsqueeze(1)  # (B, 1, 9, 9)
    logits = torch.zeros(B, VOCAB_SIZE, grid, grid, device=device)
    logits.scatter_(1, token_idx, LOGIT_PEAK)

    # Identify cells to perturb: true digit == true_d.
    perturb = (sol_2d == true_d)  # (B, 9, 9) bool

    if perturb.any():
        # Clear those cells.
        logits[:, :, :, :].masked_fill_(
            perturb.unsqueeze(1).expand_as(logits), 0.0
        )
        if cond_c < 9:
            # Peak at digit cond_c+1 (1-indexed), token = cond_c + DIGIT_OFFSET.
            logits[:, cond_c + DIGIT_OFFSET, :, :].masked_fill_(perturb, LOGIT_PEAK)
        # cond_c == 9: leave as zeros → uniform softmax on those cells.

    return logits


def x0_from_pred(noise_pred, noisy, timesteps, scheduler):
    pt = scheduler.config.prediction_type
    if pt == "epsilon":
        ab  = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
        sa  = ab.sqrt().view(-1, 1, 1, 1)
        s1a = (1 - ab).sqrt().view(-1, 1, 1, 1)
        x0  = (noisy - s1a * noise_pred.float()) / sa
    else:  # "sample"
        x0 = noise_pred.float()
    return x0.clamp(0, 1)


def ddim_prev_sample(x0_pred, noisy, timesteps, scheduler):
    """Deterministic DDIM x_{t-1}. Differentiable w.r.t. x0_pred."""
    dev     = noisy.device
    ab_t    = scheduler.alphas_cumprod.to(dev)[timesteps]
    ab_tm1  = scheduler.alphas_cumprod.to(dev)[(timesteps - 1).clamp(min=0)]
    sa_t    = ab_t.sqrt().view(-1, 1, 1, 1)
    s1a_t   = (1 - ab_t).sqrt().view(-1, 1, 1, 1)
    sa_tm1  = ab_tm1.sqrt().view(-1, 1, 1, 1)
    s1a_tm1 = (1 - ab_tm1).sqrt().view(-1, 1, 1, 1)
    eps     = (noisy - sa_t * x0_pred) / s1a_t
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
        x_tm1   = ddim_prev_sample(x0_pred, noisy, timesteps, scheduler)
        cells   = extract_cells(x_tm1, cell_size)
        clf_out = classifier(cells)
        labels  = solution_flat.reshape(-1).to(images.device)
        loss    = F.cross_entropy(clf_out, labels, ignore_index=IGNORE_IDX)

    loss.backward()
    return logits.grad.detach()


# ── Accumulation ──────────────────────────────────────────────────────────────

def accumulate_for_digit(grads, solution_flat, true_d, store):
    """Collect L2-normalised per-cell gradients for cells where true_digit == true_d.

    grads:         (B, VOCAB_SIZE, 9, 9)
    solution_flat: (B, 81) int64
    store:         list — appended with (N_cells, 9) tensor
    """
    B = grads.shape[0]
    grad_digits = grads[:, DIGIT_OFFSET:DIGIT_OFFSET + 9, :, :]  # (B, 9, 9, 9)
    grad_flat   = grad_digits.permute(0, 2, 3, 1).reshape(B * 81, 9).cpu().float()
    sol_flat    = solution_flat.reshape(B * 81).cpu()

    mask = (sol_flat == true_d)
    if not mask.any():
        return
    g     = grad_flat[mask]                                 # (N_cells, 9)
    norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    store.append(g / norms)


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(painter, loader, scheduler, classifier, args, device):
    bridge        = painter.bridge
    painter_mod   = painter.painter
    painter_dtype = painter._painter_dtype
    grid          = 9
    timesteps_list = args.timesteps

    # buckets[(lt, true_d, cond_c, t)] -> list of (N_cells, 9) tensors
    buckets = defaultdict(list)

    n_total = (args.num_batches * len(timesteps_list)
               * 9 * N_COND * 2)  # batches × timesteps × true_digits × cond × losses
    print(f"Total forward+backward passes: ~{n_total}")

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

            for true_d in range(9):
                for cond_c in tqdm(range(N_COND),
                                   desc=f"  t={t_val} d={true_d+1}",
                                   leave=False):
                    logits_init = make_logits_for_digit(
                        true_d, cond_c, sol_2d, B, grid, device
                    )
                    for lt in ("mse", "classifier"):
                        grads = compute_grad(
                            logits_init, images, noisy, t_tensor, solution,
                            bridge, painter_mod, classifier, scheduler,
                            args.cell_size, painter_dtype, lt,
                        )
                        accumulate_for_digit(
                            grads, solution.cpu(), true_d, buckets[(lt, true_d, cond_c, t_val)]
                        )

    # Aggregate
    results = {}
    for key, chunks in buckets.items():
        results[key] = (torch.cat(chunks, dim=0).mean(0).numpy()
                        if chunks else np.zeros(9))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, timesteps_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for lt in ("mse", "classifier"):
        for t_val in timesteps_list:
            fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
            fig.suptitle(
                f"Mean L2-normalised gradient on digit logits\n"
                f"loss={lt}  |  timestep t={t_val}\n"
                f"Only perturbed-digit cells counted.  "
                f"Blue=negative (↑logit→↓loss), Red=positive.",
                fontsize=10,
            )

            for d in range(9):
                ax = axes[d // 3][d % 3]

                # Build (N_COND=10, 9) matrix.
                mat = np.stack(
                    [results.get((lt, d, cond_c, t_val), np.zeros(9))
                     for cond_c in range(N_COND)],
                    axis=0,
                )

                vmax = max(np.abs(mat).max(), 1e-6)
                im = ax.imshow(mat, vmin=-vmax, vmax=vmax,
                               cmap="RdBu_r", aspect="auto")

                # Row labels: mark the "no-op" row (cond == true digit).
                row_labels = list(COND_NAMES)
                row_labels[d] = f"d={d+1} ✓"
                ax.set_title(f"true digit = {d + 1}", fontsize=9)
                ax.set_yticks(range(N_COND))
                ax.set_yticklabels(row_labels, fontsize=7)
                ax.set_xticks(range(9))
                ax.set_xticklabels(GRAD_LABELS, fontsize=7, rotation=45)
                ax.set_xlabel("grad dim", fontsize=7)
                ax.set_ylabel("cond on d-cells", fontsize=7)

                # Golden border around the "correct" column.
                ax.axvline(x=d - 0.5, color="gold", linewidth=1.5, alpha=0.8)
                ax.axvline(x=d + 0.5, color="gold", linewidth=1.5, alpha=0.8)
                # Dashed line below the "no-op" row.
                ax.axhline(y=d + 0.5, color="gold", linewidth=1.0,
                           linestyle="--", alpha=0.6)
                ax.axhline(y=d - 0.5, color="gold", linewidth=1.0,
                           linestyle="--", alpha=0.6)

                plt.colorbar(im, ax=ax, shrink=0.8)

            path = os.path.join(output_dir, f"grad_{lt}_t{t_val:03d}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved {path}")

    # ── Summary: correct-sign fraction vs timestep ────────────────────────────
    # For each (lt, t, true_d, cond_c) where cond_c != true_d:
    #   correct = gradient at col true_d is negative (pushing toward true digit)
    summary = {}
    for lt in ("mse", "classifier"):
        summary[lt] = {}
        for t_val in timesteps_list:
            signs = []
            for d in range(9):
                for cond_c in range(N_COND):
                    if cond_c == d:          # no-op row, skip
                        continue
                    g = results.get((lt, d, cond_c, t_val), np.zeros(9))
                    signs.append(float(g[d] < 0))
            frac = float(np.mean(signs)) if signs else 0.0
            summary[lt][t_val] = frac
            print(f"  {lt}  t={t_val:3d}  correct-sign fraction: {frac:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, lt in zip(axes, ("mse", "classifier")):
        ts  = sorted(summary[lt])
        frs = [summary[lt][t] for t in ts]
        ax.plot(ts, frs, "o-", color="steelblue")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("timestep t")
        ax.set_ylabel("fraction with correct-sign gradient")
        ax.set_title(f"{lt} loss")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Does backprop push logits toward the correct digit?")
    path = os.path.join(output_dir, "summary_correct_sign.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Gradient signal verification through frozen painter.")

    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--classifier_path",  default="runs/mnist_classifier_cell16.pt")
    p.add_argument("--noisy_classifier", action="store_true")
    p.add_argument("--output_dir",       default="results/grad_verify")

    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)

    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--timesteps",   type=int, nargs="+", default=[10, 25, 50, 75, 90])

    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample",
                   choices=["sample", "epsilon"])

    p.add_argument("--bridge_channels",          type=int,       default=16)
    p.add_argument("--painter_channels",         type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block", type=int,       default=2)
    p.add_argument("--painter_dtype",            default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers",              type=int, default=4)
    p.add_argument("--seed",                     type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

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

    print(f"Timesteps: {args.timesteps}")
    print(f"Batches: {args.num_batches} × {args.batch_size}\n")

    results = run_experiment(painter, loader, scheduler, classifier, args, device)
    plot_results(results, args.timesteps, args.output_dir)
    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
