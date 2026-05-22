"""
diagnose_x0pred.py — Diagnostic script for x0_pred / x_{t-1} quality.

Three experiments:
  1. Classifier accuracy on x0_pred and x_{t-1} patches at various timestep bins.
     Both are compared side-by-side so you can see which gives better signal
     for a classifier-based training loss.

  2. Visual comparison grid saved per timestep bin, 4 columns:
       x_t (noisy input) | x0_pred | x_{t-1} | x0_clean (ground truth)

  3. Perceptual / feature loss (L2 in classifier encoder space) vs CE loss
     for both x0_pred and x_{t-1}.

Usage:
    python diagnose_x0pred.py \\
        --painter_ckpt  runs/standalone_painter/checkpoint_final.pt \\
        --classifier_ckpt  runs/noisy_classifier/classifier_noisy.pt \\
        --sudoku_dir  data/mnist_sudoku \\
        --mnist_root  data/mnist \\
        --cell_size   16 \\
        --output_dir  runs/diagnose_x0pred

Pass --noisy_classifier to use load_noisy_classifier() (from train_noisy_classifier.py)
which supports both the plain noisy and the timestep-conditioned variant.
Without that flag, load_or_train_classifier() is used (clean-only classifier).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from eval.mnist_eval import load_or_train_classifier
from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from models.painters import StandalonePainter
from models.utility_models import strip_compiled_prefix
from datasets.sudoku_dataset import IGNORE_LABEL_ID


# ── helpers ────────────────────────────────────────────────────────────────────

def x0_from_noise_pred(noise_pred, noisy, timesteps, scheduler):
    pt = scheduler.config.prediction_type
    alpha_bar = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1_ab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    if pt == "epsilon":
        x0 = (noisy - sqrt_1_ab * noise_pred.float()) / sqrt_ab
    elif pt == "sample":
        x0 = noise_pred.float()
    else:
        raise ValueError(f"Unsupported prediction_type: {pt}")
    return x0.clamp(0.0, 1.0)


def extract_cells(images, cell_size):
    """(B, 1, H, W) → (B*81, 1, cell, cell)."""
    B = images.shape[0]
    cells = (images
             .unfold(2, cell_size, cell_size)
             .unfold(3, cell_size, cell_size))
    return cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B * 81, 1, cell_size, cell_size)


def clf_call(classifier, cells, t_scalar=None, B=None):
    """Call classifier, optionally passing per-cell timestep indices.

    For timestep-conditioned classifiers (have .t_emb), repeats t_scalar 81 times
    per image so shape matches (B*81,).
    """
    if t_scalar is not None and hasattr(classifier, 't_emb'):
        t_cells = torch.full((B * 81,), t_scalar, dtype=torch.long, device=cells.device)
        return classifier(cells, t_cells)
    return classifier(cells)


def clf_accuracy(logits, labels):
    """CE loss and top-1 accuracy, ignoring IGNORE_LABEL_ID cells."""
    mask = labels != IGNORE_LABEL_ID
    if mask.sum() == 0:
        return float("nan"), float("nan")
    preds = logits.argmax(dim=-1)
    acc  = (preds[mask] == labels[mask]).float().mean().item()
    loss = F.cross_entropy(logits[mask], labels[mask]).item()
    return loss, acc


def solution_tokens(solution):
    return solution.clamp(min=0) + 2


# ── Experiment 1 + 2 ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_exp1_exp2(model, classifier, dataloader, scheduler, cell_size, device,
                  n_batches, t_bins, output_dir):
    from PIL import Image as PILImage

    print("\n=== Experiment 1 + 2: classifier accuracy on x0_pred and x_{t-1} ===")

    keys = ("x0_pred", "x_tm1")
    bin_stats = {
        lbl: {k: {"ce_loss": [], "acc": []} for k in keys}
        for lbl in t_bins
    }
    # first batch: store (noisy, x0_pred, x_tm1, clean) per bin
    first_vis  = {}
    first_done = set()

    model.eval()
    classifier.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        images   = batch["images"].to(device)
        solution = batch["solution"].to(device)
        sol_tok  = solution_tokens(solution)
        labels   = solution.reshape(images.shape[0] * 81)
        B        = images.shape[0]
        noise    = torch.randn_like(images)

        for bin_lbl, (t_lo, t_hi) in t_bins.items():
            t_mid     = (t_lo + t_hi) // 2
            timesteps = torch.full((B,), t_mid, device=device, dtype=torch.long)
            noisy     = scheduler.add_noise(images, noise, timesteps)

            noise_pred, _ = model(noisy, timesteps, sol_tok)

            # x0_pred: single-step full denoising prediction
            x0_pred = x0_from_noise_pred(noise_pred, noisy, timesteps, scheduler)

            # x_{t-1}: one DDPM step (still noisy, at level t-1)
            x_tm1 = scheduler.step(noise_pred, t_mid, noisy).prev_sample.clamp(0.0, 1.0)

            for tag, img in (("x0_pred", x0_pred), ("x_tm1", x_tm1)):
                # Pass the noise level the classifier should expect:
                # x0_pred claims to be clean → t=0; x_{t-1} is at level t-1
                t_for_clf = 0 if tag == "x0_pred" else max(0, t_mid - 1)
                cells  = extract_cells(img, cell_size)
                logits = clf_call(classifier, cells, t_scalar=t_for_clf, B=B)
                ce_l, acc = clf_accuracy(logits, labels)
                bin_stats[bin_lbl][tag]["ce_loss"].append(ce_l)
                bin_stats[bin_lbl][tag]["acc"].append(acc)

            if bin_lbl not in first_done and batch_idx == 0:
                first_vis[bin_lbl] = {
                    "noisy":   noisy.cpu(),
                    "x0_pred": x0_pred.cpu(),
                    "x_tm1":   x_tm1.cpu(),
                    "clean":   images.cpu(),
                }
                first_done.add(bin_lbl)

        if (batch_idx + 1) % 5 == 0:
            print(f"  processed {batch_idx + 1}/{n_batches} batches …")

    # ── Report Exp 1 ──────────────────────────────────────────────────────────
    print("\n--- Experiment 1: classifier accuracy ---")
    print(f"{'timestep bin':>18}  {'x0_pred acc':>12}  {'x_{t-1} acc':>12}  {'x0_pred CE':>12}  {'x_{t-1} CE':>12}")
    for bin_lbl in t_bins:
        def _m(tag, metric):
            vals = bin_stats[bin_lbl][tag][metric]
            return np.nanmean(vals) if vals else float("nan")

        print(
            f"  t={bin_lbl:>14}  "
            f"{_m('x0_pred','acc'):>12.4f}  "
            f"{_m('x_tm1',  'acc'):>12.4f}  "
            f"{_m('x0_pred','ce_loss'):>12.4f}  "
            f"{_m('x_tm1',  'ce_loss'):>12.4f}"
        )

    print()
    print("Accuracy near 1/9 ≈ 0.111 means random guessing.")
    print("x_{t-1} accuracy should be higher at high t because it is still")
    print("a noisy image (classifier was trained on noisy images), whereas")
    print("x0_pred is a blurry/imprecise clean prediction.")

    # ── Save Exp 2: 4-column comparison grids ────────────────────────────────
    # columns: x_t (noisy input) | x0_pred | x_{t-1} | x0_clean
    print("\n--- Experiment 2: saving comparison grids (x_t | x0_pred | x_{t-1} | x0_clean) ---")
    os.makedirs(output_dir, exist_ok=True)
    sep_w = 4

    for bin_lbl, vis in first_vis.items():
        n_show   = min(4, vis["clean"].shape[0])
        H        = vis["clean"].shape[2]
        W        = vis["clean"].shape[3]
        cols     = ["noisy", "x0_pred", "x_tm1", "clean"]
        canvas_w = len(cols) * W + (len(cols) - 1) * sep_w
        canvas   = np.full((n_show * H, canvas_w), 128, dtype=np.uint8)

        def to_u8(t, i):
            return (t[i, 0].numpy() * 255).clip(0, 255).astype(np.uint8)

        for i in range(n_show):
            row_y = i * H
            for j, col_key in enumerate(cols):
                x = j * (W + sep_w)
                canvas[row_y:row_y + H, x:x + W] = to_u8(vis[col_key], i)

        fname = os.path.join(output_dir, f"compare_t{bin_lbl.replace(' ', '_')}.png")
        PILImage.fromarray(canvas, mode="L").save(fname)
        print(f"  t={bin_lbl:>5}  → {fname}  (x_t | x0_pred | x_{{t-1}} | x0_clean)")


# ── Experiment 3 ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_exp3(model, classifier, dataloader, scheduler, cell_size, device, n_batches, t_bins):
    print("\n=== Experiment 3: perceptual loss vs CE loss (x0_pred and x_{t-1}) ===")

    bin_stats = {
        lbl: {k: {"ce_loss": [], "feat_loss": []} for k in ("x0_pred", "x_tm1")}
        for lbl in t_bins
    }

    model.eval()
    classifier.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        images   = batch["images"].to(device)
        solution = batch["solution"].to(device)
        sol_tok  = solution_tokens(solution)
        labels   = solution.reshape(images.shape[0] * 81)
        B        = images.shape[0]
        noise    = torch.randn_like(images)

        real_cells = extract_cells(images, cell_size)

        for bin_lbl, (t_lo, t_hi) in t_bins.items():
            t_mid     = (t_lo + t_hi) // 2
            timesteps = torch.full((B,), t_mid, device=device, dtype=torch.long)
            noisy     = scheduler.add_noise(images, noise, timesteps)

            noise_pred, _ = model(noisy, timesteps, sol_tok)
            x0_pred = x0_from_noise_pred(noise_pred, noisy, timesteps, scheduler)
            x_tm1   = scheduler.step(noise_pred, t_mid, noisy).prev_sample.clamp(0.0, 1.0)

            for tag, img in (("x0_pred", x0_pred), ("x_tm1", x_tm1)):
                t_for_clf = 0 if tag == "x0_pred" else max(0, t_mid - 1)
                cells  = extract_cells(img, cell_size)
                logits = clf_call(classifier, cells, t_scalar=t_for_clf, B=B)
                ce_l, _ = clf_accuracy(logits, labels)

                feats_pred = classifier.encoder(cells)
                feats_real = classifier.encoder(real_cells)
                feat_l = F.mse_loss(feats_pred.flatten(1), feats_real.flatten(1)).item()

                bin_stats[bin_lbl][tag]["ce_loss"].append(ce_l)
                bin_stats[bin_lbl][tag]["feat_loss"].append(feat_l)

        if (batch_idx + 1) % 5 == 0:
            print(f"  processed {batch_idx + 1}/{n_batches} batches …")

    print("\n--- CE loss and feature loss by timestep ---")
    for tag, label in (("x0_pred", "x0_pred"), ("x_tm1", "x_{t-1}")):
        print(f"\n  {label}:")
        print(f"  {'timestep bin':>18}  {'CE loss':>12}  {'feat loss':>12}")
        for bin_lbl in t_bins:
            def _m(metric):
                vals = bin_stats[bin_lbl][tag][metric]
                return np.nanmean(vals) if vals else float("nan")
            print(f"    t={bin_lbl:>14}  {_m('ce_loss'):>12.4f}  {_m('feat_loss'):>12.6f}")

    print()
    print("If CE loss is flat / near log(9)≈2.197 at all timesteps, the classifier")
    print("is OOD. Feature loss still varies even when CE saturates — consider")
    print("using it as a training signal instead.")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--painter_ckpt",    required=True)
    p.add_argument("--classifier_ckpt", required=True)
    p.add_argument("--sudoku_dir",      default="data/mnist_sudoku")
    p.add_argument("--mnist_root",      default="data/mnist")
    p.add_argument("--cell_size",       type=int, default=16)
    p.add_argument("--output_dir",      default="runs/diagnose_x0pred")
    p.add_argument("--n_batches",       type=int, default=20)
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_timesteps",   type=int, default=100)
    p.add_argument("--beta_schedule",   default="linear")
    p.add_argument("--prediction_type", default="epsilon")
    p.add_argument("--bridge_channels", type=int, default=16)
    p.add_argument("--painter_channels",nargs="+", type=int, default=[32, 64, 64])
    p.add_argument("--painter_layers",  type=int, default=2)
    p.add_argument("--vocab_size",      type=int, default=11)
    p.add_argument("--noisy_classifier", action="store_true",
                   help="Use load_noisy_classifier() (supports timestep-conditioned variant)")
    p.add_argument("--skip_exp3",       action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    painter_size = 9 * args.cell_size

    T = args.num_timesteps
    bin_edges = np.linspace(0, T, 6, dtype=int)
    t_bins = {}
    for i in range(len(bin_edges) - 1):
        lo, hi = int(bin_edges[i]), int(bin_edges[i + 1]) - 1
        t_bins[f"{lo}-{hi}"] = (lo, hi)
    print(f"Timestep bins: {list(t_bins.keys())}")

    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    print(f"\nLoading StandalonePainter from {args.painter_ckpt} …")
    model = StandalonePainter(
        painter_size=painter_size,
        cell_size=args.cell_size,
        vocab_size=args.vocab_size,
        bridge_channels=args.bridge_channels,
        painter_channels=tuple(args.painter_channels),
        painter_layers_per_block=args.painter_layers,
    ).to(device)

    ckpt  = torch.load(args.painter_ckpt, map_location="cpu", weights_only=False)
    state = strip_compiled_prefix(ckpt.get("model_state", ckpt))
    ema   = ckpt.get("ema_state")
    if ema is not None and "shadow" in ema:
        shadow = {k.replace("_orig_mod.", ""): v for k, v in ema["shadow"].items()}
        model.load_state_dict(shadow, strict=False)
        print("  loaded EMA weights")
    else:
        model.load_state_dict(state, strict=False)
        print("  loaded raw weights")
    model.eval()

    print(f"Loading classifier from {args.classifier_ckpt} …")
    if args.noisy_classifier:
        from train_noisy_classifier import load_noisy_classifier
        classifier = load_noisy_classifier(args.classifier_ckpt, device)
    else:
        classifier = load_or_train_classifier(
            args.classifier_ckpt, args.mnist_root, args.cell_size, device,
        )
    for p in classifier.parameters():
        p.requires_grad_(False)

    # ── Sanity check on clean images ─────────────────────────────────────────
    test_dir  = os.path.join(args.sudoku_dir, "test")
    split_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    dataset   = MNISTSudokuDataset(
        sudoku_dir=split_dir, mnist_root=args.mnist_root,
        cell_size=args.cell_size, mnist_split="test", mask_given=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, drop_last=True)

    print("\n--- Sanity check: classifier on clean images ---")
    with torch.no_grad():
        accs = []
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            images   = batch["images"].to(device)
            solution = batch["solution"].to(device)
            B        = images.shape[0]
            cells    = extract_cells(images, args.cell_size)
            logits   = clf_call(classifier, cells, t_scalar=0, B=B)
            labels   = solution.reshape(B * 81)
            _, acc   = clf_accuracy(logits, labels)
            accs.append(acc)
        print(f"  clean acc = {np.mean(accs):.4f}  (should be ~0.99)")

    run_exp1_exp2(model, classifier, dataloader, scheduler,
                  args.cell_size, device, args.n_batches, t_bins, args.output_dir)

    if not args.skip_exp3:
        run_exp3(model, classifier, dataloader, scheduler,
                 args.cell_size, device, args.n_batches, t_bins)

    print(f"\nDone. Grids saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
