"""
cond_swap_experiment.py — Conditioning swap experiment for StandalonePainter.

Denoises images starting with cond1, then switches to cond2 at each possible
DDIM step, measuring how the generated image shifts between the two conditionings.

Two experiments:
  1. Two different puzzle solutions: measures cell accuracy against each.
  2. Digit-pair swap (x→y): replaces all x-cells in the solution token with y,
     then measures how many of those cells end up as x vs y in the generated image.
     Runs all 72 ordered pairs of distinct digits 1-9.

The cond1 trajectory is computed once per batch and reused for all swap points /
digit pairs, making the computation tractable.

Usage:
    python cond_swap_experiment.py \\
        --checkpoint runs/standalone_painter/checkpoint_final.pt \\
        --num_batches 5 --batch_size 32 --num_ddim_steps 20 \\
        --output_dir results/swap_exp

Computational cost (approximate):
  Exp 1 : num_batches × (N + N*(N+1)/2) DDIM steps on batch_size samples
  Exp 2 : num_batches × (N + 72 × N*(N+1)/2) DDIM steps on batch_size samples
  For N=20, num_batches=3, batch_size=16 this is ~0.5M model calls — budget ~30 min on GPU.
  Reduce num_batches or num_ddim_steps to speed up.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from tqdm.auto import tqdm

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.mnist_eval import evaluate_grids, load_or_train_classifier
from models.utility_models import strip_compiled_prefix
from models.painters import StandalonePainter
from models.trm.ema import EMAHelper


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solution_tokens(solution: torch.Tensor) -> torch.Tensor:
    """(B, 81) int [0-8] → (B, 81) long [2-10]."""
    return (solution.clamp(min=0) + 2).long()


def build_model(args) -> StandalonePainter:
    return StandalonePainter(
        painter_size=9 * args.cell_size,
        cell_size=args.cell_size,
        vocab_size=args.vocab_size,
        bridge_channels=args.bridge_channels,
        painter_channels=tuple(args.painter_channels),
        painter_layers_per_block=args.painter_layers_per_block,
        cfg_prob=0.0,
        cfg_scale=args.cfg_scale,
        painter_dtype=None if args.painter_dtype == "none" else args.painter_dtype,
    )


def load_checkpoint(model, path, use_ema, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compiled_prefix(ckpt["model_state"]))
    if use_ema and ckpt.get("ema_state") is not None:
        shadow = strip_compiled_prefix(ckpt["ema_state"])
        param_dict = dict(model.named_parameters())
        n = sum(1 for k, v in shadow.items()
                if k in param_dict and not param_dict[k].data.copy_(v) is None)
        print(f"EMA: applied {n}/{len(shadow)} shadow params from {path}")
    else:
        print(f"Loaded weights from {path}")
    model.to(device)
    return model


# ── Core DDIM primitives ──────────────────────────────────────────────────────

@torch.no_grad()
def run_trajectory(model, x_init, cond, timesteps, ddim, device):
    """Run full DDIM with cond, return list of N+1 intermediate states (including x_init)."""
    states = [x_init]
    x = x_init
    cond = cond.to(device)
    for t in timesteps:
        ts = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        noise_pred, _ = model(x, ts, cond)
        x = ddim.step(noise_pred, int(t), x).prev_sample
        states.append(x)
    return states   # states[k] = image after k denoising steps with cond


@torch.no_grad()
def continue_from(model, state, cond, timesteps_remaining, ddim, device):
    """Continue denoising from a saved state with new conditioning."""
    x = state.clone()
    cond = cond.to(device)
    for t in timesteps_remaining:
        ts = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        noise_pred, _ = model(x, ts, cond)
        x = ddim.step(noise_pred, int(t), x).prev_sample
    return x


def _swap_timesteps(timesteps, k):
    """Timestep at which the swap happens (noise level when we switch to cond2).
    k=0 → swap at start (highest noise), k=N → no swap."""
    N = len(timesteps)
    return int(timesteps[k]) if k < N else 0


# ── Experiment 1: two different puzzle conditionings ──────────────────────────

def run_experiment_1(model, loader, ddim, timesteps, classifier, args, device):
    """
    For each swap step k in [0..N]:
      - Run k DDIM steps with cond1 (puzzle A)
      - Continue with cond2 (puzzle B, rolled by 1 in batch)
    Measures cell accuracy against both solutions.
    """
    N = len(timesteps)
    painter_size = 9 * args.cell_size

    # Accumulators: list over swap points, each entry = list of batch values
    acc1_all = [[] for _ in range(N + 1)]
    acc2_all = [[] for _ in range(N + 1)]

    for bi, batch in enumerate(tqdm(loader, desc="Exp1 batches", total=args.num_batches)):
        if bi >= args.num_batches:
            break

        sol1  = batch["solution"].to(device)           # (B, 81) int 0-8
        sol2  = torch.roll(sol1, shifts=1, dims=0)     # different solution for each sample
        given1 = batch.get("given_mask")
        given2 = torch.roll(given1, 1, 0) if given1 is not None else None

        cond1 = _solution_tokens(sol1)
        cond2 = _solution_tokens(sol2)

        x_init = torch.randn(sol1.shape[0], 1, painter_size, painter_size, device=device)
        states = run_trajectory(model, x_init, cond1, timesteps, ddim, device)

        for k in range(N + 1):
            x_final = continue_from(model, states[k], cond2, timesteps[k:], ddim, device)
            x_cpu   = x_final.clamp(0, 1).cpu()
            g1_cpu  = given1.cpu() if given1 is not None else None
            g2_cpu  = given2.cpu() if given2 is not None else None
            acc1_all[k].append(evaluate_grids(x_cpu, sol1.cpu(), classifier, args.cell_size,
                                              given_masks=g1_cpu)["cell_acc"])
            acc2_all[k].append(evaluate_grids(x_cpu, sol2.cpu(), classifier, args.cell_size,
                                              given_masks=g2_cpu)["cell_acc"])

    swap_ts = [_swap_timesteps(timesteps, k) for k in range(N + 1)]
    return {
        "swap_steps":      list(range(N + 1)),
        "swap_timesteps":  swap_ts,
        "cond1_cell_acc":  [float(np.mean(acc1_all[k])) for k in range(N + 1)],
        "cond2_cell_acc":  [float(np.mean(acc2_all[k])) for k in range(N + 1)],
    }


# ── Experiment 2: digit pair swap ─────────────────────────────────────────────

def run_experiment_2(model, loader, ddim, timesteps, classifier, args, device):
    """
    For each ordered digit pair (x, y) (digits 1-9, 72 pairs):
      Build cond2 = cond1 with all x-cells replaced by y.
      For each swap step k, continue denoising from states[k] with cond2.
      Among cells that originally had digit x, count fraction now showing x vs y.
    """
    N        = len(timesteps)
    painter_size = 9 * args.cell_size
    digits   = list(range(9))                             # 0-8 (0-indexed, = digit-1)
    pairs    = [(x, y) for x in digits for y in digits if x != y]  # 72 ordered pairs

    # results[pair_key][k] = {"count_x": [], "count_y": []}
    results = {
        f"{x+1}->{y+1}": {
            "count_x": [[] for _ in range(N + 1)],
            "count_y": [[] for _ in range(N + 1)],
        }
        for x, y in pairs
    }

    for bi, batch in enumerate(tqdm(loader, desc="Exp2 batches", total=args.num_batches)):
        if bi >= args.num_batches:
            break

        sol   = batch["solution"].to(device)    # (B, 81) int 0-8
        cond1 = _solution_tokens(sol)           # (B, 81) tokens 2-10
        B     = sol.shape[0]

        x_init = torch.randn(B, 1, painter_size, painter_size, device=device)
        states = run_trajectory(model, x_init, cond1, timesteps, ddim, device)

        for x_d, y_d in tqdm(pairs, desc=f"  batch {bi+1} digit pairs", leave=False):
            mask_x = (sol == x_d).cpu()                 # (B, 81) bool — cells with digit x
            if mask_x.sum() == 0:
                continue

            # Build cond2: replace all x-cells with y token
            cond2 = cond1.clone()
            cond2[sol == x_d] = y_d + 2                 # token = 0-indexed digit + 2

            key = f"{x_d+1}->{y_d+1}"
            for k in range(N + 1):
                x_final = continue_from(model, states[k], cond2, timesteps[k:], ddim, device)
                preds   = evaluate_grids(x_final.clamp(0, 1).cpu(), sol.cpu(),
                                         classifier, args.cell_size)["preds"]  # (B, 81) 0-8

                for b in range(B):
                    cells = mask_x[b]                   # (81,) bool
                    if cells.sum() == 0:
                        continue
                    p = preds[b][cells]
                    results[key]["count_x"][k].append((p == x_d).float().mean().item())
                    results[key]["count_y"][k].append((p == y_d).float().mean().item())

    swap_ts = [_swap_timesteps(timesteps, k) for k in range(N + 1)]
    # Collapse accumulators to means
    output = {"swap_steps": list(range(N + 1)), "swap_timesteps": swap_ts, "pairs": {}}
    for key, data in results.items():
        cx = [float(np.mean(data["count_x"][k])) if data["count_x"][k] else float("nan")
              for k in range(N + 1)]
        cy = [float(np.mean(data["count_y"][k])) if data["count_y"][k] else float("nan")
              for k in range(N + 1)]
        output["pairs"][key] = {"mean_count_x": cx, "mean_count_y": cy}
    return output


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_experiment_1(data, output_dir):
    ts   = data["swap_timesteps"]
    acc1 = data["cond1_cell_acc"]
    acc2 = data["cond2_cell_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, acc1, "o-", label="cond1 (original)", color="steelblue")
    ax.plot(ts, acc2, "s-", label="cond2 (new)",      color="tomato")
    ax.set_xlabel("Timestep at which conditioning is swapped\n"
                  "(high = swap early / mostly cond2, 0 = no swap / pure cond1)")
    ax.set_ylabel("Cell accuracy")
    ax.set_title("Conditioning swap: cond1 → cond2 at different denoising timesteps")
    ax.legend()
    ax.invert_xaxis()   # left = late swap (pure cond1), right = early swap (pure cond2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "exp1_cond_swap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_experiment_2(data, output_dir):
    pairs     = sorted(data["pairs"].keys())
    swap_ts   = data["swap_timesteps"]
    N_pairs   = len(pairs)

    # ── Per-pair grid plot ────────────────────────────────────────────────────
    ncols = 9
    nrows = (N_pairs + ncols - 1) // ncols   # 8 rows of 9 = 72
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)

    for idx, key in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        d   = data["pairs"][key]
        ax.plot(swap_ts, d["mean_count_x"], "o-", color="steelblue",
                label=f"digit {key.split('->')[0]}", linewidth=1.2, markersize=3)
        ax.plot(swap_ts, d["mean_count_y"], "s-", color="tomato",
                label=f"digit {key.split('->')[1]}", linewidth=1.2, markersize=3)
        ax.set_title(key, fontsize=9)
        ax.set_ylim(0, 1)
        ax.invert_xaxis()
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Fraction of cells", fontsize=7)

    # Hide unused subplots
    for idx in range(N_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Digit pair swap: fraction of x-cells showing original (x) vs new (y) digit\n"
                 "x-axis = timestep of cond swap (inverted: left=pure cond1, right=pure cond2)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "exp2_digit_pairs_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")

    # ── Summary heatmap: at mid-trajectory swap, how dominant is new digit? ──
    # For each pair (x→y), "influence" = mean_count_y at the midpoint swap step
    mid_k = len(swap_ts) // 2
    influence = np.full((9, 9), np.nan)
    for key, d in data["pairs"].items():
        xd, yd = (int(v) - 1 for v in key.split("->"))
        v = d["mean_count_y"][mid_k]
        if not np.isnan(v):
            influence[xd, yd] = v

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(influence, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("New digit (y)")
    ax.set_ylabel("Original digit (x)")
    ax.set_title(f"Fraction of x-cells that adopted digit y\n"
                 f"(cond swap at mid-trajectory, timestep={swap_ts[mid_k]})")
    fig.colorbar(im, ax=ax, label="mean_count_y")
    fig.tight_layout()
    path = os.path.join(output_dir, "exp2_heatmap_mid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Conditioning swap experiment for StandalonePainter.")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="results/cond_swap")
    p.add_argument("--no_ema",     action="store_true")

    # Data
    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)
    p.add_argument("--classifier_path", default="runs/mnist_classifier_cell16.pt")

    # Experiment scale
    p.add_argument("--num_batches",    type=int, default=3,
                   help="Batches to average over per experiment")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--num_ddim_steps", type=int, default=20,
                   help="DDIM steps N. Creates N+1 swap points.")
    p.add_argument("--skip_exp2",  action="store_true",
                   help="Skip the digit-pair experiment (much slower than exp1)")

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample",
                   choices=["sample", "epsilon"])
    p.add_argument("--cfg_scale",           type=float, default=1.0)

    # Sampling seed
    p.add_argument("--seed", type=int, default=42)

    # Architecture (must match checkpoint)
    p.add_argument("--vocab_size",           type=int,   default=11)
    p.add_argument("--bridge_channels",      type=int,   default=16)
    p.add_argument("--painter_channels",     type=int,   nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block", type=int, default=2)
    p.add_argument("--painter_dtype",        default="bfloat16",
                   choices=["bfloat16", "float16", "none"])
    p.add_argument("--num_workers",          type=int,   default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Dataset & classifier ─────────────────────────────────────────────────
    test_dir = os.path.join(args.sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    ds = MNISTSudokuDataset(sudoku_dir=eval_dir, mnist_root=args.mnist_root,
                             cell_size=args.cell_size, mnist_split="test", mask_given=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))

    classifier = load_or_train_classifier(
        args.classifier_path, mnist_root=args.mnist_root,
        cell_size=args.cell_size, device=device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint, use_ema=not args.no_ema, device=device)
    model.eval()

    # ── DDIM scheduler ───────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    ddim.set_timesteps(args.num_ddim_steps)
    timesteps = ddim.timesteps.tolist()   # descending, e.g. [99, 94, 89, ...]

    print(f"DDIM schedule: {len(timesteps)} steps, "
          f"t=[{timesteps[0]}..{timesteps[-1]}], "
          f"{len(timesteps)+1} swap points to evaluate")

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    print("\n─── Experiment 1: two-puzzle conditioning swap ───")
    exp1 = run_experiment_1(model, loader, ddim, timesteps, classifier, args, device)
    plot_experiment_1(exp1, args.output_dir)

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    exp2 = None
    if not args.skip_exp2:
        print("\n─── Experiment 2: digit-pair swap (72 pairs × "
              f"{len(timesteps)+1} swap points × {args.num_batches} batches) ───")
        n_calls = args.num_batches * (len(timesteps) + 72 * len(timesteps) * (len(timesteps) + 1) // 2)
        print(f"  Estimated model calls: ~{n_calls:,}  (use --skip_exp2 to skip)")
        exp2 = run_experiment_2(model, loader, ddim, timesteps, classifier, args, device)
        plot_experiment_2(exp2, args.output_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "args":         vars(args),
        "ddim_timesteps": timesteps,
        "experiment_1": exp1,
        "experiment_2": exp2,
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {json_path}")


if __name__ == "__main__":
    main()
