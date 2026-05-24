"""
noise_back_experiment.py — Denoising commitment / noise-back experiment.

At each DDIM step k, re-noise the intermediate image back by M steps (to a higher
noise level), then continue denoising with fresh noise.  Measures the fraction of
cells that match the original fully-denoised image, revealing when the model
"commits" to a solution.

Produces:
  noise_back_heatmap.png   — triangular (k, M) heatmap of cell agreement
  noise_back_curves.png    — per-M commitment curves over denoising steps
  noise_back_results.json  — raw numbers

Usage (Hydra, same as train_trm.py):
  python experiments/noise_back_experiment.py \\
      experiment=thinker_frozen_painter_v0 \\
      painter.painter_checkpoint=runs/standalone_painter/checkpoint_final.pt \\
      checkpoint=runs/thinker_frozen_painter_v0/checkpoint_final.pt \\
      output_dir=results/noise_back \\
      noise_back.num_batches=3 \\
      noise_back.num_ddim_steps=20 \\
      noise_back.num_resamples=3
"""

import json
import logging
import os

import hydra
import matplotlib
import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add parent dir to path so local modules are importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from eval.mnist_eval import evaluate_grids, load_or_train_classifier
from factory import build_model, build_scheduler
from models.utility_models import strip_compiled_prefix

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_cond(model, batch, device):
    if getattr(model, "token_input", True):
        return batch["puzzle_tokens"].to(device)
    return batch["conditions"].to(device)


def _load_model(cfg, scheduler, device):
    model = build_model(cfg, scheduler)
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path is None:
        raise ValueError("checkpoint= must be set to the trained model checkpoint path.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=False)
    model.to(device).eval()
    logger.info(f"Loaded model from {ckpt_path}")
    return model


# ── Re-noising ────────────────────────────────────────────────────────────────


def renoise(x_k, t_k_idx, t_back_idx, scheduler, noise):
    """
    Go from x at timestep t_{k_idx} (less noisy) to t_{back_idx} (more noisy).

    x_{t_back} = sqrt(ā_back / ā_k) * x_k + sqrt(1 - ā_back / ā_k) * noise
    """
    alpha_k = scheduler.alphas_cumprod[t_k_idx].to(x_k.device)
    alpha_back = scheduler.alphas_cumprod[t_back_idx].to(x_k.device)
    ratio = alpha_back / alpha_k
    return ratio.sqrt() * x_k + (1.0 - ratio).sqrt() * noise


# ── Trajectory sampling ───────────────────────────────────────────────────────


@torch.no_grad()
def run_trajectory(model, x_init, cond, timesteps, ddim, device):
    """Full DDIM. Returns list of N+1 states (state[0]=pure noise, state[N]=final)."""
    states = [x_init.clone()]
    x = x_init
    for t in timesteps:
        ts = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        noise_pred, _ = model(x, ts, cond)
        x = ddim.step(noise_pred, int(t), x).prev_sample
        states.append(x.clone())
    return states


@torch.no_grad()
def continue_from(model, x_start, cond, remaining_timesteps, ddim, device):
    """Continue denoising from x_start using remaining_timesteps."""
    x = x_start.clone()
    for t in remaining_timesteps:
        ts = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        noise_pred, _ = model(x, ts, cond)
        x = ddim.step(noise_pred, int(t), x).prev_sample
    return x


# ── Core experiment ───────────────────────────────────────────────────────────


@torch.no_grad()
def run_experiment(model, loader, ddim, timesteps, classifier, nb_cfg, cell_size, device):
    """
    Returns agreement[k][M] = mean cell-agreement with original (and accuracy vs GT).
    agreement[k][M] is None when k < M (invalid).
    """
    N = len(timesteps)
    num_batches = nb_cfg.get("num_batches", 3)
    num_resamples = nb_cfg.get("num_resamples", 3)
    painter_size = cell_size * 9

    # agreement_acc[k][M] = list of per-batch cell-agreement values
    agreement_acc = [[[] for _ in range(N + 1)] for _ in range(N + 1)]

    for bi, batch in enumerate(tqdm(loader, desc="Batches", total=num_batches)):
        if bi >= num_batches:
            break

        solutions = batch["solution"]
        given_masks = batch.get("given_mask")
        cond = _get_cond(model, batch, device)
        B = cond.shape[0]

        x_init = torch.randn(B, 1, painter_size, painter_size, device=device)
        states = run_trajectory(model, x_init, cond, timesteps, ddim, device)

        # Classify original final image once
        orig_final = states[N].clamp(0, 1).cpu()
        orig_preds = evaluate_grids(orig_final, solutions, classifier, cell_size,
                                    given_masks=given_masks)["preds"]  # (B, 81)

        for k in range(1, N + 1):
            t_k = timesteps[k - 1]  # DDIM timestep at step k (0-indexed from high noise)
            # We denoised k steps, so we're AT timestep timesteps[k-1].
            # Going M steps back means going back to timestep timesteps[k-1-M].

            for M in range(1, k + 1):
                t_back = timesteps[k - 1 - M] if (k - M - 1) >= 0 else timesteps[0]
                # timestep index in alphas_cumprod
                t_k_val = int(timesteps[k - 1])
                t_back_val = int(timesteps[k - 1 - M]) if (k - M - 1) >= 0 else int(timesteps[0])

                remaining = timesteps[k - M:]  # timesteps to continue denoising

                batch_agreements = []
                for _ in range(num_resamples):
                    noise = torch.randn_like(states[k])
                    x_back = renoise(states[k], t_k_val, t_back_val, ddim, noise)
                    x_final = continue_from(model, x_back, cond, remaining, ddim, device)
                    preds = evaluate_grids(x_final.clamp(0, 1).cpu(), solutions, classifier, cell_size,
                                          given_masks=given_masks)["preds"]  # (B, 81)
                    # Cell agreement: fraction of cells matching original
                    agreement = (preds == orig_preds).float().mean().item()
                    batch_agreements.append(agreement)

                agreement_acc[k][M].append(float(np.mean(batch_agreements)))

    # Compute means
    mean_agreement = [[None] * (N + 1) for _ in range(N + 1)]
    for k in range(1, N + 1):
        for M in range(1, k + 1):
            vals = agreement_acc[k][M]
            mean_agreement[k][M] = float(np.mean(vals)) if vals else None

    return {"N": N, "mean_agreement": mean_agreement, "timesteps": [int(t) for t in timesteps]}


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_heatmap(data, output_dir):
    N = data["N"]
    mat = np.full((N, N), np.nan)
    for k in range(1, N + 1):
        for M in range(1, k + 1):
            v = data["mean_agreement"][k][M]
            if v is not None:
                mat[M - 1, k - 1] = v

    fig, ax = plt.subplots(figsize=(max(8, N // 2), max(6, N // 2)))
    im = ax.imshow(mat, origin="lower", vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xlabel("Denoising step k (at which noise-back is applied)")
    ax.set_ylabel("M (steps added back)")
    ax.set_title("Cell agreement with original denoising\n"
                 "(higher = model committed to same solution after re-noising)")
    fig.colorbar(im, ax=ax, label="Cell agreement")
    fig.tight_layout()
    path = os.path.join(output_dir, "noise_back_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_curves(data, output_dir):
    N = data["N"]
    M_values = [m for m in [1, 2, 3, 5, 8, 10] if m <= N]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(M_values)))
    for M, color in zip(M_values, colors):
        ks, vals = [], []
        for k in range(M, N + 1):
            v = data["mean_agreement"][k][M]
            if v is not None:
                ks.append(k)
                vals.append(v)
        if ks:
            ax.plot(ks, vals, "o-", label=f"M={M}", color=color, linewidth=1.5, markersize=4)

    ax.set_xlabel("Denoising step k")
    ax.set_ylabel("Cell agreement with original")
    ax.set_title("Denoising commitment: agreement after noise-back by M steps")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(output_dir, "noise_back_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.get("output_dir", "results/noise_back")
    os.makedirs(output_dir, exist_ok=True)

    nb_cfg = cfg.get("noise_back", {})
    seed = nb_cfg.get("seed", 42)
    num_ddim_steps = nb_cfg.get("num_ddim_steps", 20)
    batch_size = nb_cfg.get("batch_size", 16)
    num_workers = cfg.data.get("num_workers", 4)
    classifier_path = nb_cfg.get("classifier_path", "runs/mnist_classifier_cell16.pt")
    cell_size = int(cfg.data.cell_size)

    torch.manual_seed(seed)

    logger.info(OmegaConf.to_yaml(cfg))

    # ── Dataset ───────────────────────────────────────────────────────────────
    sudoku_dir = cfg.data.sudoku_dir
    test_dir = os.path.join(sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(sudoku_dir, "train")
    ds = MNISTSudokuDataset(
        sudoku_dir=eval_dir,
        mnist_root=cfg.data.mnist_root,
        cell_size=cell_size,
        mnist_split="test",
        mask_given=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # ── Classifier ────────────────────────────────────────────────────────────
    classifier = load_or_train_classifier(
        classifier_path,
        mnist_root=cfg.data.mnist_root,
        cell_size=cell_size,
        device=device,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    scheduler = build_scheduler(cfg)
    model = _load_model(cfg, scheduler, device)

    # ── DDIM scheduler ────────────────────────────────────────────────────────
    ddim = DDIMScheduler(
        num_train_timesteps=int(cfg.diffusion.num_train_timesteps),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        prediction_type=str(cfg.diffusion.prediction_type),
    )
    ddim.set_timesteps(num_ddim_steps)
    timesteps = ddim.timesteps.tolist()
    logger.info(f"DDIM: {len(timesteps)} steps, t=[{timesteps[0]}..{timesteps[-1]}]")

    # ── Experiment ────────────────────────────────────────────────────────────
    results = run_experiment(model, loader, ddim, timesteps, classifier, nb_cfg, cell_size, device)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_heatmap(results, output_dir)
    plot_curves(results, output_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "noise_back_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()