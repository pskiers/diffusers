"""
Analyze early stopping in TRM diffusion models.

Experiment 1 – Convergence heatmap
  Runs a full denoising pass on a batch of images.
  At every timestep t and every n_sup step n ∈ {2,...,n_sup}, logs:
    • step-to-step distance  Δ(n)   = √((1-ᾱₜ)/ᾱₜ) ‖ε(n) − ε(n-1)‖₂  (mean over batch)
    • distance to final      Δfin(n) = √((1-ᾱₜ)/ᾱₜ) ‖ε(n) − ε(n_sup)‖₂
  Produces:
    • convergence_scatter.png  – scatter(t, Δ(n)), coloured by Δfin(n), one panel per n
    • convergence_mean.png     – line plot: mean Δ(n) per n vs t

Experiment 2 – Visual comparison of early stopping thresholds
  Generates images from fixed initial noises with different stopping thresholds.
  Produces:
    • image_grid.png                – rows = thresholds, columns = seeds
    • inference_counts_per_seed.png – n_sup iterations used per timestep, per seed
    • inference_counts_mean.png     – mean n_sup iterations used per timestep

Usage (Hydra CLI, same pattern as sample.py):
    python analyze_early_stopping.py experiment=cond_cifar100_trm \\
        analysis.num_images=64 \\
        analysis.num_seeds=6 \\
        analysis.thresholds=[2.0,1.0,0.5,0.1] \\
        analysis.output_dir=early_stopping_analysis \\
        analysis.run_exp1=true \\
        analysis.run_exp2=true

Threshold semantics:
    null / not given  →  no early stopping (run all n_sup steps)
    large value       →  stop aggressively after step 2
    small value       →  conservative stopping (close to running all steps)
    0.0               →  never stop early (same as null)
"""

import os
import sys
import logging
import math

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from hydra.utils import instantiate
from safetensors.torch import load_file

from model_utils import load_with_backward_compatibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model & scheduler loading  (mirrors sample.py)
# ---------------------------------------------------------------------------

def load_model(args, device, weight_dtype):
    if args.get("checkpoint_step") is not None:
        if str(args.checkpoint_step).lower() == "latest":
            dirs = [
                d for d in os.listdir(args.output_dir)
                if d.startswith("checkpoint-") and d.split("-")[1].isdigit()
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if not dirs:
                raise ValueError(f"No checkpoints found in {args.output_dir}")
            resolved = os.path.join(args.output_dir, dirs[-1])
        else:
            resolved = os.path.join(args.output_dir, f"checkpoint-{args.checkpoint_step}")
    elif args.get("checkpoint_path") is not None:
        resolved = args.checkpoint_path
    else:
        raise ValueError("Provide 'checkpoint_step' or 'checkpoint_path'.")

    logger.info(f"Loading weights from {resolved}")
    unet = instantiate(args.model, _convert_="all")

    if hasattr(unet, "load"):
        unet.load(resolved)

    unet_dir = os.path.join(resolved, "unet_ema" if args.get("use_ema", False) else "unet")
    sf = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bn = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
    if os.path.exists(sf):
        raw_sd = load_file(sf)
    elif os.path.exists(bn):
        raw_sd = torch.load(bn, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights in {unet_dir}")

    target = unet.core_model if hasattr(unet, "core_model") else unet
    load_with_backward_compatibility(target, raw_sd, logger)
    unet.eval()

    if hasattr(unet, "core_model"):
        unet.core_model.to(device, dtype=weight_dtype)
    else:
        unet.to(device, dtype=weight_dtype)

    if hasattr(unet, "get_trainable_modules"):
        for m in unet.get_trainable_modules().values():
            if isinstance(m, torch.nn.Module):
                m.to(device, dtype=weight_dtype)

    return unet


def load_scheduler(args):
    Cls = DDIMScheduler if args.use_ddim else DDPMScheduler
    kw = {"num_train_timesteps": args.ddpm_num_steps, "beta_schedule": args.ddpm_beta_schedule}
    if "prediction_type" in Cls.__init__.__code__.co_varnames:
        kw["prediction_type"] = args.prediction_type
    sched = Cls(**kw)
    sched.set_timesteps(args.ddpm_num_inference_steps)
    return sched


# ---------------------------------------------------------------------------
# Condition building  (mirrors generate_image_batch)
# ---------------------------------------------------------------------------

def build_conditions(args, bsz, device, generator):
    """Returns (conds, masks, unconds, do_cfg)."""
    model_config = getattr(args, "model", {})
    if hasattr(model_config, "thinker_model"):
        target_cfg = model_config.thinker_model
    else:
        target_cfg = getattr(model_config, "core_model", model_config)

    cond_mode = getattr(target_cfg, "condition_mode", None)
    is_class = cond_mode in ["class", "class_adaln"]
    is_sequence = cond_mode == "sequence"
    target_str = str(getattr(target_cfg, "_target_", ""))
    is_standard = (
        ("UNet2DModel" in target_str or "UNet2DConditionModel" in target_str)
        and getattr(args.dataset, "num_classes", None)
    )

    conds = masks = unconds = None
    do_cfg = False

    if is_class or is_standard:
        conds = torch.randint(0, args.dataset.num_classes, [bsz], generator=generator, device=device)
        if args.guidance_scale > 1.0:
            unconds = torch.full_like(conds, args.dataset.num_classes)
            do_cfg = True

    elif is_sequence:
        from clevr_dataset import sample_random_scene, make_tensor_from_scene
        cs, ms = [], []
        for _ in range(bsz):
            scene = sample_random_scene(num_objects=None, mode=args.dataset.dataset_mode)
            c, m = make_tensor_from_scene(scene)
            cs.append(c)
            ms.append(m)
        conds = torch.cat(cs, dim=0).to(device)
        masks = torch.cat(ms, dim=0).to(device)
        # CFG not supported for sequence conditioning in this script

    return conds, masks, unconds, do_cfg


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def get_alpha_bar(scheduler, t_scalar):
    """ᾱₜ as a scalar float32 tensor (CPU)."""
    return scheduler.alphas_cumprod[int(t_scalar)].float()


def x0_distance(eps1, eps2, alpha_bar):
    """
    Per-sample x0-space L2 distance:
        ‖x̂₀(n) − x̂₀(n−1)‖₂ = √((1−ᾱ)/ᾱ) · ‖ε(n) − ε(n−1)‖₂

    Returns [B] tensor.
    """
    ab = alpha_bar.to(eps1.device).float()
    scale = torch.sqrt((1.0 - ab) / ab.clamp(min=1e-8))
    diff = eps1.float() - eps2.float()
    return scale * diff.view(diff.shape[0], -1).norm(dim=-1)


# ---------------------------------------------------------------------------
# Per-step n_sup prediction collector
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_nsup_predictions(unet, latent_input, t, class_input, mask_input, do_cfg, guidance_scale):
    """
    Run all n_sup iterations for one denoising step.

    `latent_input` is already scaled and CFG-doubled if needed.
    Returns list of n_sup tensors, each [B_effective, C, H, W], float32, post-CFG.
    """
    bsz_total = latent_input.shape[0]
    y, z = unet.get_initial_states(bsz_total)

    preds = []
    for _ in range(unet.n_sup):
        raw, y, z = unet.reasoning_step(latent_input, y, z, t, class_input, mask_input)
        raw = raw.float()
        if do_cfg:
            cond_p, uncond_p = raw.chunk(2)
            merged = uncond_p + guidance_scale * (cond_p - uncond_p)
        else:
            merged = raw
        preds.append(merged.detach())

    return preds


# ---------------------------------------------------------------------------
# Experiment 1: convergence data collection
# ---------------------------------------------------------------------------

def run_convergence_experiment(unet, scheduler, vae, vae_sf, args, device, weight_dtype, out_dir, num_images):
    if not hasattr(unet, "reasoning_step"):
        raise RuntimeError("Model has no reasoning_step – cannot collect per-n_sup predictions.")
    if unet.n_sup < 2:
        raise RuntimeError("n_sup must be ≥ 2 to measure step-to-step distances.")

    sample_size = args.dataset.resolution if vae is None else args.dataset.resolution // 8
    gen = torch.Generator(device=device).manual_seed(0)

    latents = torch.randn(
        (num_images, args.dataset.input_channels, sample_size, sample_size),
        generator=gen, device=device, dtype=torch.float32,
    )
    conds, masks, unconds, do_cfg = build_conditions(args, num_images, device, gen)

    if do_cfg:
        class_input = torch.cat([conds, unconds])
        mask_input = torch.cat([masks, masks]) if masks is not None else None
    else:
        class_input = conds
        mask_input = masks

    n_sup = unet.n_sup
    data_t, data_n, data_delta, data_to_final = [], [], [], []

    for t in tqdm(scheduler.timesteps, desc="Exp.1 Denoising"):
        t_idx = t.long().cpu().item()
        alpha_bar = get_alpha_bar(scheduler, t_idx)

        scaled = scheduler.scale_model_input(latents, t)
        if do_cfg:
            latent_input = torch.cat([scaled] * 2).to(weight_dtype)
        else:
            latent_input = scaled.to(weight_dtype)

        preds = collect_nsup_predictions(unet, latent_input, t, class_input, mask_input, do_cfg, args.guidance_scale)
        eps_final = preds[-1]  # n_sup-th prediction

        for n_idx in range(1, n_sup):   # n_idx=1 → n=2, n_idx=2 → n=3, ...
            delta = x0_distance(preds[n_idx], preds[n_idx - 1], alpha_bar).mean().item()
            if n_idx == n_sup - 1:
                to_final = 0.0
            else:
                to_final = x0_distance(preds[n_idx], eps_final, alpha_bar).mean().item()

            data_t.append(t_idx)
            data_n.append(n_idx + 1)   # 1-indexed (2, 3, ..., n_sup)
            data_delta.append(delta)
            data_to_final.append(to_final)

        # Denoising step uses the final prediction
        latents = scheduler.step(preds[-1], t, latents).prev_sample

    data = {
        "t": np.array(data_t),
        "n": np.array(data_n),
        "delta": np.array(data_delta),
        "to_final": np.array(data_to_final),
    }
    np.savez(os.path.join(out_dir, "convergence_data.npz"), **data)
    logger.info(f"Convergence data saved.")

    _plot_convergence_scatter(data, n_sup, out_dir)
    _plot_convergence_mean(data, n_sup, out_dir)
    return data


def _plot_convergence_scatter(data, n_sup, out_dir):
    """Scatter: t on X, Δ(n) on Y, coloured by Δfin(n). One panel per n."""
    n_values = sorted(np.unique(data["n"]))
    ncols = len(n_values)

    # Global colour scale across all panels
    vmin = min(data["to_final"].min(), 1e-9)
    vmax = max(data["to_final"].max(), vmin + 1e-9)

    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), squeeze=False)
    axes = axes[0]

    for ax, n_val in zip(axes, n_values):
        mask = data["n"] == n_val
        t_vals  = data["t"][mask]
        delta   = data["delta"][mask]
        to_fin  = data["to_final"][mask]

        sc = ax.scatter(
            t_vals, delta,
            c=to_fin, cmap="plasma",
            vmin=vmin, vmax=vmax,
            s=8, alpha=0.6, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label="Distance to final ε(n_sup)")
        ax.set_xlabel("Timestep t")
        ax.set_ylabel(f"Δ(n={n_val}, n-1)  [x0 norm]")
        ax.set_title(f"Step {n_val} vs {n_val - 1}")
        # Log-scale Y if the range spans more than one decade
        pos = delta[delta > 0]
        if pos.size and pos.max() > pos.min() * 10:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    fig.suptitle("TRM Convergence: Step-to-Step Distance vs Timestep", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "convergence_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {path}")


def _plot_convergence_mean(data, n_sup, out_dir):
    """Line plot: mean step-to-step distance vs t, one line per n."""
    n_values = sorted(np.unique(data["n"]))
    # Bin by timestep (each t might appear once after aggregation)
    t_unique = np.unique(data["t"])

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(n_values)))

    for n_val, col in zip(n_values, colors):
        mask = data["n"] == n_val
        t_vals = data["t"][mask]
        deltas = data["delta"][mask]
        # Sort by t
        order = np.argsort(t_vals)
        ax.plot(t_vals[order], deltas[order], label=f"n={n_val}", color=col, alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Mean Δ(n, n-1)  [x0 norm]")
    ax.set_title("Step-to-Step Distance vs Timestep (mean over batch)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    pos = data["delta"][data["delta"] > 0]
    if pos.size and pos.max() > pos.min() * 10:
        ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(out_dir, "convergence_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {path}")


# ---------------------------------------------------------------------------
# Experiment 2: early stopping visual comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def _denoise_batch_with_threshold(unet, scheduler, vae, vae_sf, args, device, weight_dtype,
                                   latents_batch, batch_class_input, batch_mask_input,
                                   do_cfg, threshold, vae_chunk_size=1):
    """
    Denoise a batch of seeds simultaneously with per-sample early stopping.

    Per-sample semantics: sample i stops (and freezes its output) as soon as
        √((1−ᾱₜ)/ᾱₜ) · ‖ε_i(n) − ε_i(n−1)‖₂ < threshold

    Args:
        latents_batch:      [B, C, H, W]  – one row per seed
        batch_class_input:  [B, ...]  or  [2B, ...]  when do_cfg=True
                            (blocked: all cond first, then all uncond)
        batch_mask_input:   [B, seq] or None
        vae_chunk_size:     how many latents to decode at once (keep small to avoid OOM)

    Returns:
        images      – [B, 3, H, W] float32 in [0, 1], on CPU
        inf_counts  – np.int32 [num_steps, B]  – n_sup steps used per (step, seed)
    """
    B = latents_batch.shape[0]
    latents = latents_batch.clone().to(device).float()
    do_early_stop = threshold is not None and threshold > 0.0
    all_inf_counts = []  # list of [B] arrays, one per scheduler step

    for t in scheduler.timesteps:
        t_idx = t.long().cpu().item()
        alpha_bar = get_alpha_bar(scheduler, t_idx).to(device)

        scaled = scheduler.scale_model_input(latents, t)
        if do_cfg:
            latent_input = torch.cat([scaled, scaled]).to(weight_dtype)  # [2B, C, H, W]
        else:
            latent_input = scaled.to(weight_dtype)

        bsz_total = latent_input.shape[0]
        y, z = unet.get_initial_states(bsz_total)

        # Per-sample tracking
        final_output = None                                          # [B, C, H, W]
        prev_output  = None                                          # [B, C, H, W]
        converged    = torch.zeros(B, dtype=torch.bool, device=device)
        n_taken      = torch.zeros(B, dtype=torch.long)             # CPU

        for i in range(unet.n_sup):
            raw, y, z = unet.reasoning_step(latent_input, y, z, t, batch_class_input, batch_mask_input)
            raw = raw.float()
            if do_cfg:
                cond_p, uncond_p = raw.chunk(2)                     # each [B, C, H, W]
                merged = uncond_p + args.guidance_scale * (cond_p - uncond_p)
            else:
                merged = raw                                         # [B, C, H, W]

            not_done = ~converged                                    # [B]

            # Update n_taken and frozen output for every sample still running
            n_taken[not_done.cpu()] = i + 1
            if final_output is None:
                final_output = merged.clone()
            else:
                mask4d = not_done.view(B, 1, 1, 1).expand_as(merged)
                final_output = torch.where(mask4d, merged, final_output)

            # Per-sample convergence check (needs at least two steps)
            if do_early_stop and prev_output is not None:
                ab    = alpha_bar.float()
                scale = torch.sqrt((1.0 - ab) / ab.clamp(min=1e-8))
                diff  = (merged - prev_output).view(B, -1).norm(dim=-1)  # [B]
                newly = (scale * diff < threshold) & not_done
                converged |= newly
                if converged.all():
                    break

            prev_output = merged.detach()

        all_inf_counts.append(n_taken.numpy())
        latents = scheduler.step(final_output, t, latents).prev_sample

    # VAE decode in small chunks to avoid OOM
    if vae is not None:
        chunks = []
        for start in range(0, B, vae_chunk_size):
            chunk = (latents[start:start + vae_chunk_size] / vae_sf).to(vae.dtype)
            chunks.append(vae.decode(chunk).sample.float())
        images = (torch.cat(chunks, dim=0) / 2 + 0.5).clamp(0, 1)
    else:
        images = (latents / 2 + 0.5).clamp(0, 1).float()

    return images.cpu(), np.stack(all_inf_counts, axis=0)  # [num_steps, B]


def run_visual_experiment(unet, scheduler, vae, vae_sf, args, device, weight_dtype, out_dir, num_seeds, thresholds):
    if not hasattr(unet, "reasoning_step"):
        raise RuntimeError("Model has no reasoning_step – cannot run early stopping.")

    sample_size = args.dataset.resolution if vae is None else args.dataset.resolution // 8

    # ── Build one fixed latent + condition per seed ──────────────────────────
    seed_latents  = []
    seed_conds    = []   # raw cond tensor per seed  [1, ...]
    seed_unconds  = []   # raw uncond tensor per seed [1, ...] or None
    seed_masks    = []   # mask per seed              [1, seq] or None
    do_cfg = False

    for seed in range(num_seeds):
        gen = torch.Generator(device=device).manual_seed(seed)
        lat = torch.randn(
            (1, args.dataset.input_channels, sample_size, sample_size),
            generator=gen, device=device, dtype=torch.float32,
        )
        conds, masks, unconds, _do_cfg = build_conditions(args, 1, device, gen)
        seed_latents.append(lat)
        seed_conds.append(conds)
        seed_masks.append(masks)
        seed_unconds.append(unconds)
        do_cfg = _do_cfg

    # ── Batch seeds into single tensors ──────────────────────────────────────
    latents_batch = torch.cat(seed_latents, dim=0)           # [B, C, H, W]

    if do_cfg:
        # Blocked layout: [cond_0, ..., cond_B, uncond_0, ..., uncond_B]
        # so that raw.chunk(2) gives [B, ...] cond and [B, ...] uncond correctly.
        all_conds   = torch.cat(seed_conds,   dim=0)         # [B, ...]
        all_unconds = torch.cat(seed_unconds, dim=0)         # [B, ...]
        batch_class_input = torch.cat([all_conds, all_unconds], dim=0)  # [2B, ...]
        batch_mask_input  = None
    else:
        batch_class_input = torch.cat(seed_conds, dim=0)     # [B, ...]
        batch_mask_input  = (
            torch.cat(seed_masks, dim=0) if any(m is not None for m in seed_masks) else None
        )

    # ── Labels ───────────────────────────────────────────────────────────────
    def _label(thresh):
        if thresh is None or thresh <= 0.0:
            return f"No stop  (all {unet.n_sup} steps)"
        return f"threshold = {thresh:.3g}"

    thresh_labels = [_label(th) for th in thresholds]

    # ── One denoising run per threshold (all seeds batched) ──────────────────
    images_grid = []   # [thresh_idx][seed_idx] → PIL image
    counts_grid = []   # [thresh_idx][seed_idx] → np.int32 [num_steps]

    for thresh, label in zip(thresholds, thresh_labels):
        logger.info(f"  Running threshold={thresh!r}  ({label})")

        imgs, inf_counts = _denoise_batch_with_threshold(
            unet, scheduler, vae, vae_sf, args, device, weight_dtype,
            latents_batch, batch_class_input, batch_mask_input,
            do_cfg, thresh,
        )
        # imgs: [B, C, H, W]   inf_counts: [num_steps, B]

        row_imgs   = []
        row_counts = []
        for s_idx in range(num_seeds):
            arr = (imgs[s_idx].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            row_imgs.append(Image.fromarray(arr))
            row_counts.append(inf_counts[:, s_idx])          # [num_steps]

        images_grid.append(row_imgs)
        counts_grid.append(row_counts)

    timesteps_np = scheduler.timesteps.cpu().numpy()
    _plot_image_grid(images_grid, thresh_labels, num_seeds, out_dir)
    _plot_inference_counts(counts_grid, thresh_labels, num_seeds, timesteps_np, out_dir)
    logger.info("Visual experiment done.")


def _plot_image_grid(images_grid, thresh_labels, num_seeds, out_dir):
    n_rows = len(thresh_labels)
    n_cols = num_seeds

    # Determine a good display size
    ref_w, ref_h = images_grid[0][0].size
    scale = max(1, 96 // min(ref_w, ref_h))  # Upscale tiny images for legibility
    cell_w, cell_h = ref_w * scale, ref_h * scale

    # Extra width for row labels
    label_px = 220
    fig_w = (n_cols * cell_w + label_px) / 100
    fig_h = (n_rows * cell_h + 40) / 100

    fig = plt.figure(figsize=(max(fig_w, 4), max(fig_h, 3)))

    # Use gridspec: narrow first column for labels, rest for images
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[label_px / cell_w] + [1] * n_cols,
        hspace=0.05, wspace=0.05,
        figure=fig,
    )

    for r_idx, (row_imgs, label) in enumerate(zip(images_grid, thresh_labels)):
        # Label cell
        ax_label = fig.add_subplot(gs[r_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.98, 0.5, label,
            ha="right", va="center", fontsize=8,
            transform=ax_label.transAxes,
        )
        # Image cells
        for c_idx, img in enumerate(row_imgs):
            ax = fig.add_subplot(gs[r_idx, c_idx + 1])
            if scale > 1:
                img = img.resize((cell_w, cell_h), Image.NEAREST)
            ax.imshow(img)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"Seed {c_idx}", fontsize=9, pad=2)

    fig.suptitle("Early Stopping: Same Noise, Different Thresholds", fontsize=12, y=1.01)
    path = os.path.join(out_dir, "image_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {path}")


def _plot_inference_counts(counts_grid, thresh_labels, num_seeds, timesteps_np, out_dir):
    n_thresh = len(thresh_labels)
    colors = plt.cm.plasma(np.linspace(0.05, 0.90, n_thresh))

    # ── Per-seed subplots ──
    fig, axes = plt.subplots(1, num_seeds, figsize=(4.5 * num_seeds, 3.5), squeeze=False)
    axes = axes[0]

    for s_idx in range(num_seeds):
        ax = axes[s_idx]
        for t_idx, (label, col) in enumerate(zip(thresh_labels, colors)):
            ax.plot(timesteps_np, counts_grid[t_idx][s_idx], color=col, alpha=0.85,
                    linewidth=1.4, label=label)
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("n_sup steps used")
        ax.set_title(f"Seed {s_idx}", fontsize=9)
        ax.set_ylim(0.5, max(counts_grid[0][0]) + 0.8)
        ax.grid(True, alpha=0.25)
        if s_idx == num_seeds - 1:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("n_sup Iterations per Diffusion Step", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "inference_counts_per_seed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {path}")

    # ── Mean over seeds ──
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for t_idx, (label, col) in enumerate(zip(thresh_labels, colors)):
        mean_c = np.mean([counts_grid[t_idx][s] for s in range(num_seeds)], axis=0)
        std_c  = np.std( [counts_grid[t_idx][s] for s in range(num_seeds)], axis=0)
        ax2.plot(timesteps_np, mean_c, color=col, linewidth=2, label=label)
        ax2.fill_between(timesteps_np, mean_c - std_c, mean_c + std_c, color=col, alpha=0.15)

    ax2.set_xlabel("Timestep t")
    ax2.set_ylabel("Mean n_sup steps used")
    ax2.set_title("Mean Inference Count per Timestep (± 1 std over seeds)")
    ax2.set_ylim(0.5, max(counts_grid[0][0]) + 0.8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "inference_counts_mean.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  → {path2}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # --- Analysis-specific config (all optional) ---
    acfg = args.get("analysis", OmegaConf.create({}))
    num_images   = int(acfg.get("num_images", 64))
    num_seeds    = int(acfg.get("num_seeds", 6))
    run_exp1     = bool(acfg.get("run_exp1", True))
    run_exp2     = bool(acfg.get("run_exp2", True))
    out_dir      = str(acfg.get("output_dir", "early_stopping_analysis"))

    # Thresholds for experiment 2.
    # In YAML/CLI use a list of floats; 'null' or 0.0 means no early stopping.
    raw_thresh = acfg.get("thresholds", None)
    if raw_thresh is None:
        # Sensible defaults; update after seeing Experiment 1 results.
        thresholds = [None, 2.0, 1.0, 0.5, 0.1]
    else:
        thresholds = [None if (t is None or t == "null") else float(t) for t in raw_thresh]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(args.get("mixed_precision", "no"), torch.float32)

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Device: {device}  |  dtype: {weight_dtype}")

    unet = load_model(args, device, weight_dtype)
    logger.info(f"Model class: {type(unet).__name__}  |  n_sup={unet.n_sup}")

    vae, vae_sf = None, 1.0
    if args.dataset.get("vae_name"):
        vae = AutoencoderKL.from_pretrained(args.dataset.vae_name).to(device, dtype=torch.float32)
        vae.requires_grad_(False)
        vae.eval()
        vae_sf = vae.config.scaling_factor

    scheduler = load_scheduler(args)

    if run_exp1:
        logger.info("=" * 60)
        logger.info("Experiment 1: Convergence Analysis")
        logger.info(f"  num_images={num_images}  |  n_sup={unet.n_sup}  |  steps={len(scheduler.timesteps)}")
        logger.info("=" * 60)
        run_convergence_experiment(
            unet, scheduler, vae, vae_sf, args, device, weight_dtype, out_dir, num_images,
        )

    if run_exp2:
        logger.info("=" * 60)
        logger.info("Experiment 2: Visual Early Stopping Comparison")
        logger.info(f"  num_seeds={num_seeds}  |  thresholds={thresholds}")
        logger.info("=" * 60)
        run_visual_experiment(
            unet, scheduler, vae, vae_sf, args, device, weight_dtype, out_dir, num_seeds, thresholds,
        )

    logger.info(f"All done.  Results in: {out_dir}/")


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()
