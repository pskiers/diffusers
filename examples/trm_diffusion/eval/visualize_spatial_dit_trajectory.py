"""
visualize_spatial_dit_trajectory.py — Denoising trajectory visualisation for LatentSpatialDiT.

For each sampled puzzle the script runs the confidence-driven DDIM loop and,
at every step, saves a 3×4 panel figure containing:

  Row 0  │ puzzle (condition) │ x_noisy (decoded) │ x0_pred (decoded) │ true solution
  Row 1  │ T_field heatmap    │ T_field on x_noisy│ T_field on x0_pred│ ΔU (signed change)
  Row 2  │ U (uncertainty)    │ U on puzzle        │ U on x_noisy      │ U on x0_pred

Colorbars are added to the heatmap panels so you can read off actual timestep /
confidence values.  Results are saved as per-step PNGs plus an optional GIF.

Usage:
  python eval/visualize_spatial_dit_trajectory.py \\
    --checkpoint runs/spatial_dit_perlin/checkpoint_final.pt \\
    --vae_checkpoint runs/vae_4x/checkpoint_final.pt \\
    --attention_head_dim 64 \\
    --num_puzzles 4 --num_steps 30 --cfg_scale 2.0 \\
    --output_dir runs/spatial_dit_perlin/trajectories
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from models.spatial_dit import LatentSpatialDiT, SpatialLatentConfig, SpatialLatentOptimConfig
from models.spatial_diffusion_utils import compute_denoising_speed, ddim_step_spatial, ddim_step_spatial_c
from models.utility_models import strip_compiled_prefix


# ── Model loading ─────────────────────────────────────────────────────────────

def build_vae(args) -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=1, out_channels=1, latent_channels=4,
        down_block_types=["DownEncoderBlock2D"] * args.vae_blocks,
        up_block_types=["UpDecoderBlock2D"] * args.vae_blocks,
        block_out_channels=args.vae_block_out_channels,
        layers_per_block=2, norm_num_groups=32, act_fn="silu",
    )


def load_model(args, device):
    vae = build_vae(args)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    sf_path = os.path.join(os.path.dirname(args.vae_checkpoint), "scaling_factor.pt")
    scaling_factor = (
        torch.load(sf_path, map_location="cpu", weights_only=True)["scaling_factor"]
        if os.path.exists(sf_path) else 1.0
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    cell_size = args.cell_size
    model_cfg = SpatialLatentConfig(
        vae_checkpoint=args.vae_checkpoint,
        latent_channels=4,
        latent_size=args.latent_size,
        model_type="dit",
        patch_size=args.patch_size,
        n_heads=args.n_heads,
        attention_head_dim=args.attention_head_dim,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        t_freq_dim=args.t_freq_dim,
        dropout=0.0,
        vocab_size=11,
        cond_embed_dim=args.cond_embed_dim,
        f_spatial=args.f_spatial,
        tau_init=args.tau_init,
        tau_student=args.tau_student,
        n_octaves=1,
        p_refine_max=0.0,
        p_refine_warmup_steps=0,
        teacher_ema_rate=0.999,
        continuous_time=args.continuous_time,
        cell_size=cell_size,
        painter_size=cell_size * 9,
    )
    model = LatentSpatialDiT(
        model_cfg=model_cfg,
        optim_cfg=SpatialLatentOptimConfig(),
        scheduler=scheduler,
        vae=vae,
        scaling_factor=scaling_factor,
        eval_clf=None,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compiled_prefix(ckpt["model_state"]), strict=False)
    return model.to(device).eval()


# ── Trajectory collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_trajectory(model, puzzle_tokens, device, args):  # noqa: C901
    """
    Run the DDIM loop and collect a snapshot at every step.
    Returns list of dicts with decoded images and heatmaps.
    """
    T_max = model.scheduler.config.num_train_timesteps
    C = model.model_cfg.latent_channels
    lH = lW = model.model_cfg.latent_size
    dt = T_max / args.num_steps
    continuous = model.model_cfg.continuous_time

    ddim = DDIMScheduler(
        num_train_timesteps=T_max,
        beta_schedule=model.scheduler.config.beta_schedule,
        prediction_type=model.scheduler.config.prediction_type,
    )
    ddim.set_timesteps(args.num_steps)
    alphas_cumprod = ddim.alphas_cumprod.to(device)

    B = puzzle_tokens.shape[0]
    z = torch.randn(B, C, lH, lW, device=device)
    _dtype = torch.float32 if continuous else torch.long
    T_field = torch.full((B, 1, lH, lW), float(T_max - 1), device=device, dtype=_dtype)

    cfg_scale = args.cfg_scale
    null_tokens = torch.ones_like(puzzle_tokens) if cfg_scale > 1.0 else None

    snapshots = []
    prev_U = None

    for step_idx in range(args.num_steps):
        T_old = T_field.clone()

        x0_pred, log_var = model._run_model(z, T_field, puzzle_tokens, model.dit)

        if cfg_scale > 1.0:
            x0_uncond, _ = model._run_model(z, T_field, null_tokens, model.dit)
            x0_pred = x0_uncond + cfg_scale * (x0_pred - x0_uncond)

        U = log_var.sigmoid()   # (B, 1, lH, lW) uncertainty in (0,1)

        # Decode to pixel space
        x_noisy_px  = model._decode(z)            # (B, 1, H, W)
        x0_pred_px  = model._decode(x0_pred)      # (B, 1, H, W)

        # Confidence change (signed; first step has no prev → zeros)
        if prev_U is None:
            dU = torch.zeros_like(U)
        else:
            dU = U - prev_U                        # negative = more confident

        # DDIM step — use the same configurable speed formula as eval_step
        speed = compute_denoising_speed(
            U, dt,
            alpha=getattr(args, "guidance_alpha", 1.0),
            power=getattr(args, "guidance_power", 1.0),
            top_m=getattr(args, "guidance_top_m", None),
        )
        T_new = (T_field.float() - speed).clamp(0, T_max - 1)
        if not continuous:
            T_new = T_new.round().long()

        if continuous:
            z = ddim_step_spatial_c(z, x0_pred, T_old, T_new, T_max)
        else:
            z = ddim_step_spatial(z, x0_pred, T_old, T_new, alphas_cumprod)

        snapshots.append({
            "step":       step_idx,
            "t_mean":     T_field.float().mean().item(),
            "x_noisy":    x_noisy_px.cpu(),
            "x0_pred":    x0_pred_px.cpu(),
            "T_field":    T_field.float().cpu(),   # (B, 1, lH, lW)
            "U":          U.float().cpu(),          # (B, 1, lH, lW)
            "dU":         dU.float().cpu(),         # (B, 1, lH, lW)
        })

        prev_U = U
        T_field = T_new

        if T_field.max() < 0.5:
            break

    return snapshots


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _to_gray(t: torch.Tensor) -> np.ndarray:
    """(1, H, W) or (H, W) tensor → (H, W) float32 numpy [0,1]."""
    return t.squeeze().float().numpy().clip(0, 1)


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    """Grayscale (H, W) → (H, W, 3) RGB numpy [0,1]."""
    g = _to_gray(t)
    return np.stack([g, g, g], axis=-1)


def _upsample_map(m: torch.Tensor, H: int, W: int) -> np.ndarray:
    """(1, 1, lH, lW) → (H, W) numpy float32."""
    up = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    return up.squeeze().numpy()


def _apply_cmap(data: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    """data (H, W) → (H, W, 3) RGB float32 via colormap."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.get_cmap(cmap_name)
    return mapper(norm(data))[:, :, :3].astype(np.float32)


def _blend(img_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend grayscale (as RGB) with heatmap RGB."""
    return np.clip(alpha * img_rgb + (1 - alpha) * heat_rgb, 0, 1)


def _ax_img(ax, img, title="", cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else None, aspect="auto")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _ax_heat(fig, ax, data, cmap_name, vmin, vmax, title, cb_label):
    """Show heatmap with a small colorbar."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb  = _apply_cmap(data, cmap_name, vmin, vmax)
    ax.imshow(rgb, aspect="auto")
    # Proper colorbar via ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _ax_divheat(fig, ax, data, vabs, title, cb_label):
    """Show signed diverging heatmap (RdBu_r) with ±vabs range."""
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    mapper = cm.get_cmap("RdBu_r")
    rgb = mapper(norm(data))[:, :, :3].astype(np.float32)
    ax.imshow(rgb, aspect="auto")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ── Trajectory plot (T over steps) ───────────────────────────────────────────

def make_trajectory_plot(
    snapshots: list,
    T_max: int,
    lH: int, lW: int,
    puzzle_px: np.ndarray,
) -> plt.Figure:
    """
    Three-panel figure summarising how T_field evolves across denoising steps.

    Panel A — Sorted heatmap
      Rows   = patches (81 cells) sorted ascending by their mean T over the
               trajectory (slow-to-converge patches at top, fast ones at bottom).
      Columns = denoising step.
      Colour  = T value (plasma, 0–T_max).
      Immediately reveals whether certain cells always converge first and
      whether the convergence clusters spatially.

    Panel B — Percentile fan
      10th/25th/median/75th/90th percentile of T across all patches at each step.
      Shows the spread of convergence speeds without 81 overlapping lines.

    Panel C — Spatial snapshots
      The 9×9 T_field grid (one square per sudoku cell) at five key moments:
      step 0, 25 %, 50 %, 75 %, and the final step.  The puzzle image is shown
      alongside for reference.  Directly connects convergence patterns to the
      puzzle structure (given vs blank cells).
    """
    n_steps = len(snapshots)
    n_patches = lH * lW   # 81 for 9×9 patches at patch_size=4

    # Collect T_field per step as (n_steps, n_patches) array
    T_matrix = np.zeros((n_steps, n_patches), dtype=np.float32)
    for i, snap in enumerate(snapshots):
        # (1, 1, lH, lW) → (lH*lW,)
        T_matrix[i] = snap["T_field"][0, 0].reshape(-1).numpy()

    steps = np.arange(n_steps)

    # Sort patches by mean T (ascending = fastest convergers at bottom)
    patch_mean_T = T_matrix.mean(axis=0)       # (n_patches,)
    sort_order   = np.argsort(patch_mean_T)[::-1]   # slowest first → top
    T_sorted     = T_matrix[:, sort_order].T         # (n_patches, n_steps)

    # Percentiles
    pcts = [10, 25, 50, 75, 90]
    pct_vals = np.percentile(T_matrix, pcts, axis=1)   # (5, n_steps)

    # Snapshot steps
    snap_indices = sorted(set([
        0,
        n_steps // 4,
        n_steps // 2,
        3 * n_steps // 4,
        n_steps - 1,
    ]))
    n_snap = len(snap_indices)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    # GridSpec: left col = heatmap, centre = percentile fan, right = snapshots
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, n_snap + 2,
        figure=fig,
        left=0.05, right=0.97, top=0.92, bottom=0.08,
        wspace=0.35, hspace=0.45,
    )
    ax_heat = fig.add_subplot(gs[:, 0])           # spans both rows, col 0
    ax_fan  = fig.add_subplot(gs[:, 1])           # spans both rows, col 1
    # Snapshot panels: top row = T_field, bottom row = puzzle
    snap_axes_T = [fig.add_subplot(gs[0, 2 + k]) for k in range(n_snap)]
    snap_axes_P = [fig.add_subplot(gs[1, 2 + k]) for k in range(n_snap)]

    fig.suptitle("T_field trajectory across denoising steps", fontsize=11, fontweight="bold")

    # ── Panel A: sorted heatmap ───────────────────────────────────────────────
    im = ax_heat.imshow(
        T_sorted,
        aspect="auto",
        origin="upper",
        extent=[0, n_steps - 1, n_patches, 0],
        cmap="plasma",
        vmin=0, vmax=T_max,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.04)
    cb.set_label("T value", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax_heat.set_xlabel("Denoising step", fontsize=8)
    ax_heat.set_ylabel("Patch (sorted by mean T, slow→fast)", fontsize=7)
    ax_heat.set_title("A  Sorted patch trajectories", fontsize=9, fontweight="bold")
    ax_heat.tick_params(labelsize=6)

    # ── Panel B: percentile fan ───────────────────────────────────────────────
    colours = ["#d73027", "#fc8d59", "#4575b4", "#91bfdb", "#e0f3f8"]
    labels  = ["90th pct", "75th pct", "Median", "25th pct", "10th pct"]
    fills   = [
        (0, 1),   # fill between 10th & 25th → light blue
        (1, 2),   # fill between 25th & median → blue
        (2, 3),   # fill between median & 75th → red
        (3, 4),   # fill between 75th & 90th → dark red
    ]
    fill_colours = ["#91bfdb", "#4575b4", "#fc8d59", "#d73027"]
    for (lo, hi), fc in zip(fills, fill_colours):
        ax_fan.fill_between(steps, pct_vals[lo], pct_vals[hi],
                            alpha=0.3, color=fc)
    for k, (pv, lbl, col) in enumerate(zip(pct_vals, labels, colours)):
        lw = 2.0 if k == 2 else 1.0  # thicker median
        ax_fan.plot(steps, pv, color=col, lw=lw, label=lbl)
    ax_fan.set_xlim(0, n_steps - 1)
    ax_fan.set_ylim(0, T_max)
    ax_fan.set_xlabel("Denoising step", fontsize=8)
    ax_fan.set_ylabel("T value", fontsize=8)
    ax_fan.set_title("B  Percentile fan", fontsize=9, fontweight="bold")
    ax_fan.legend(fontsize=6, loc="upper right")
    ax_fan.tick_params(labelsize=6)
    ax_fan.grid(alpha=0.3)

    # ── Panel C: spatial snapshots ────────────────────────────────────────────
    cmap_T = cm.get_cmap("plasma")
    norm_T = Normalize(vmin=0, vmax=T_max)

    for col, si in enumerate(snap_indices):
        snap = snapshots[si]
        # T_field as 9×9 (one value per patch)
        T_grid = snap["T_field"][0, 0].numpy()  # (lH, lW) = (9, 9) typically
        T_rgb  = cmap_T(norm_T(T_grid))[:, :, :3]

        ax_t = snap_axes_T[col]
        ax_p = snap_axes_P[col]

        ax_t.imshow(T_rgb, aspect="equal", interpolation="nearest")
        ax_t.set_title(f"step {si}", fontsize=7)
        ax_t.axis("off")

        # Show puzzle image with T overlay
        puzzle_up = puzzle_px  # already (H, W)
        T_up = _upsample_map(snap["T_field"][0:1], puzzle_up.shape[0], puzzle_up.shape[1])
        T_norm_up = T_up / max(T_max - 1, 1)
        T_rgb_up = _apply_cmap(T_norm_up, "plasma", 0, 1)
        blended = _blend(np.stack([puzzle_up] * 3, axis=-1), T_rgb_up, alpha=0.6)
        ax_p.imshow(blended, aspect="equal")
        ax_p.axis("off")

    snap_axes_T[0].set_ylabel("T_field (9×9)", fontsize=7)
    snap_axes_P[0].set_ylabel("T on puzzle", fontsize=7)
    snap_axes_T[len(snap_indices) // 2].set_title(
        "C  Spatial snapshots at key steps  (top: T grid, bottom: T on puzzle)",
        fontsize=9, fontweight="bold",
        pad=14,
    )

    # Shared colorbar for panel C
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_T)
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=snap_axes_T + snap_axes_P,
                       fraction=0.02, pad=0.04, shrink=0.7)
    cb2.set_label("T value", fontsize=6)
    cb2.ax.tick_params(labelsize=5)

    return fig


# ── Per-step figure ────────────────────────────────────────────────────────────

def make_step_figure(
    snap: dict,
    puzzle_px: np.ndarray,    # (H, W) float [0,1]
    solution_px: np.ndarray,  # (H, W) float [0,1]
    T_max: int,
    b: int = 0,               # which sample in the batch
) -> plt.Figure:
    H, W = puzzle_px.shape

    x_noisy = _to_gray(snap["x_noisy"][b])           # (H, W)
    x0_pred  = _to_gray(snap["x0_pred"][b])           # (H, W)
    T_map    = _upsample_map(snap["T_field"][b:b+1], H, W)  # (H, W) [0, T_max]
    U_map    = _upsample_map(snap["U"][b:b+1], H, W)        # (H, W) [0, 1]
    dU_map   = _upsample_map(snap["dU"][b:b+1], H, W)       # (H, W) signed

    t_norm   = T_map / max(T_max - 1, 1)                    # [0,1] for plasma
    T_rgb    = _apply_cmap(t_norm, "plasma", 0, 1)
    U_rgb    = _apply_cmap(U_map,  "viridis", 0, 1)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    fig.suptitle(
        f"Step {snap['step']:03d}  |  mean T = {snap['t_mean']:.1f}",
        fontsize=9, fontweight="bold",
    )

    row0, row1, row2 = axes

    # ── Row 0: images ─────────────────────────────────────────────────────────
    _ax_img(row0[0], puzzle_px,   "Puzzle (condition)")
    _ax_img(row0[1], x_noisy,     "x_t  (noisy, decoded)")
    _ax_img(row0[2], x0_pred,     "x0_pred (decoded)")
    _ax_img(row0[3], solution_px, "Ground truth")

    # ── Row 1: T_field ────────────────────────────────────────────────────────
    _ax_heat(fig, row1[0], t_norm, "plasma", 0, 1,
             "T_field", f"timestep  (0 – {T_max})")
    # Manually fix the colorbar ticks to show actual T values
    row1[0].images[0].set_clim(0, 1)

    _ax_img(row1[1], _blend(_to_rgb(torch.tensor(x_noisy)), T_rgb),
            "T_field  ⊕  x_t")
    _ax_img(row1[2], _blend(_to_rgb(torch.tensor(x0_pred)), T_rgb),
            "T_field  ⊕  x0_pred")

    # Signed confidence change — blue = more confident, red = less confident
    dU_abs = np.abs(dU_map).max() or 1e-6
    _ax_divheat(fig, row1[3], dU_map, vabs=dU_abs,
                title="ΔU  (confidence change)",
                cb_label="ΔU  (−=more conf., +=less conf.)")

    # ── Row 2: uncertainty (U) ────────────────────────────────────────────────
    _ax_heat(fig, row2[0], U_map, "viridis", 0, 1,
             "U  (uncertainty)", "U   0=certain  1=uncertain")

    _ax_img(row2[1], _blend(_to_rgb(torch.tensor(puzzle_px)), U_rgb),
            "U  ⊕  puzzle")
    _ax_img(row2[2], _blend(_to_rgb(torch.tensor(x_noisy)), U_rgb),
            "U  ⊕  x_t")
    _ax_img(row2[3], _blend(_to_rgb(torch.tensor(x0_pred)), U_rgb),
            "U  ⊕  x0_pred")

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--vae_checkpoint", required=True)
    p.add_argument("--output_dir",     default="trajectories")

    # Data
    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_puzzles", type=int, default=4,
                   help="Number of puzzles to visualise.")

    # VAE architecture
    p.add_argument("--vae_blocks", type=int, default=3,
                   help="Number of encoder/decoder blocks in the VAE (default 3 → 4x compression)")
    p.add_argument("--vae_block_out_channels", type=int, nargs="+",
                   default=[32, 64, 128])

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample")

    # Sampling
    p.add_argument("--num_steps",    type=int,   default=30)
    p.add_argument("--cfg_scale",    type=float, default=2.0)
    p.add_argument("--continuous_time", action="store_true")

    # DiT architecture (must match checkpoint)
    p.add_argument("--latent_size",         type=int,   default=36)
    p.add_argument("--patch_size",          type=int,   default=4)
    p.add_argument("--n_heads",             type=int,   default=8)
    p.add_argument("--attention_head_dim",  type=int,   default=64)
    p.add_argument("--n_layers",            type=int,   default=6)
    p.add_argument("--mlp_ratio",           type=float, default=4.0)
    p.add_argument("--t_freq_dim",          type=int,   default=256)
    p.add_argument("--cond_embed_dim",      type=int,   default=256)
    p.add_argument("--f_spatial",           type=float, default=0.25)
    p.add_argument("--tau_init",            type=float, default=20.0)
    p.add_argument("--tau_student",         type=float, default=30.0)

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Confidence-driven denoising controls
    p.add_argument("--guidance_alpha", type=float, default=1.0,
                   help="Speed scale factor. 0=uniform DDIM, 1=default, >1=amplified.")
    p.add_argument("--guidance_power", type=float, default=1.0,
                   help="Exponent on confidence. >1 sharpens differences, <1 flattens.")
    p.add_argument("--guidance_top_m", type=float, default=None,
                   help="Only denoise top-m fraction of most-confident pixels (e.g. 0.5).")

    p.add_argument("--make_gif", action="store_true",
                   help="Assemble per-step PNGs into an animated GIF per puzzle.")
    p.add_argument("--blend_alpha", type=float, default=0.5,
                   help="Alpha for overlay blending (0=full heatmap, 1=full image).")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dir = os.path.join(args.sudoku_dir, "test")
    ds_dir   = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    ds = MNISTSudokuDataset(
        sudoku_dir=ds_dir, mnist_root=args.mnist_root,
        cell_size=args.cell_size, mnist_split="test", mask_given=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(args, device)
    T_max = model.scheduler.config.num_train_timesteps
    lH = lW = model.model_cfg.latent_size
    print(f"  T_max={T_max}  latent_size={lH}×{lW}  "
          f"scaling_factor={model.scaling_factor:.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    puzzle_count = 0
    for batch in loader:
        if puzzle_count >= args.num_puzzles:
            break

        puzzle_tokens = batch["puzzle_tokens"].to(device)    # (1, 81)
        conditions    = batch["conditions"]                   # (1, 1, H, W) CPU
        images        = batch["images"]                       # (1, 1, H, W) CPU — full solution

        puzzle_px   = conditions[0, 0].float().numpy().clip(0, 1)   # (H, W)
        solution_px = images[0, 0].float().numpy().clip(0, 1)        # (H, W)

        print(f"\nPuzzle {puzzle_count + 1}/{args.num_puzzles} …")
        snapshots = collect_trajectory(model, puzzle_tokens, device, args)
        print(f"  Collected {len(snapshots)} steps.")

        puzzle_dir = os.path.join(args.output_dir, f"puzzle_{puzzle_count:02d}")
        os.makedirs(puzzle_dir, exist_ok=True)

        frame_paths = []
        for snap in snapshots:
            fig = make_step_figure(snap, puzzle_px, solution_px, T_max, b=0)
            path = os.path.join(puzzle_dir, f"step_{snap['step']:03d}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(path)
            print(f"    step {snap['step']:3d}  mean_T={snap['t_mean']:6.2f}", end="\r")

        print(f"  Saved {len(frame_paths)} frames to {puzzle_dir}")

        # Trajectory plot (sorted heatmap + percentile fan + spatial snapshots)
        traj_fig = make_trajectory_plot(
            snapshots, T_max, lH, lW, puzzle_px
        )
        traj_path = os.path.join(args.output_dir, f"puzzle_{puzzle_count:02d}_trajectory.png")
        traj_fig.savefig(traj_path, dpi=120, bbox_inches="tight")
        plt.close(traj_fig)
        print(f"  Trajectory plot → {traj_path}")

        if args.make_gif:
            try:
                from PIL import Image as PILImage
                frames = [PILImage.open(p) for p in frame_paths]
                gif_path = os.path.join(args.output_dir, f"puzzle_{puzzle_count:02d}.gif")
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:],
                    duration=300, loop=0,
                )
                print(f"  GIF saved → {gif_path}")
            except ImportError:
                print("  (Pillow not installed — skipping GIF)")

        puzzle_count += 1

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
