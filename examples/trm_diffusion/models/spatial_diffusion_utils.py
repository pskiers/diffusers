"""
models/spatial_diffusion_utils.py — Utilities for spatially-varying diffusion.

Each image pixel carries its own timestep (stored in a T_field spatial map)
instead of the usual single scalar per sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_noise_field(
    B: int,
    H: int,
    W: int,
    f_spatial: float,
    device: torch.device,
    n_octaves: int = 1,
) -> torch.Tensor:
    """
    Generate a smooth spatial noise field in [0, 1].

    Blob size is controlled by f_spatial: the low-resolution grid is sampled at
    grid_h = max(2, round(H * f_spatial)), then bicubic-upsampled to (H, W).
    Lower f_spatial → fewer, larger blobs.  Higher → finer variation.

    For 36×36 latents with 9×9 sudoku cells (4px per cell):
      f_spatial = 1/9  → grid_h = 4  → blobs ≈ one cell wide
      f_spatial = 1/3  → grid_h = 12 → sub-cell variation

    n_octaves stacks frequencies f, 2f, 4f… with halving amplitudes for
    more natural-looking texture (1 octave is usually sufficient).

    Returns (B, 1, H, W) float32.
    """
    result = torch.zeros(B, 1, H, W, device=device)
    amplitude = 1.0
    total = 0.0
    for i in range(n_octaves):
        freq = f_spatial * (2**i)
        gh = max(2, round(H * freq))
        gw = max(2, round(W * freq))
        z = torch.rand(B, 1, gh, gw, device=device)
        z = F.interpolate(z, size=(H, W), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        result = result + amplitude * z
        total += amplitude
        amplitude *= 0.5
    return result / total


def make_t_field(
    t_base: torch.Tensor,  # (B,) float — per-sample base noise level
    tau: float,  # max deviation from base (in timestep units)
    perlin: torch.Tensor,  # (B, 1, H, W) in [0, 1]
    T_max: int,
) -> torch.Tensor:
    """
    T_field = clip(t_base + tau * perlin, 0, T_max-1)

    Returns (B, 1, H, W) long.
    """
    t = t_base[:, None, None, None].float() + tau * perlin
    return t.clamp(0, T_max - 1).long()


def add_noise_spatial(
    x0: torch.Tensor,  # (B, C, H, W)
    noise: torch.Tensor,  # (B, C, H, W)
    T_field: torch.Tensor,  # (B, 1, H, W) long in [0, T_max-1]
    alphas_cumprod: torch.Tensor,  # (T_max,) float, precomputed from scheduler
) -> torch.Tensor:
    """
    Forward diffusion x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    with per-pixel timestep.

    Returns (B, C, H, W).
    """
    alpha_bar = alphas_cumprod[T_field.expand_as(x0)]  # (B, C, H, W)
    return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise


def ddim_step_spatial(
    z: torch.Tensor,  # (B, C, H, W) current noisy latent
    x0_pred: torch.Tensor,  # (B, C, H, W) model's x0 prediction
    T_old: torch.Tensor,  # (B, 1, H, W) long — current timestep field
    T_new: torch.Tensor,  # (B, 1, H, W) long — next (lower) timestep field
    alphas_cumprod: torch.Tensor,  # (T_max,) float
) -> torch.Tensor:
    """
    Deterministic DDIM step with per-pixel timesteps.

    Recovers epsilon from (z, x0_pred, T_old), then computes new z at T_new.
    Returns (B, C, H, W).
    """
    ab_old = alphas_cumprod[T_old.expand_as(z)]
    ab_new = alphas_cumprod[T_new.expand_as(z)]
    eps = (z - ab_old.sqrt() * x0_pred) / (1.0 - ab_old).sqrt().clamp(min=1e-6)
    return ab_new.sqrt() * x0_pred + (1.0 - ab_new).sqrt() * eps


def gaussian_nll_loss(
    x0: torch.Tensor,  # (B, C, H, W) ground-truth clean latent
    x0_pred: torch.Tensor,  # (B, C, H, W) predicted clean latent
    log_var: torch.Tensor,  # (B, 1, H, W) predicted log-variance
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL:
        sum_i [ (x0_i - pred_i)^2 / (2 * exp(log_var_i)) + 0.5 * log_var_i ]

    The log term prevents the model from maximising uncertainty everywhere;
    the denominator stops over-penalising blurry but valid predictions.
    """
    var = log_var.exp().clamp(min=1e-6)
    return ((x0 - x0_pred).pow(2) / (2.0 * var) + 0.5 * log_var).mean()
