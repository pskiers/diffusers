from __future__ import annotations

import torch
from configs.schemas import NoisySwapConfig


def fix_swap_target(images, noisy, noise, swap, timesteps, scheduler):
    """For eps prediction, correct the target for swapped samples.

    After noisy[swap] = alt_noisy, the target must be the virtual noise that
    would denoise alt_noisy toward x_clean_initial (images[swap]):
        eps* = (x_noisy_swapped - sqrt(alpha_bar_t) * x_clean_initial) / sqrt(1 - alpha_bar_t)
    For x0 prediction no fix is needed: target = images is already x_clean_initial.
    """
    if swap.numel() == 0 or scheduler.config.prediction_type != "epsilon":
        return noise
    alpha_bar = scheduler.alphas_cumprod.to(noisy.device)[timesteps[swap]]
    sqrt_ab = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1_ab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    target = noise.clone()
    target[swap] = (noisy[swap] - sqrt_ab * images[swap]) / sqrt_1_ab
    return target


def apply_noisy_swap(
    images: torch.Tensor,
    noisy: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    swap_cfg: NoisySwapConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For eligible samples (timestep in range, random draw < prob), replace
    x_noisy with a *different* clean image from the batch noised to the same
    timestep. Returns a (possibly modified) noisy tensor and target
    """
    if swap_cfg is None:
        return noisy, target
    if swap_cfg.prob <= 0.0:
        return noisy, target

    B = images.shape[0]
    eligible = (timesteps >= swap_cfg.t_min) & (timesteps <= swap_cfg.t_max)
    draw = torch.rand(B, device=images.device) < swap_cfg.prob
    swap = (eligible & draw).nonzero(as_tuple=True)[0]  # indices to swap

    if swap.numel() == 0:
        return noisy, target

    noisy = noisy.clone()
    # Pick a different image for each swap candidate (cyclic shift avoids self-swap)
    src_idx = (swap + 1) % B
    alt_images = images[src_idx]  # different clean images
    alt_noisy = scheduler.add_noise(
        alt_images,
        torch.randn_like(alt_images),
        timesteps[swap],  # exact same timestep
    )
    noisy[swap] = alt_noisy

    # fix targets
    target = fix_swap_target(images, noisy, target, swap, timesteps, scheduler)
    return noisy, target


def x0_from_noise_pred(noise_pred, noisy, timesteps, scheduler):
    """Differentiably recover x0_pred from model output.

    Supports epsilon and x0 prediction types.  Result is clamped to [0, 1]
    (the image range used in this codebase) so it can be fed to the classifier.
    """
    pt = scheduler.config.prediction_type
    if pt == "epsilon":
        alpha_bar = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
        sqrt_ab = alpha_bar.sqrt().view(-1, 1, 1, 1)
        sqrt_1_ab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
        x0 = (noisy - sqrt_1_ab * noise_pred.float()) / sqrt_ab
    elif pt == "sample":
        x0 = noise_pred.float()
    else:
        raise ValueError(f"Unsupported prediction_type for classifier loss: {pt}")
    return x0.clamp(0.0, 1.0)


def ddim_prev_sample(x0_pred: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor, scheduler) -> torch.Tensor:
    """Compute x_{t-1} from x0_pred using the deterministic DDIM formula.

    Differentiable w.r.t. x0_pred — suitable as input to a classifier loss.
    """
    device = noisy.device
    ab_t = scheduler.alphas_cumprod.to(device)[timesteps]
    ab_tm1 = scheduler.alphas_cumprod.to(device)[(timesteps - 1).clamp(min=0)]
    sqrt_ab_t = ab_t.sqrt().view(-1, 1, 1, 1)
    sqrt_1ab_t = (1.0 - ab_t).sqrt().view(-1, 1, 1, 1)
    sqrt_ab_tm1 = ab_tm1.sqrt().view(-1, 1, 1, 1)
    sqrt_1ab_tm1 = (1.0 - ab_tm1).sqrt().view(-1, 1, 1, 1)
    eps = (noisy - sqrt_ab_t * x0_pred) / sqrt_1ab_t
    return (sqrt_ab_tm1 * x0_pred + sqrt_1ab_tm1 * eps).clamp(0.0, 1.0)
