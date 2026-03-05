import math
import torch
import re


def load_with_backward_compatibility(unet, state_dict, logger=None):
    """
    Translates old checkpoint keys to the new unified model architecture keys.
    """

    # Regex map: pattern -> replacement
    RENAME_MAP = {
        r"^module\.": "",
        r"^unet\.": "",
        r"(^|\.)projector(\.)": r"\g<1>condition_projector\g<2>",
        r"(^|\.)class_embedding(\.)": r"\g<1>condition_projector\g<2>",
    }

    adapted_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in RENAME_MAP.items():
            new_key = re.sub(pattern, replacement, new_key)
        adapted_dict[new_key] = value

    # strict=False prevents crashing if minor structural parameters differ
    missing, unexpected = unet.load_state_dict(adapted_dict, strict=False)

    if len(missing) > 0:
        if logger is not None:
            logger.warning(
                f"Missing keys when loading checkpoint (showing first 5): {missing[:5]} ... ({len(missing)} total)"
            )
        else:
            print(f"Missing keys when loading checkpoint (showing first 5): {missing[:5]} ... ({len(missing)} total)")
    if len(unexpected) > 0:
        if logger is not None:
            logger.warning(
                f"Unexpected keys in checkpoint (showing first 5): {unexpected[:5]} ... ({len(unexpected)} total)"
            )
            logger.warning("If these unexpected keys should map to the missing keys, add them to the RENAME_MAP!")
        else:
            print(f"Unexpected keys in checkpoint (showing first 5): {unexpected[:5]} ... ({len(unexpected)} total)")
            print("If these unexpected keys should map to the missing keys, add them to the RENAME_MAP!")


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
