import torch


def get_model_output(model, x, timesteps, conditions=None, masks=None):
    """
    Helper function to cleanly route the forward pass
    based on whether we have conditions and/or masks.
    """
    kwargs = {}
    if conditions is not None:
        # Handle potential DDP/Accelerate wrappers gracefully
        base_model = model.module if hasattr(model, "module") else model

        # If it's our Unified model, use 'condition_tensors'
        if hasattr(base_model.config, "condition_mode"):
            kwargs["condition_tensors"] = conditions
        # Otherwise, standard diffusers UNet2DModel expects 'class_labels'
        else:
            kwargs["class_labels"] = conditions

    if masks is not None:
        kwargs["attention_mask"] = masks

    return model(x, timesteps, **kwargs).sample


def latent_recursion(model, x, y, z, timesteps, conditions=None, masks=None, n=6):
    """
    Executes 'n' microscopic iterations of the latent reasoning loop.
    """
    for _ in range(n):
        out = get_model_output(model, torch.cat([x, y, z], dim=1), timesteps, conditions, masks)
        _, z = out.chunk(2, dim=1)

    out = get_model_output(model, torch.cat([x, y, z], dim=1), timesteps, conditions, masks)
    y, _ = out.chunk(2, dim=1)

    return y, z


def deep_recursion(model, x, y, z, timesteps, conditions=None, masks=None, n=6, T=3):
    """
    Executes 'T' macroscopic steps of the latent reasoning loop.
    Returns the model output for the loss calculation, AND the safely detached
    y and z states for the next iteration of the supervision loop.
    """
    # 1. Unroll T-1 steps WITHOUT gradients
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(model, x, y, z, timesteps, conditions, masks, n)

    # 2. Do the final step WITH gradients enabled
    y_final, z_final = latent_recursion(model, x, y, z, timesteps, conditions, masks, n)

    # RETURN TUPLE:
    # y_final: Retains the .grad_fn graph for loss
    # y_final.detach(): Stripped of gradients.
    # z_final.detach(): Stripped of gradients.
    return y_final, y_final.detach(), z_final.detach()
