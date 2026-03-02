def load_with_backward_compatibility(unet, state_dict, logger=None):
    """
    Translates old checkpoint keys to the new unified model architecture keys.
    """

    RENAME_MAP = {
        # 1. DDP/Wrapper Artifacts
        "module.": "",
        "unet.": "",

        # 2. Old CLEVR -> New Unified
        "projector.": "condition_projector.",

        # 3. Old ImageNet -> New Unified
        "class_embedding.": "condition_projector.",
    }

    adapted_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_str, new_str in RENAME_MAP.items():
            if old_str in new_key:
                new_key = new_key.replace(old_str, new_str)
        adapted_dict[new_key] = value

    # strict=False prevents crashing if minor structural parameters differ
    missing, unexpected = unet.load_state_dict(adapted_dict, strict=False)

    if len(missing) > 0:
        if logger is not None:
            logger.warning(f"Missing keys when loading checkpoint (showing first 5): {missing[:5]} ... ({len(missing)} total)")
        else:
            print(f"Missing keys when loading checkpoint (showing first 5): {missing[:5]} ... ({len(missing)} total)")
    if len(unexpected) > 0:
        if logger is not None:
            logger.warning(f"Unexpected keys in checkpoint (showing first 5): {unexpected[:5]} ... ({len(unexpected)} total)")
            logger.warning("If these unexpected keys should map to the missing keys, add them to the RENAME_MAP!")
        else:
            print(f"Unexpected keys in checkpoint (showing first 5): {unexpected[:5]} ... ({len(unexpected)} total)")
            print("If these unexpected keys should map to the missing keys, add them to the RENAME_MAP!")