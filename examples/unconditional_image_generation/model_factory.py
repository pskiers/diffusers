import torch
from diffusers import UNet2DModel
from conditional_unet_model import UnifiedConditionUNet
from model_utils import trunc_normal_init_

def build_model(args):
    """
    Factory to instantiate the correct UNet based on explicit model_type args.
    """
    sample_size = args.resolution if args.vae_name is None else args.resolution // 8

    in_channels = args.input_channels * 3 if args.use_small_loop else args.input_channels
    out_channels = args.input_channels * 2 if args.use_small_loop else args.input_channels

    if args.model_type == "unet2d":
        num_class_embeds = args.num_classes if (args.num_classes is not None and args.num_classes > 0) else None
        if args.model_config_name_or_path is None:
            model = UNet2DModel(
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=args.layers_per_block,
                block_out_channels=args.channels,
                down_block_types=args.down_block_types,
                up_block_types=args.up_block_types,
                num_class_embeds=num_class_embeds,
            )
        else:
            config = UNet2DModel.load_config(args.model_config_name_or_path)
            model = UNet2DModel.from_config(config)

    elif args.model_type == "unified_class":
        if args.model_config_name_or_path is None:
            model = UnifiedConditionUNet(
                condition_mode="class",
                num_classes=args.num_classes,
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=args.layers_per_block,
                block_out_channels=args.channels,
                down_block_types=args.down_block_types,
                up_block_types=args.up_block_types,
                cross_attention_dim=args.channels[-1] if args.channels else 1024,
            )
        else:
            config = UnifiedConditionUNet.load_config(args.model_config_name_or_path)
            model = UnifiedConditionUNet.from_config(config)

    elif args.model_type == "unified_sequence":
        raw_dim = 21 if args.dataset_mode == "absolute" else 55
        if args.model_config_name_or_path is None:
            model = UnifiedConditionUNet(
                condition_mode="sequence",
                raw_dim=raw_dim,
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=args.layers_per_block,
                block_out_channels=args.channels,
                down_block_types=args.down_block_types,
                up_block_types=args.up_block_types,
                cross_attention_dim=args.channels[-1] if args.channels else 1024,
            )
        else:
            config = UnifiedConditionUNet.load_config(args.model_config_name_or_path)
            model = UnifiedConditionUNet.from_config(config)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Safely attach recursive tokens for the Small Loop (in-memory only)
    if args.use_small_loop:
        y_init = trunc_normal_init_(torch.empty((1, args.input_channels, sample_size, sample_size), dtype=torch.float32), std=1)
        z_init = trunc_normal_init_(torch.empty((1, args.input_channels, sample_size, sample_size), dtype=torch.float32), std=1)
        model.y_init = y_init
        model.z_init = z_init

    return model
