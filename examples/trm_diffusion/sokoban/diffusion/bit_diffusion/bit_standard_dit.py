import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers import UNet2DConditionModel, Transformer2DModel
from diffusers.configuration_utils import register_to_config



class UnifiedConditionDiT(Transformer2DModel):
    """DiT model for standard diffusion generation"""
    @register_to_config
    def __init__(
        self,
        condition_mode="class",  # "class", "class_adaln", or "sequence"
        num_classes=1000,
        raw_dim=21,
        # --- Standard Transformer2DModel args ---
        dropout=0.0,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        num_layers=12,
        patch_size=2,
        attention_head_dim=64,
        num_attention_heads=16,
        cross_attention_dim=1024,
        activation_fn="gelu-approximate",
    ):
        # Configure norm and cross-attention based on mode.
        # "spatial_concat" concatenates the condition map to the noisy input before
        # patch-embedding, so it also needs no cross-attention.
        is_adaln = condition_mode in ("class_adaln", "spatial_concat")

        # FIX: DiTs ALWAYS require ada_norm_zero to process the diffusion timestep.
        norm_type = "ada_norm_zero"

        # class_adaln uses all classes + dropout token; everything else just needs
        # a single dummy embedding to carry the timestep through adaLN.
        num_embeds_ada_norm = num_classes + 1 if condition_mode == "class_adaln" else 1

        # In adaLN / spatial_concat mode, strip out cross-attention. Otherwise, use it.
        actual_cross_attn_dim = None if is_adaln else cross_attention_dim

        # Call parent explicitly
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            patch_size=patch_size,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=actual_cross_attn_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            norm_type=norm_type,
            dropout=dropout
        )

        if self.config.condition_mode == "class":
            self.condition_projector = nn.Embedding(self.config.num_classes + 1, self.config.cross_attention_dim)
        elif self.config.condition_mode == "sequence":
            self.condition_projector = nn.Sequential(
                nn.Linear(self.config.raw_dim, self.config.cross_attention_dim),
                nn.SiLU(),
                nn.Linear(self.config.cross_attention_dim, self.config.cross_attention_dim),
            )
        elif self.config.condition_mode == "class_adaln":
            # Diffusers handles the adaLN embedding internally, so we don't need a projector
            self.condition_projector = None
        elif self.config.condition_mode == "spatial_concat":
            # Condition is a (C, H, W) spatial map concatenated to the noisy input
            # before patch-embedding; no projector needed.
            self.condition_projector = None
        else:
            raise ValueError(f"Unknown condition_mode: {self.config.condition_mode}")

    def forward(self, sample, timestep, condition_tensors=None, attention_mask=None, **kwargs):
        """
        Generic forward pass matching your custom UNet signature.
        """
        encoder_hidden_states = None

        # Default dummy class_labels to carry the timestep through the adaLN block
        class_labels = torch.zeros((sample.shape[0],), dtype=torch.long, device=sample.device)

        if condition_tensors is not None:
            if self.config.condition_mode == "class":
                encoder_hidden_states = self.condition_projector(condition_tensors).unsqueeze(1)
            elif self.config.condition_mode == "sequence":
                encoder_hidden_states = self.condition_projector(condition_tensors)
            elif self.config.condition_mode == "class_adaln":
                # Override the dummy labels with the actual condition labels
                class_labels = condition_tensors
            elif self.config.condition_mode == "spatial_concat":
                # Concatenate the spatial map to the noisy input before patch-embedding.
                # condition_tensors: (B, C_mask, H, W); sample: (B, C_noise, H, W)
                sample = torch.cat([sample, condition_tensors.to(sample.dtype)], dim=1)
        elif self.config.condition_mode == "class_adaln":
            # Unconditional inference fallback for class_adaln
            class_labels = torch.full(
                (sample.shape[0],), self.config.num_classes, dtype=torch.long, device=sample.device
            )

        return super().forward(
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            encoder_attention_mask=attention_mask,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Bypass the buggy diffusers class remapping logic for custom classes.
        This forces diffusers to instantiate THIS class during EMA saving/loading.
        """
        init_dict, unused_kwargs, hidden_config_dict = cls.extract_init_dict(config, **kwargs)
        return cls(**init_dict)
