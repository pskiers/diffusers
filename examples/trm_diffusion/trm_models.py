import torch
import torch.nn as nn
import contextlib
from diffusers.models import UNet2DConditionModel, Transformer2DModel, UNet2DModel
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from trm_utils import deep_recursion, get_model_output


@dataclass
class TRMOutput(BaseOutput):
    """Simple dataclass to mimic Diffusers model outputs."""

    sample: torch.FloatTensor


class BaseIterativeModel(nn.Module):
    """
    Abstract base class. Accepts a generic `state_shape` tuple so y and z
    can be natively 3D (for DiT) or 4D (for UNet).
    """

    def __init__(self, core_model, state_shape, n=6, T=3, n_sup=1):
        super().__init__()
        self.core_model = core_model
        self.n = n
        self.T = T
        self.n_sup = n_sup

        if hasattr(self.core_model, "register_to_config"):
            self.core_model.register_to_config(trm_n=n, trm_T=T, trm_n_sup=n_sup, trm_state_shape=list(state_shape))

        # Initialize directly in the shape required by the architecture
        self.core_model.register_buffer("y_init", torch.randn(1, *state_shape))
        self.core_model.register_buffer("z_init", torch.randn(1, *state_shape))

    def _format_timestep(self, timestep, batch_size, device):
        """Ensures timestep is a 1D tensor matching the batch size."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        timestep = timestep.to(device)
        if timestep.shape[0] != batch_size:
            timestep = timestep.expand(batch_size)
        return timestep

    @property
    def device(self):
        return self.core_model.device

    def get_initial_states(self, batch_size):
        expand_dims = [-1] * len(self.core_model.y_init.shape[1:])
        y = self.core_model.y_init.expand(batch_size, *expand_dims).clone()
        z = self.core_model.z_init.expand(batch_size, *expand_dims).clone()
        return y, z

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        raise NotImplementedError

    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        raise NotImplementedError


class StandardTRM(BaseIterativeModel):
    """Original Tiny Recursive Model."""

    def __init__(self, core_model, state_channels, resolution, n=6, T=3, n_sup=1):
        state_shape = (state_channels, resolution, resolution)
        super().__init__(core_model, state_shape, n, T, n_sup)

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        model_output, y_next, z_next = deep_recursion(
            self.core_model, x, y, z, timesteps, conditions, masks, self.n, self.T
        )
        return model_output, y_next, z_next

    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)
        y, z = y.to(sample.device), z.to(sample.device)
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        model_output = None
        for _ in range(self.n_sup):
            model_output, y, z = self.reasoning_step(sample, y, z, timestep, conditions, attention_mask)
        return TRMOutput(sample=model_output)


# =========================================================================
# UNET SPECIFIC MODEL
# =========================================================================
class UNetTRMv2(BaseIterativeModel):
    """High-Dimensional Additive TRM specialized for UNet architectures."""

    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        if not isinstance(core_model, (UNet2DConditionModel, UNet2DModel)):
            raise ValueError(f"UNetTRMv2 requires a UNet model, got {core_model.__class__}")

        dim = core_model.conv_in.out_channels
        state_shape = (dim, resolution, resolution)
        super().__init__(core_model, state_shape, n, T, n_sup)

    @contextlib.contextmanager
    def bypass_projections(self):
        """Safely swaps in/out convolutions as a class method."""
        proj_in, proj_out = self.core_model.conv_in, self.core_model.conv_out
        self.core_model.conv_in, self.core_model.conv_out = nn.Identity(), nn.Identity()
        try:
            yield proj_in, proj_out
        finally:
            self.core_model.conv_in, self.core_model.conv_out = proj_in, proj_out

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks):
        for _ in range(self.n):
            z = get_model_output(self.core_model, x_high + y + z, timesteps, conditions, masks)
        y = get_model_output(self.core_model, y + z, timesteps, conditions, masks)
        return y, z

    def _deep_recursion(self, x_high, y, z, timesteps, conditions, masks):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x_high, y, z, timesteps, conditions, masks)
        y_final, z_final = self._latent_recursion(x_high, y, z, timesteps, conditions, masks)
        return y_final, y_final.detach(), z_final.detach()

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        """Training entry point called by train.py."""
        with self.bypass_projections() as (proj_in, proj_out):
            x_high = proj_in(x)
            y_final_high, y_next, z_next = self._deep_recursion(x_high, y, z, timesteps, conditions, masks)
            y_final_4ch = proj_out(y_final_high)
        return y_final_4ch, y_next, z_next

    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        """Standard forward pass for Diffusers pipelines (evaluation/inference)."""
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        with self.bypass_projections() as (proj_in, proj_out):
            x_high = proj_in(sample)

            bsz = sample.shape[0]
            y_high, z_high = self.get_initial_states(bsz)
            y_high, z_high = y_high.to(sample.device), z_high.to(sample.device)

            for _ in range(self.n_sup):
                y_high, z_high = self._latent_recursion(x_high, y_high, z_high, timestep, conditions, attention_mask)

            model_output = proj_out(y_high)
        return TRMOutput(sample=model_output)


# =========================================================================
# DIT SPECIFIC MODEL
# =========================================================================
class DiTTRMv2(BaseIterativeModel):
    """High-Dimensional Additive TRM specialized for DiT architectures."""

    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        if not isinstance(core_model, Transformer2DModel):
            raise ValueError(f"DiTTRMv2 requires a Transformer2DModel, got {core_model.__class__}")

        dim = core_model.config.num_attention_heads * core_model.config.attention_head_dim
        patch_size = getattr(core_model.config, "patch_size", 1)
        self.h_p = resolution // patch_size
        self.w_p = resolution // patch_size

        state_shape = (self.h_p * self.w_p, dim)
        super().__init__(core_model, state_shape, n, T, n_sup)

    def _prepare_conditions(self, conditions, bs, device):
        """Replicates the conditioning override logic from UnifiedConditionDiT."""
        encoder_hidden_states = None
        class_labels = torch.zeros((bs,), dtype=torch.long, device=device)

        if not hasattr(self.core_model.config, "condition_mode"):
            return conditions, None

        if conditions is not None:
            if self.core_model.config.condition_mode == "class":
                encoder_hidden_states = self.core_model.condition_projector(conditions).unsqueeze(1)
            elif self.core_model.config.condition_mode == "sequence":
                encoder_hidden_states = self.core_model.condition_projector(conditions)
            elif self.core_model.config.condition_mode == "class_adaln":
                class_labels = conditions
        elif self.core_model.config.condition_mode == "class_adaln":
            class_labels = torch.full((bs,), self.core_model.config.num_classes, dtype=torch.long, device=device)

        return encoder_hidden_states, class_labels

    def _dit_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        class_labels,
        attention_mask,
        encoder_attention_mask,
        cross_attention_kwargs,
    ):
        """Passes the 3D sequence directly through the transformer block list exactly like Transformer2DModel.forward."""
        for block in self.core_model.transformer_blocks:
            if torch.is_grad_enabled() and self.core_model.gradient_checkpointing:
                hidden_states = self.core_model._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
        return hidden_states

    def _latent_recursion(self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs):
        for _ in range(self.n):
            z = self._dit_blocks(x_high + y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
        # Prediction pass (drop x)
        y = self._dit_blocks(y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
        return y, z

    def _deep_recursion(self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(
                    x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
        y_final, z_final = self._latent_recursion(
            x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
        )
        return y_final, y_final.detach(), z_final.detach()

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        bsz = x.shape[0]
        timestep = self._format_timestep(timesteps, bsz, x.device)

        # 1. Route conditioning correctly (UnifiedConditionDiT logic)
        encoder_hidden_states, class_labels = self._prepare_conditions(conditions, bs=x.shape[0], device=x.device)

        # 2. Hardcode the unused Diffusers kwargs to None
        cross_attention_kwargs = None
        added_cond_kwargs = None
        attention_mask = None

        # Route the generic 'masks' from train.py strictly to cross-attention
        encoder_attention_mask = masks

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(x.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 3. Input processing
        x_high, encoder_hidden_states, ts, embedded_ts = self.core_model._operate_on_patched_inputs(
            x, encoder_hidden_states, timestep, added_cond_kwargs
        )

        # 4. TRM Loop entirely in 3D sequence space
        y_final_high, y_next, z_next = self._deep_recursion(
            x_high,
            y,
            z,
            encoder_hidden_states,
            ts,
            class_labels,
            attention_mask,
            encoder_attention_mask,
            cross_attention_kwargs,
        )

        # 5. Output Unpatchify
        y_final_4ch = self.core_model._get_output_for_patched_inputs(
            hidden_states=y_final_high,
            timestep=ts,
            class_labels=class_labels,
            embedded_timestep=embedded_ts,
            height=self.h_p,
            width=self.w_p,
        )
        return y_final_4ch, y_next, z_next

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        class_labels=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        bsz = sample.shape[0]
        timestep = self._format_timestep(timestep, bsz, sample.device)
        y, z = self.get_initial_states(bsz)
        y, z = y.to(sample.device), z.to(sample.device)

        # 1. Route conditioning correctly (UnifiedConditionDiT logic)
        conditions = class_labels if class_labels is not None else encoder_hidden_states
        encoder_hidden_states, class_labels = self._prepare_conditions(
            conditions, bs=sample.shape[0], device=sample.device
        )

        # If train.py passed the sequence mask into attention_mask, reroute it to cross-attention.
        if attention_mask is not None and encoder_attention_mask is None:
            encoder_attention_mask = attention_mask
            attention_mask = None

        # Process standard self-attention mask (now safely None for CLEVR)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # Process cross-attention mask (now properly holding the CLEVR sequence mask)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 3. Input processing
        x_high, encoder_hidden_states, ts, embedded_ts = self.core_model._operate_on_patched_inputs(
            sample, encoder_hidden_states, timestep, added_cond_kwargs
        )

        # 4. Loop entirely in high-dimensional space
        for _ in range(self.n_sup):
            y_final_high, y, z = self._deep_recursion(
                x_high,
                y,
                z,
                encoder_hidden_states,
                ts,
                class_labels,
                attention_mask,
                encoder_attention_mask,
                cross_attention_kwargs,
            )

        # 5. Decode ONCE
        model_output = self.core_model._get_output_for_patched_inputs(
            hidden_states=y_final_high,
            timestep=ts,
            class_labels=class_labels,
            embedded_timestep=embedded_ts,
            height=self.h_p,
            width=self.w_p,
        )

        return TRMOutput(sample=model_output)


class UNetTRMv3(UNetTRMv2):
    """High-Dimensional Concatenated TRM inherited from UNetTRMv2."""

    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)

        dim = self.core_model.conv_in.out_channels
        # Attached to the WRAPPER, not the core_model, to protect Diffusers config
        self.fusion = nn.Conv2d(3 * dim, dim, kernel_size=1, device=self.device, dtype=self.core_model.dtype)

        with torch.no_grad():
            eye = torch.eye(dim, device=self.device, dtype=self.core_model.dtype).view(dim, dim, 1, 1)
            self.fusion.weight[:, :dim].copy_(eye)
            self.fusion.weight[:, dim:2*dim].copy_(eye)
            self.fusion.weight[:, 2*dim:].copy_(eye)
            self.fusion.weight.data /= 3.0
            nn.init.zeros_(self.fusion.bias)

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks):
        zeros = torch.zeros_like(x_high)

        for _ in range(self.n):
            z_in = self.fusion(torch.cat([x_high, y, z], dim=1))
            z = z + get_model_output(self.core_model, z_in, timesteps, conditions, masks)

        y_in = self.fusion(torch.cat([zeros, y, z], dim=1))
        y = y + get_model_output(self.core_model, y_in, timesteps, conditions, masks)

        return y, z


class DiTTRMv3(DiTTRMv2):
    """High-Dimensional Concatenated TRM inherited from DiTTRMv2."""

    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)

        dim = self.core_model.config.num_attention_heads * self.core_model.config.attention_head_dim
        # Attached to the WRAPPER, not the core_model
        self.fusion = nn.Linear(3 * dim, dim, device=self.device, dtype=self.core_model.dtype)

        with torch.no_grad():
            eye = torch.eye(dim, device=self.device, dtype=self.core_model.dtype)
            self.fusion.weight[:, :dim].copy_(eye)
            self.fusion.weight[:, dim:2*dim].copy_(eye)
            self.fusion.weight[:, 2*dim:].copy_(eye)
            self.fusion.weight.data /= 3.0
            nn.init.zeros_(self.fusion.bias)

    def _latent_recursion(self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs):
        zeros = torch.zeros_like(x_high)

        for _ in range(self.n):
            z_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
            out = self._dit_blocks(z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
            z = z + out

        y_in = self.fusion(torch.cat([zeros, y, z], dim=-1))
        out = self._dit_blocks(y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
        y = y + out

        return y, z
