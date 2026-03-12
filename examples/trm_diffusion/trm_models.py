import torch
import torch.nn as nn
import os
import contextlib
from dataclasses import dataclass
from diffusers.models import UNet2DConditionModel, Transformer2DModel, UNet2DModel
from diffusers.utils import BaseOutput
from accelerate.utils import extract_model_from_parallel as unwrap_model
from trm_utils import deep_recursion, get_model_output
from model_utils import load_with_backward_compatibility
from safetensors.torch import load_file


@dataclass
class TRMOutput(BaseOutput):
    """Simple dataclass to mimic Diffusers model outputs."""

    sample: torch.FloatTensor


class BaseIterativeStrategy:
    """
    Pure Python Strategy base class for TRM models. Natively resolves Hugging Face
    mixed-precision autowrapping and radically simplifies state/checkpoint tracking.
    """

    def __init__(self, core_model, state_shape, n=6, T=3, n_sup=1):
        self.core_model = core_model
        self.n = n
        self.T = T
        self.n_sup = n_sup
        self.state_shape = state_shape

        if hasattr(self.core_model, "register_to_config"):
            self.core_model.register_to_config(trm_n=n, trm_T=T, trm_n_sup=n_sup, trm_state_shape=list(state_shape))

        # Tracked manually rather than as nn.Module buffers
        self.y_init = torch.randn(1, *state_shape)
        self.z_init = torch.randn(1, *state_shape)

    @property
    def device(self):
        return self.core_model.device

    def get_trainable_modules(self):
        """Returns a dict of standard PyTorch modules to be hooked by Accelerate."""
        return {"core_model": self.core_model}

    def update_modules(self, prepared_modules):
        """Injects the wrapped modules back into the strategy."""
        if "core_model" in prepared_modules:
            self.core_model = prepared_modules["core_model"]

    def train(self):
        for m in self.get_trainable_modules().values():
            m.train()

    def eval(self):
        for m in self.get_trainable_modules().values():
            m.eval()

    def get_initial_states(self, batch_size):
        expand_dims = [-1] * len(self.state_shape)
        y = self.y_init.to(self.device).expand(batch_size, *expand_dims).clone()
        z = self.z_init.to(self.device).expand(batch_size, *expand_dims).clone()
        return y, z

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        core = self.core_model.module if hasattr(self.core_model, "module") else self.core_model
        core.save_pretrained(os.path.join(output_dir, "unet"))
        torch.save({"y_init": self.y_init, "z_init": self.z_init}, os.path.join(output_dir, "strategy_state.pt"))

    def load(self, input_dir):
        unet_dir = os.path.join(input_dir, "unet")
        sf_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

        if os.path.exists(sf_path):
            state_dict = load_file(sf_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Could not find model weights in {unet_dir}")

        core = self.core_model.module if hasattr(self.core_model, "module") else self.core_model
        load_with_backward_compatibility(core, state_dict)

        state_path = os.path.join(input_dir, "strategy_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            self.y_init, self.z_init = state["y_init"], state["z_init"]

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        raise NotImplementedError

    def __call__(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        raise NotImplementedError

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


class StandardTRM(BaseIterativeStrategy):
    """Original Tiny Recursive Model, now powered by the Strategy Pattern."""

    def __init__(self, core_model, state_channels, resolution, n=6, T=3, n_sup=1):
        state_shape = (state_channels, resolution, resolution)
        super().__init__(core_model, state_shape, n, T, n_sup)

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        model_output, y_next, z_next = deep_recursion(
            self.core_model, x, y, z, timesteps, conditions, masks, self.n, self.T
        )
        return model_output, y_next, z_next

    def __call__(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        model_output = None
        for _ in range(self.n_sup):
            model_output, y, z = self.reasoning_step(sample, y, z, timestep, conditions, attention_mask)
        return TRMOutput(sample=model_output)


class ExtraModulesMixin:
    """Handles Accelerate prep and I/O natively for auxiliary layers"""

    @property
    def _extra_modules(self):
        return ["fusion", "out_proj", "norm_y", "norm_z"]

    def get_trainable_modules(self):
        modules = super().get_trainable_modules() if hasattr(super(), "get_trainable_modules") else {}
        for attr in self._extra_modules:
            if hasattr(self, attr):
                modules[attr] = getattr(self, attr)
        return modules

    def update_modules(self, prepared_modules):
        if hasattr(super(), "update_modules"):
            super().update_modules(prepared_modules)
        for attr in self._extra_modules:
            if attr in prepared_modules:
                setattr(self, attr, prepared_modules[attr])

    def save(self, output_dir):
        if hasattr(super(), "save"):
            super().save(output_dir)
        for attr in self._extra_modules:
            if hasattr(self, attr):
                m = unwrap_model(getattr(self, attr))
                torch.save(m.state_dict(), os.path.join(output_dir, f"{attr}.pt"))

    def load(self, input_dir):
        if hasattr(super(), "load"):
            super().load(input_dir)
        for attr in self._extra_modules:
            if hasattr(self, attr) and os.path.exists(os.path.join(input_dir, f"{attr}.pt")):
                m = unwrap_model(getattr(self, attr))
                m.load_state_dict(torch.load(os.path.join(input_dir, f"{attr}.pt"), map_location="cpu"))


# V2: TRM exactly as in the og paper. Addition instead of concat, single output, no x for y prediction, in high dim latent
class UNetTRMv2(ExtraModulesMixin, BaseIterativeStrategy):
    """High-Dimensional Additive TRM specialized for UNet architectures."""

    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        if not isinstance(core_model, (UNet2DConditionModel, UNet2DModel)):
            raise ValueError(f"UNetTRMv2 requires a UNet model, got {core_model.__class__}")

        dim = core_model.conv_in.out_channels
        state_shape = (dim, resolution, resolution)
        super().__init__(core_model, state_shape, n, T, n_sup)

        self.norm_y = nn.GroupNorm(32, dim)
        self.norm_z = nn.GroupNorm(32, dim)

    @property
    def _core(self):
        return self.core_model.module if hasattr(self.core_model, "module") else self.core_model

    @contextlib.contextmanager
    def bypass_projections(self):
        proj_in, proj_out = self._core.conv_in, self._core.conv_out
        self._core.conv_in, self._core.conv_out = nn.Identity(), nn.Identity()
        try:
            yield proj_in, proj_out
        finally:
            self._core.conv_in, self._core.conv_out = proj_in, proj_out

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks, autocast_ctx):
        for _ in range(self.n):
            with autocast_ctx:
                z_out = get_model_output(self.core_model, x_high + y + z, timesteps, conditions, masks)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)  # Force state to remain FP32

        with autocast_ctx:
            y_out = get_model_output(self.core_model, y + z, timesteps, conditions, masks)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z

    def _deep_recursion(self, x_high, y, z, timesteps, conditions, masks, autocast_ctx):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x_high, y, z, timesteps, conditions, masks, autocast_ctx)
        y_final, z_final = self._latent_recursion(x_high, y, z, timesteps, conditions, masks, autocast_ctx)
        return y_final, y_final.detach(), z_final.detach()

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )

        with self.bypass_projections() as (proj_in, proj_out):
            with autocast_ctx:
                x_high = proj_in(x)
            x_high = x_high.to(torch.float32)  # Force state to remain FP32

            y_final_high, y_next, z_next = self._deep_recursion(
                x_high, y, z, timesteps, conditions, masks, autocast_ctx
            )

            with autocast_ctx:
                y_final_4ch = proj_out(y_final_high)

        return y_final_4ch, y_next, z_next

    def __call__(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        dtype = sample.dtype
        autocast_ctx = (
            torch.autocast(device_type=sample.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        with self.bypass_projections() as (proj_in, proj_out):
            with autocast_ctx:
                x_high = proj_in(sample)
            x_high = x_high.to(torch.float32)

            bsz = sample.shape[0]
            y_high, z_high = self.get_initial_states(bsz)

            for _ in range(self.n_sup):
                y_high, z_high = self._latent_recursion(
                    x_high, y_high, z_high, timestep, conditions, attention_mask, autocast_ctx
                )

            with autocast_ctx:
                model_output = proj_out(y_high)

        return TRMOutput(sample=model_output)


class DiTTRMv2(ExtraModulesMixin, BaseIterativeStrategy):
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

        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)

    @property
    def _core(self):
        return self.core_model.module if hasattr(self.core_model, "module") else self.core_model

    def get_trainable_modules(self):
        """Expose all manual DiT submodules so Accelerate can wrap them with DDP."""
        modules = super().get_trainable_modules()
        if "core_model" in modules:
            del modules["core_model"] # Prevent full DDP wrapping on DiT
        core = self._core

        # The definitive list of every possible nn.Module across all Diffusers Transformer2D configs
        possible_modules = [
            "time_proj",
            "time_embedding",
            "proj",
            "pos_embed",
            "condition_projector",
            "norm_out",
            "proj_out",
            "proj_out_1",
            "proj_out_2",
            "adaln_single",
            "caption_projection",
            "norm",
            "proj_in",
            "latent_image_embedding",
            "out",
        ]

        for attr in possible_modules:
            if hasattr(core, attr) and getattr(core, attr) is not None:
                modules[attr] = getattr(core, attr)

        # Unroll transformer blocks
        if hasattr(core, "transformer_blocks"):
            for i, block in enumerate(core.transformer_blocks):
                modules[f"transformer_block_{i}"] = block

        return modules

    def update_modules(self, prepared_modules):
        """Re-inject the DDP-wrapped modules back into the DiT core."""
        super().update_modules(prepared_modules)
        core = self._core

        possible_modules = [
            "time_proj",
            "time_embedding",
            "proj",
            "pos_embed",
            "condition_projector",
            "norm_out",
            "proj_out",
            "proj_out_1",
            "proj_out_2",
            "adaln_single",
            "caption_projection",
            "norm",
            "proj_in",
            "latent_image_embedding",
            "out",
        ]

        for attr in possible_modules:
            if attr in prepared_modules:
                setattr(core, attr, prepared_modules[attr])

        if hasattr(core, "transformer_blocks"):
            for i in range(len(core.transformer_blocks)):
                if f"transformer_block_{i}" in prepared_modules:
                    core.transformer_blocks[i] = prepared_modules[f"transformer_block_{i}"]

    def save(self, output_dir):
        """Temporarily unwrap modules to save cleanly without 'module.' prefixes."""

        core = self._core
        original_modules = {}

        # 1. Unwrap
        for name, module in self.get_trainable_modules().items():
            if name in self._extra_modules:
                continue # Let ExtraModulesMixin handle norm_y, etc.
            original_modules[name] = module
            unwrapped = unwrap_model(module)
            if name.startswith("transformer_block_"):
                idx = int(name.split("_")[-1])
                core.transformer_blocks[idx] = unwrapped
            else:
                setattr(core, name, unwrapped)

        # 2. Save
        super().save(output_dir)

        # 3. Restore DDP Wrappers for continued training
        for name, module in original_modules.items():
            if name.startswith("transformer_block_"):
                idx = int(name.split("_")[-1])
                core.transformer_blocks[idx] = module
            else:
                setattr(core, name, module)

    def _prepare_conditions(self, conditions, bs, device):
        """Replicates the conditioning override logic from UnifiedConditionDiT."""
        encoder_hidden_states = None
        class_labels = torch.zeros((bs,), dtype=torch.long, device=device)

        if not hasattr(self._core.config, "condition_mode"):
            return conditions, None
        if conditions is not None:
            if self._core.config.condition_mode == "class":
                encoder_hidden_states = self._core.condition_projector(conditions).unsqueeze(1)
            elif self._core.config.condition_mode == "sequence":
                encoder_hidden_states = self._core.condition_projector(conditions)
            elif self._core.config.condition_mode == "class_adaln":
                class_labels = conditions
        elif self._core.config.condition_mode == "class_adaln":
            class_labels = torch.full((bs,), self._core.config.num_classes, dtype=torch.long, device=device)
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
        for block in self._core.transformer_blocks:
            if torch.is_grad_enabled() and self._core.gradient_checkpointing:
                hidden_states = self._core._gradient_checkpointing_func(
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

    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
    ):
        for _ in range(self.n):
            with autocast_ctx:
                z_out = self._dit_blocks(
                    x_high + y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)  # Force state to remain FP32

        with autocast_ctx:
            y_out = self._dit_blocks(y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z

    def _deep_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
    ):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(
                    x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
                )
        y_final, z_final = self._latent_recursion(
            x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
        )
        return y_final, y_final.detach(), z_final.detach()

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )

        bsz = x.shape[0]
        timestep = self._format_timestep(timesteps, bsz, x.device)
        encoder_hidden_states, class_labels = self._prepare_conditions(conditions, bs=bsz, device=x.device)
        encoder_attention_mask = masks

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(x.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        with autocast_ctx:
            x_high, encoder_hidden_states, ts, embedded_ts = self._core._operate_on_patched_inputs(
                x, encoder_hidden_states, timestep, None
            )
        x_high = x_high.to(torch.float32)  # Force state to remain FP32

        y_final_high, y_next, z_next = self._deep_recursion(
            x_high, y, z, encoder_hidden_states, ts, class_labels, None, encoder_attention_mask, None, autocast_ctx
        )

        with autocast_ctx:
            y_final_4ch = self._core._get_output_for_patched_inputs(
                hidden_states=y_final_high,
                timestep=ts,
                class_labels=class_labels,
                embedded_timestep=embedded_ts,
                height=self.h_p,
                width=self.w_p,
            )
        return y_final_4ch, y_next, z_next

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        class_labels=None,
        attention_mask=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        dtype = sample.dtype
        autocast_ctx = (
            torch.autocast(device_type=sample.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )

        bsz = sample.shape[0]
        timestep = self._format_timestep(timestep, bsz, sample.device)
        y, z = self.get_initial_states(bsz)

        conditions = class_labels if class_labels is not None else encoder_hidden_states
        encoder_hidden_states, class_labels = self._prepare_conditions(conditions, bs=bsz, device=sample.device)

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

        with autocast_ctx:
            x_high, encoder_hidden_states, ts, embedded_ts = self._core._operate_on_patched_inputs(
                sample, encoder_hidden_states, timestep, None
            )
        x_high = x_high.to(torch.float32)

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
                None,
                autocast_ctx,
            )

        with autocast_ctx:
            model_output = self._core._get_output_for_patched_inputs(
                hidden_states=y_final_high,
                timestep=ts,
                class_labels=class_labels,
                embedded_timestep=embedded_ts,
                height=self.h_p,
                width=self.w_p,
            )

        return TRMOutput(sample=model_output)


# ==========================================
# V3: Concatenation with Single Output
# ==========================================
class UNetTRMv3(UNetTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.conv_in.out_channels
        self.fusion = nn.Conv2d(3 * dim, dim, kernel_size=1, device=self.device, dtype=self.core_model.dtype)

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks, autocast_ctx=None):
        zeros = torch.zeros_like(x_high)
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=1))
                z_out = get_model_output(self.core_model, z_in, timesteps, conditions, masks)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=1))
            y_out = get_model_output(self.core_model, y_in, timesteps, conditions, masks)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


class DiTTRMv3(DiTTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.config.num_attention_heads * self.core_model.config.attention_head_dim
        self.fusion = nn.Linear(3 * dim, dim, device=self.device, dtype=self.core_model.dtype)

    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx=None
    ):
        zeros = torch.zeros_like(x_high)
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
                z_out = self._dit_blocks(z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=-1))
            y_out = self._dit_blocks(y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
            y = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


# ==========================================
# V4: Concatenation with Two Outputs (y & z)
# ==========================================
class UNetTRMv4(UNetTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.conv_in.out_channels
        self.fusion = nn.Conv2d(3 * dim, dim, kernel_size=1, device=self.device)
        self.out_proj = nn.Conv2d(dim, 2 * dim, kernel_size=1, device=self.device)

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks, autocast_ctx=None):
        zeros = torch.zeros_like(x_high)
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=1))
                out = get_model_output(self.core_model, z_in, timesteps, conditions, masks)
                _, z_out = self.out_proj(out).chunk(2, dim=1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=1))
            out = get_model_output(self.core_model, y_in, timesteps, conditions, masks)
            y_out, _ = self.out_proj(out).chunk(2, dim=1)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


class DiTTRMv4(DiTTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.config.num_attention_heads * self.core_model.config.attention_head_dim
        self.fusion = nn.Linear(3 * dim, dim, device=self.device)
        self.out_proj = nn.Linear(dim, 2 * dim, device=self.device)

    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx=None
    ):
        zeros = torch.zeros_like(x_high)
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
                out = self._dit_blocks(z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
                _, z_out = self.out_proj(out).chunk(2, dim=-1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=-1))
            out = self._dit_blocks(y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
            y_out, _ = self.out_proj(out).chunk(2, dim=-1)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


# ==========================================
# V5: Concatenation + Two Outputs + x_high in y pass
# ==========================================
class UNetTRMv5(UNetTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.conv_in.out_channels
        self.fusion = nn.Conv2d(3 * dim, dim, kernel_size=1, device=self.device)
        self.out_proj = nn.Conv2d(dim, 2 * dim, kernel_size=1, device=self.device)

    def _latent_recursion(self, x_high, y, z, timesteps, conditions, masks, autocast_ctx=None):
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=1))
                out = get_model_output(self.core_model, z_in, timesteps, conditions, masks)
                _, z_out = self.out_proj(out).chunk(2, dim=1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            # x_high is passed instead of zeros!
            y_in = self.fusion(torch.cat([x_high, y, z], dim=1))
            out = get_model_output(self.core_model, y_in, timesteps, conditions, masks)
            y_out, _ = self.out_proj(out).chunk(2, dim=1)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


class DiTTRMv5(DiTTRMv2):
    def __init__(self, core_model, resolution, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, resolution, n, T, n_sup, **kwargs)
        dim = self.core_model.config.num_attention_heads * self.core_model.config.attention_head_dim
        self.fusion = nn.Linear(3 * dim, dim, device=self.device)
        self.out_proj = nn.Linear(dim, 2 * dim, device=self.device)

    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx=None
    ):
        for _ in range(self.n):
            with autocast_ctx:
                z_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
                out = self._dit_blocks(z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
                _, z_out = self.out_proj(out).chunk(2, dim=-1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            # x_high is passed instead of zeros!
            y_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
            out = self._dit_blocks(y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs)
            y_out, _ = self.out_proj(out).chunk(2, dim=-1)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z
