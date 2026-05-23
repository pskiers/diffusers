import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import contextlib
from dataclasses import dataclass
from diffusers.models import UNet2DConditionModel, Transformer2DModel, UNet2DModel
from diffusers.utils import BaseOutput
from accelerate.utils import extract_model_from_parallel as unwrap_model
from trm_utils import deep_recursion, get_model_output
from model_utils import load_with_backward_compatibility
from safetensors.torch import load_file
from models_pt import SpatialEncoder, AttentiveBridge, ConditioningPyramid


@dataclass
class TRMOutput(BaseOutput):
    """Simple dataclass to mimic Diffusers model outputs."""

    sample: torch.FloatTensor


class FastQKNormProcessor(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        # THE FIX: Disable learnable weights.
        # 1. Mathematically enforces bounds so the model can't re-learn the explosion.
        # 2. Adds zero new parameters, so EMAModel copies perfectly to standard dummy models.
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs
    ):
        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # ---------------------------------------------------------
        # SAFEGUARD: Ensure mask is 4D for proper SDPA broadcasting
        # ---------------------------------------------------------
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(2)

        # Apply parameterless norm and cast back to native precision
        query = self.q_norm(query).to(value.dtype)
        key = self.k_norm(key).to(value.dtype)

        # Native SDPA
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


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

        self._inject_qk_norm()

    def _inject_qk_norm(self):
        """Automatically applies QK-Norm ONLY to Transformer-based models."""
        core = self.core_model.module if hasattr(self.core_model, "module") else self.core_model

        # SAFEGUARD: Skip QK-Norm for UNets, they don't need it.
        if not isinstance(core, Transformer2DModel):
            return

        for name, module in core.named_modules():
            # Check for Diffusers Attention block
            if type(module).__name__ == "Attention" and hasattr(module, "set_processor"):
                if module.to_k.in_features != module.inner_dim:
                    continue

                head_dim = module.inner_dim // module.heads
                processor = FastQKNormProcessor(head_dim).to(device=core.device, dtype=core.dtype)
                module.set_processor(processor)

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

    def call_with_early_stop(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        class_labels=None,
        attention_mask=None,
        threshold=None,
        alpha_bar=None,
        **kwargs,
    ):
        """
        Like __call__ but optionally stops early when consecutive predictions converge.

        Stops when:  sqrt((1 - alpha_bar) / alpha_bar) * ||eps_n - eps_{n-1}|| < threshold

        This is the x0-space distance between consecutive noise predictions.

        NOTE: Does NOT apply CFG internally. Pass an already-doubled batch (cond + uncond
        concatenated) and merge the output yourself, exactly as in the standard denoising loop.

        Args:
            threshold:  float stopping threshold (x0-space L2 norm, mean over batch).
                        If None, runs all n_sup steps (no early stopping).
            alpha_bar:  scheduler's alphas_cumprod[t] at the current timestep.
                        Required when threshold is not None.

        Returns:
            (TRMOutput, n_steps_taken)
        """
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        model_output = None
        prev_output = None
        n_taken = 0

        for i in range(self.n_sup):
            model_output, y, z = self.reasoning_step(sample, y, z, timestep, conditions, attention_mask)
            n_taken = i + 1

            if threshold is not None and prev_output is not None:
                if alpha_bar is None:
                    raise ValueError("alpha_bar is required when threshold is set")
                ab = alpha_bar.to(model_output.device).float()
                scale = torch.sqrt((1.0 - ab) / ab.clamp(min=1e-8))
                diff_norm = (model_output.float() - prev_output).view(bsz, -1).norm(dim=-1).mean()
                if (scale * diff_norm).item() < threshold:
                    break

            prev_output = model_output.detach().float()

        return TRMOutput(sample=model_output), n_taken

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
            if not hasattr(self, attr):
                continue
            path = os.path.join(input_dir, f"{attr}.pt")
            m = unwrap_model(getattr(self, attr))
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location="cpu"))
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                # Checkpoint predates this norm layer — replace with Identity to preserve
                # the original unnormalized dynamics the model was trained with.
                setattr(self, attr, nn.Identity())


class DiTUtilsMixin:
    """Shared utility methods for processing and DDP-wrapping Diffusers DiTs."""

    modules_for_wrapping = [
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

    @contextlib.contextmanager
    def _unwrap_first_block(self, model):
        """Temporarily unwrap the first transformer block so Diffusers can access .norm1 natively."""
        if not hasattr(model, "transformer_blocks") or len(model.transformer_blocks) == 0:
            yield
            return
        original_block = model.transformer_blocks[0]
        model.transformer_blocks[0] = unwrap_model(original_block)
        try:
            yield
        finally:
            model.transformer_blocks[0] = original_block

    def _prepare_conditions(self, conditions, bs, device, model):
        """Replicates the conditioning override logic from UnifiedConditionDiT."""
        encoder_hidden_states = None
        class_labels = torch.zeros((bs,), dtype=torch.long, device=device)

        if not hasattr(model.config, "condition_mode"):
            return conditions, None

        if conditions is not None:
            if model.config.condition_mode == "class":
                # Embeddings STRICTLY require Long tensors
                encoder_hidden_states = model.condition_projector(conditions.long()).unsqueeze(1)
            elif model.config.condition_mode == "sequence":
                # Linear layers STRICTLY require matching float/half precision
                proj_dtype = next(model.condition_projector.parameters()).dtype
                encoder_hidden_states = model.condition_projector(conditions.to(dtype=proj_dtype))
            elif model.config.condition_mode == "class_adaln":
                # AdaLN uses native Diffusers embeddings, which require Long tensors
                class_labels = conditions.long()
        elif model.config.condition_mode == "class_adaln":
            class_labels = torch.full((bs,), model.config.num_classes, dtype=torch.long, device=device)

        return encoder_hidden_states, class_labels

    def _dit_blocks(
        self,
        model,
        hidden_states,
        encoder_hidden_states,
        timestep,
        class_labels,
        attention_mask,
        encoder_attention_mask,
        cross_attention_kwargs,
    ):
        for block in model.transformer_blocks:
            if torch.is_grad_enabled() and model.gradient_checkpointing:
                hidden_states = model._gradient_checkpointing_func(
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

    def _get_dit_modules(self, model, prefix=""):
        """Extracts native Diffusers DiT modules for Accelerate DDP wrapping."""
        modules = {}
        for attr in self.modules_for_wrapping:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                modules[f"{prefix}{attr}"] = getattr(model, attr)
        if hasattr(model, "transformer_blocks"):
            for i, block in enumerate(model.transformer_blocks):
                modules[f"{prefix}transformer_block_{i}"] = block
        return modules

    def _update_dit_modules(self, model, prepared_modules, prefix=""):
        """Re-injects DDP-wrapped modules back into the Diffusers DiT."""
        for attr in self.modules_for_wrapping:
            key = f"{prefix}{attr}"
            if key in prepared_modules:
                setattr(model, attr, prepared_modules[key])
        if hasattr(model, "transformer_blocks"):
            for i in range(len(model.transformer_blocks)):
                key = f"{prefix}transformer_block_{i}"
                if key in prepared_modules:
                    model.transformer_blocks[i] = prepared_modules[key]


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
                _, y_high, z_high = self._deep_recursion(
                    x_high, y_high, z_high, timestep, conditions, attention_mask, autocast_ctx
                )

            with autocast_ctx:
                model_output = proj_out(y_high)

        return TRMOutput(sample=model_output)


class DiTTRMv2(DiTUtilsMixin, ExtraModulesMixin, BaseIterativeStrategy):
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
        modules = super().get_trainable_modules()
        if "core_model" in modules:
            del modules["core_model"]  # Prevent full DDP wrapping on DiT
        modules.update(self._get_dit_modules(self._core))
        return modules

    def update_modules(self, prepared_modules):
        super().update_modules(prepared_modules)
        self._update_dit_modules(self._core, prepared_modules)

    def save(self, output_dir):
        """Temporarily unwrap modules to save cleanly without 'module.' prefixes."""
        original_modules = {}

        # 1. Unwrap
        for name, module in self.get_trainable_modules().items():
            if name in self._extra_modules:
                continue  # Let ExtraModulesMixin handle norm_y, etc.
            original_modules[name] = module

        unwrapped_modules = {k: unwrap_model(v) for k, v in original_modules.items()}

        # Inject unwrapped layers back into Diffusers so it can save cleanly
        self._update_dit_modules(self._core, unwrapped_modules)

        # 2. Save using base strategy
        super().save(output_dir)

        # 3. Restore DDP Wrappers for continued training
        self._update_dit_modules(self._core, original_modules)

    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
    ):
        for _ in range(self.n):
            with autocast_ctx:
                z_out = self._dit_blocks(
                    self._core, x_high + y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_out = self._dit_blocks(
                self._core, y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
            )
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

    def encode_features(self, x, timesteps, conditions=None, masks=None):
        dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )

        bsz = x.shape[0]
        timestep = self._format_timestep(timesteps, bsz, x.device)
        encoder_hidden_states, class_labels = self._prepare_conditions(
            conditions, bs=bsz, device=x.device, model=self._core
        )

        encoder_attention_mask = masks
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(x.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        with autocast_ctx:
            x_high, encoder_hidden_states, ts, embedded_ts = self._core._operate_on_patched_inputs(
                x, encoder_hidden_states, timestep, None
            )

        x_high = x_high.to(torch.float32) # Force FP32 for latent stability
        return x_high, encoder_hidden_states, ts, embedded_ts, class_labels, encoder_attention_mask, autocast_ctx

    def reasoning_core(self, x_high, y, z, encoder_hidden_states, ts, class_labels, encoder_attention_mask, autocast_ctx):
        y_final_high, y_next, z_next = self._deep_recursion(
            x_high, y, z, encoder_hidden_states, ts, class_labels, None, encoder_attention_mask, None, autocast_ctx
        )
        return y_final_high, y_next, z_next

    def decode_features(self, y_final_high, ts, class_labels, embedded_ts, autocast_ctx):
        with autocast_ctx:
            with self._unwrap_first_block(self._core):
                y_final_4ch = self._core._get_output_for_patched_inputs(
                    hidden_states=y_final_high,
                    timestep=ts,
                    class_labels=class_labels,
                    embedded_timestep=embedded_ts,
                    height=self.h_p,
                    width=self.w_p,
                )
        return y_final_4ch

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
        y, z = self.get_initial_states(bsz)

        # If train.py passed the sequence mask into attention_mask, reroute it to cross-attention.
        if attention_mask is not None and encoder_attention_mask is None:
            encoder_attention_mask = attention_mask
            attention_mask = None

        conditions = class_labels if class_labels is not None else encoder_hidden_states

        # 1. ENCODE (Raz na krok dyfuzji)
        x_high, enc_hs, ts, embedded_ts, class_labels, enc_mask, autocast_ctx = self.encode_features(
            sample, timestep, conditions, encoder_attention_mask
        )

        # 2. REASONING LOOP (Głęboki nadzór w trybie inferencji)
        for _ in range(self.n_sup):
            y_final_high, y, z = self.reasoning_core(
                x_high, y, z, enc_hs, ts, class_labels, enc_mask, autocast_ctx
            )

        # 3. DECODE (Raz na krok dyfuzji)
        model_output = self.decode_features(
            y_final_high, ts, class_labels, embedded_ts, autocast_ctx
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
                z_out = self._dit_blocks(
                    self._core, z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=-1))
            y_out = self._dit_blocks(
                self._core, y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
            )
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
                out = self._dit_blocks(
                    self._core, z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                _, z_out = self.out_proj(out).chunk(2, dim=-1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_in = self.fusion(torch.cat([zeros, y, z], dim=-1))
            out = self._dit_blocks(
                self._core, y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
            )
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
                out = self._dit_blocks(
                    self._core, z_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                _, z_out = self.out_proj(out).chunk(2, dim=-1)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            # x_high is passed instead of zeros!
            y_in = self.fusion(torch.cat([x_high, y, z], dim=-1))
            out = self._dit_blocks(
                self._core, y_in, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
            )
            y_out, _ = self.out_proj(out).chunk(2, dim=-1)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z


class BaseRatatouilleUNet(ExtraModulesMixin, BaseIterativeStrategy):
    """
    Abstract Base Class for the 'Thinker-Painter' Architecture.
    Remy (Thinker) does the reasoning; Linguini (Painter/core_model) renders.
    """

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        if not isinstance(core_model, (UNet2DConditionModel, UNet2DModel)) or not isinstance(
            thinker_model, (UNet2DConditionModel, UNet2DModel)
        ):
            raise ValueError("Ratatouille architecture requires both models to be UNets.")

        self.thinker_model = thinker_model
        self.painter_dim = core_model.conv_in.out_channels
        self.thinker_dim = thinker_model.conv_in.out_channels

        low_res = resolution // downsample_factor
        state_shape = (self.thinker_dim, low_res, low_res)

        super().__init__(core_model, state_shape, n, T, n_sup)

        self.encoder = SpatialEncoder(
            in_channels=core_model.config.in_channels, out_channels=self.thinker_dim, factor=downsample_factor
        )
        self.decoder = AttentiveBridge(
            in_channels=self.thinker_dim,
            out_channels=self.thinker_dim,
            out_resolution=resolution,
            factor=downsample_factor,
        )

        self.norm_y = nn.GroupNorm(32, self.thinker_dim)
        self.norm_z = nn.GroupNorm(32, self.thinker_dim)

    @property
    def _extra_modules(self):
        return ["thinker_model", "encoder", "decoder", "norm_y", "norm_z"]

    @contextlib.contextmanager
    def bypass_projections(self, model):
        core = model.module if hasattr(model, "module") else model
        proj_in, proj_out = core.conv_in, core.conv_out
        core.conv_in, core.conv_out = nn.Identity(), nn.Identity()
        try:
            yield proj_in, proj_out
        finally:
            core.conv_in, core.conv_out = proj_in, proj_out

    def _latent_recursion(self, x_low, y, z, timesteps, conditions, masks, autocast_ctx):
        for _ in range(self.n):
            with autocast_ctx:
                z_out = get_model_output(self.thinker_model, x_low + y + z, timesteps, conditions, masks)
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_out = get_model_output(self.thinker_model, y + z, timesteps, conditions, masks)
            y_out = self.norm_y(y_out)
        y = y_out.to(torch.float32)
        return y, z

    def _deep_recursion(self, x_low, y, z, timesteps, conditions, masks, autocast_ctx):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x_low, y, z, timesteps, conditions, masks, autocast_ctx)
        y_final, z_final = self._latent_recursion(x_low, y, z, timesteps, conditions, masks, autocast_ctx)
        return y_final, y_final.detach(), z_final.detach()

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, dtype=dtype)
            if dtype in [torch.float16, torch.bfloat16]
            else contextlib.nullcontext()
        )

        # Phase A: Remy Thinks (Operates purely in compressed space)
        with autocast_ctx:
            x_for_thinker = self.encoder(x)

        with self.bypass_projections(self.thinker_model) as (proj_in, _):
            with autocast_ctx:
                x_low = proj_in(x_for_thinker)
            x_low = x_low.to(torch.float32)
            y_final_low, y_next, z_next = self._deep_recursion(x_low, y, z, timesteps, conditions, masks, autocast_ctx)

        # Phase B: Linguini Paints (Delegated to Subclasses)
        with autocast_ctx:
            y_final_high = self.decoder(y_final_low)
            painter_out = self._render_painting(x, y_final_high, timesteps)

        return painter_out, y_next, z_next

    def __call__(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        """
        Inference loop hook for standard Diffusers pipelines.
        Automatically unrolls the supervision loops and returns the final painting.
        """
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)

        conditions = class_labels if class_labels is not None else encoder_hidden_states

        painter_out = None
        for _ in range(self.n_sup):
            painter_out, y, z = self.reasoning_step(sample, y, z, timestep, conditions, attention_mask)

        return TRMOutput(sample=painter_out)

    def _render_painting(self, x_high, y_final_high, timesteps):
        """To be implemented by subclasses (Concat vs ControlNet)"""
        raise NotImplementedError


class RatatouilleUNetConcat(BaseRatatouilleUNet):
    """Linguini is controlled via simple Input Concatenation."""

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, thinker_model, resolution, downsample_factor, n, T, n_sup, **kwargs)
        # Re-initialize the encoder to handle the channel offset caused by concatenation
        noise_channels = core_model.config.in_channels - self.thinker_dim
        self.encoder = SpatialEncoder(noise_channels, self.thinker_dim, factor=downsample_factor)

    def _render_painting(self, x_high, y_final_high, timesteps):
        painter_input = torch.cat([x_high, y_final_high], dim=1)
        # Painter must be completely blind to conditions/text
        return get_model_output(self.core_model, painter_input, timesteps, conditions=None, masks=None)


class RatatouilleUNetControl(BaseRatatouilleUNet):
    """Linguini is controlled via deep ControlNet-style residual injections."""

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, thinker_model, resolution, downsample_factor, n, T, n_sup, **kwargs)

        # Add the ControlNet pyramid
        painter_blocks = core_model.config.block_out_channels
        painter_layers = core_model.config.layers_per_block
        self.control_pyramid = ConditioningPyramid(
            self.thinker_dim, block_out_channels=painter_blocks, layers_per_block=painter_layers
        )

    @property
    def _extra_modules(self):
        # We must extend the parent's extra_modules to include the pyramid
        base_modules = super()._extra_modules
        return base_modules + ["control_pyramid"]

    def _render_painting(self, x_high, y_final_high, timesteps):
        down_block_res, mid_block_res = self.control_pyramid(y_final_high)

        dummy_context = torch.zeros((x_high.shape[0], 1, 1), device=x_high.device, dtype=x_high.dtype)

        return self.core_model(
            x_high,
            timesteps,
            encoder_hidden_states=dummy_context,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
        ).sample


class BaseRatatouilleDiT(DiTUtilsMixin, ExtraModulesMixin, BaseIterativeStrategy):
    """Abstract Base Class for Thinker-Painter Architecture using DiTs."""

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        if not isinstance(core_model, Transformer2DModel) or not isinstance(thinker_model, Transformer2DModel):
            raise ValueError("RatatouilleDiT requires both models to be Transformer2DModels.")

        self.thinker_model = thinker_model
        self.painter_dim = core_model.config.in_channels
        self.thinker_dim = thinker_model.config.in_channels
        self.painter_hidden_dim = core_model.config.num_attention_heads * core_model.config.attention_head_dim
        self.thinker_hidden_dim = thinker_model.config.num_attention_heads * thinker_model.config.attention_head_dim

        low_res = resolution // downsample_factor
        patch_size = getattr(thinker_model.config, "patch_size", 1)
        self.h_p = low_res // patch_size
        self.w_p = low_res // patch_size

        state_shape = (self.h_p * self.w_p, self.thinker_hidden_dim)
        super().__init__(core_model, state_shape, n, T, n_sup)

        self.encoder = SpatialEncoder(self.painter_dim, self.thinker_dim, factor=downsample_factor)
        self.decoder = AttentiveBridge(self.thinker_dim, self.thinker_dim, resolution, factor=downsample_factor)
        self.norm_y = nn.LayerNorm(self.thinker_hidden_dim)
        self.norm_z = nn.LayerNorm(self.thinker_hidden_dim)

    @property
    def _extra_modules(self):
        return ["encoder", "decoder", "norm_y", "norm_z"]

    @property
    def _core(self):
        return self.core_model.module if hasattr(self.core_model, "module") else self.core_model

    @property
    def _thinker(self):
        return self.thinker_model.module if hasattr(self.thinker_model, "module") else self.thinker_model

    # --- DDP UNROLLING ---
    def get_trainable_modules(self):
        modules = super().get_trainable_modules()
        if "core_model" in modules:
            del modules["core_model"]
        if "thinker_model" in modules:
            del modules["thinker_model"]

        # Magically fetch everything using the Mixin
        modules.update(self._get_dit_modules(self._core, prefix="core_"))
        modules.update(self._get_dit_modules(self._thinker, prefix="thinker_"))
        return modules

    def update_modules(self, prepared_modules):
        super().update_modules(prepared_modules)
        # Re-inject DDP wrappers
        self._update_dit_modules(self._core, prepared_modules, prefix="core_")
        self._update_dit_modules(self._thinker, prepared_modules, prefix="thinker_")

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        original_modules = {}

        for name, module in self.get_trainable_modules().items():
            if name in self._extra_modules:
                m_unwrap = unwrap_model(module)
                torch.save(m_unwrap.state_dict(), os.path.join(output_dir, f"{name}.pt"))
            else:
                original_modules[name] = module

        unwrapped_modules = {k: unwrap_model(v) for k, v in original_modules.items()}

        # Inject unwrapped layers back into Diffusers so it can save cleanly
        self._update_dit_modules(self._core, unwrapped_modules, prefix="core_")
        self._update_dit_modules(self._thinker, unwrapped_modules, prefix="thinker_")

        self._core.save_pretrained(os.path.join(output_dir, "unet"))
        self._thinker.save_pretrained(os.path.join(output_dir, "thinker_model"))
        torch.save({"y_init": self.y_init, "z_init": self.z_init}, os.path.join(output_dir, "strategy_state.pt"))

        # Restore DDP wrappers to continue training
        self._update_dit_modules(self._core, original_modules, prefix="core_")
        self._update_dit_modules(self._thinker, original_modules, prefix="thinker_")

    def load(self, input_dir):
        super().load(input_dir)
        thinker_dir = os.path.join(input_dir, "thinker_model")
        if os.path.exists(thinker_dir):
            sf_path = os.path.join(thinker_dir, "diffusion_pytorch_model.safetensors")
            bin_path = os.path.join(thinker_dir, "diffusion_pytorch_model.bin")
            state_dict = load_file(sf_path) if os.path.exists(sf_path) else torch.load(bin_path, map_location="cpu")
            load_with_backward_compatibility(self._thinker, state_dict)

    # --- BPTT LOOP ---
    def _latent_recursion(
        self, x_high, y, z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs, autocast_ctx
    ):
        for _ in range(self.n):
            with autocast_ctx:
                z_out = self._dit_blocks(
                    self._thinker, x_high + y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
                )
                z_out = self.norm_z(z_out)
            z = z_out.to(torch.float32)

        with autocast_ctx:
            y_out = self._dit_blocks(
                self._thinker, y + z, encoder_hs, ts, class_labels, attn_mask, enc_attn_mask, cross_kwargs
            )
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

        with autocast_ctx:
            x_for_thinker = self.encoder(x)

        encoder_hs, class_labels = self._prepare_conditions(conditions, bs=bsz, device=x.device, model=self._thinker)
        enc_attn_mask = masks
        if enc_attn_mask is not None and enc_attn_mask.ndim == 2:
            enc_attn_mask = (1 - enc_attn_mask.to(x.dtype)) * -10000.0
            enc_attn_mask = enc_attn_mask.unsqueeze(1)

        with autocast_ctx:
            x_low, encoder_hs, ts, embedded_ts = self._thinker._operate_on_patched_inputs(
                x_for_thinker, encoder_hs, timestep, None
            )
        x_low = x_low.to(torch.float32)

        y_final_low_tokens, y_next, z_next = self._deep_recursion(
            x_low, y, z, encoder_hs, ts, class_labels, None, enc_attn_mask, None, autocast_ctx
        )

        with autocast_ctx:
            with self._unwrap_first_block(self._thinker):
                y_final_low_2d = self._thinker._get_output_for_patched_inputs(
                    hidden_states=y_final_low_tokens,
                    timestep=ts,
                    class_labels=class_labels,
                    embedded_timestep=embedded_ts,
                    height=self.h_p,
                    width=self.w_p,
                )

            y_final_high_2d = self.decoder(y_final_low_2d)
            painter_out = self._render_painting(x, y_final_high_2d, timesteps)

        return painter_out, y_next, z_next

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
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)
        conditions = class_labels if class_labels is not None else encoder_hidden_states

        if attention_mask is not None and encoder_attention_mask is None:
            encoder_attention_mask = attention_mask

        painter_out = None
        for _ in range(self.n_sup):
            painter_out, y, z = self.reasoning_step(sample, y, z, timestep, conditions, encoder_attention_mask)

        return TRMOutput(sample=painter_out)

    def _render_painting(self, x_high, y_final_high, timesteps):
        raise NotImplementedError


class RatatouilleDiTConcat(BaseRatatouilleDiT):
    """Linguini is controlled via simple Input Concatenation."""

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, thinker_model, resolution, downsample_factor, n, T, n_sup, **kwargs)
        noise_channels = core_model.config.in_channels - self.thinker_dim
        self.encoder = SpatialEncoder(noise_channels, self.thinker_dim, factor=downsample_factor)

    def _render_painting(self, x_high, y_final_high, timesteps):
        painter_input = torch.cat([x_high, y_final_high], dim=1)
        # Linguini is blind. get_model_output will handle default class labels if needed!
        return get_model_output(self.core_model, painter_input, timesteps, conditions=None, masks=None)


class RatatouilleDiTResidual(BaseRatatouilleDiT):
    """Linguini is controlled via IP-Adapter style direct Token Addition."""

    def __init__(self, core_model, thinker_model, resolution, downsample_factor=4, n=6, T=3, n_sup=1, **kwargs):
        super().__init__(core_model, thinker_model, resolution, downsample_factor, n, T, n_sup, **kwargs)

        patch_size = getattr(core_model.config, "patch_size", 1)

        self.blueprint_proj = nn.Conv2d(
            self.thinker_dim, self.painter_hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    @property
    def _extra_modules(self):
        return super()._extra_modules + ["blueprint_proj"]

    def _render_painting(self, x_high, y_final_high, timesteps):
        # 1. Project blueprint to Linguini's hidden dim and flatten to a sequence of tokens
        blueprint_tokens = self.blueprint_proj(y_final_high).flatten(2).transpose(1, 2)

        bsz = x_high.shape[0]
        timestep = self._format_timestep(timesteps, bsz, x_high.device)

        # Linguini is blind to real text, so we generate dummy conditions for AdaLN
        encoder_hs, class_labels = self._prepare_conditions(None, bs=bsz, device=x_high.device, model=self._core)

        # 2. Patchify the standard noisy image
        hidden_states, encoder_hs, ts, embedded_ts = self._core._operate_on_patched_inputs(
            x_high, encoder_hs, timestep, None
        )

        # 3. T2I-Adapter Injection: Add the blueprint directly to the noisy image tokens!
        hidden_states = hidden_states + blueprint_tokens

        # 4. Run the deep ViT blocks
        hidden_states = self._dit_blocks(self._core, hidden_states, encoder_hs, ts, class_labels, None, None, None)

        # 5. Unpatchify back to a standard image
        with self._unwrap_first_block(self._core):
            out = self._core._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=ts,
                class_labels=class_labels,
                embedded_timestep=embedded_ts,
                height=x_high.shape[2] // self._core.config.patch_size,
                width=x_high.shape[3] // self._core.config.patch_size,
            )
        return out
