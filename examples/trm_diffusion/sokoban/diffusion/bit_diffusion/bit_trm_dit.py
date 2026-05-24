import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import contextlib
from dataclasses import dataclass
from diffusers.models import Transformer2DModel
from accelerate.utils import extract_model_from_parallel as unwrap_model
from safetensors.torch import load_file
import re


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
        self._load_with_backward_compatibility(core, state_dict)

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

    def _load_with_backward_compatibility(self, model, state_dict, logger=None, strict=True):
        """
        Translates old checkpoint keys to the new unified model architecture keys.
        If strict=True, raises a RuntimeError if there are missing or unexpected keys.
        """

        # Base rename map for removing DDP and strategy wrappers
        RENAME_MAP = {
            r"^module\.": "",
            r"^model\.": "",
        }

        # Only map legacy class/sequence embeddings to 'condition_projector' if
        # the target model actually uses the Unified architecture!
        if hasattr(model, "condition_projector"):
            RENAME_MAP[r"(^|\.)projector(\.)"] = r"\g<1>condition_projector\g<2>"
            RENAME_MAP[r"(^|\.)class_embedding(\.)"] = r"\g<1>condition_projector\g<2>"

        adapted_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in RENAME_MAP.items():
                new_key = re.sub(pattern, replacement, new_key)
            adapted_dict[new_key] = value

        # We use strict=False internally so we can format our own clean error messages
        missing, unexpected = model.load_state_dict(adapted_dict, strict=False)

        error_msgs = []

        if len(missing) > 0:
            msg = f"Missing keys when loading checkpoint (showing first 5): {missing[:5]} ... ({len(missing)} total)"
            error_msgs.append(msg)
            if logger is not None:
                logger.error(msg)
            else:
                print(msg)

        if len(unexpected) > 0:
            msg1 = f"Unexpected keys in checkpoint (showing first 5): {unexpected[:5]} ... ({len(unexpected)} total)"
            msg2 = "If these unexpected keys should map to the missing keys, add them to the RENAME_MAP!"
            error_msgs.extend([msg1, msg2])
            if logger is not None:
                logger.error(msg1)
                logger.error(msg2)
            else:
                print(msg1)
                print(msg2)

        # Crash the script if strict mode is enabled and there's a mismatch
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError("State dict mismatch detected! Strict mode is ON.\n" + "\n".join(error_msgs))



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

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
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
        x_high = x_high.to(torch.float32)  # Force state to remain FP32

        y_final_high, y_next, z_next = self._deep_recursion(
            x_high, y, z, encoder_hidden_states, ts, class_labels, None, encoder_attention_mask, None, autocast_ctx
        )

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
        encoder_hidden_states, class_labels = self._prepare_conditions(
            conditions, bs=bsz, device=sample.device, model=self._core
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
            with self._unwrap_first_block(self._core):
                model_output = self._core._get_output_for_patched_inputs(
                    hidden_states=y_final_high,
                    timestep=ts,
                    class_labels=class_labels,
                    embedded_timestep=embedded_ts,
                    height=self.h_p,
                    width=self.w_p,
                )

        return TRMOutput(sample=model_output)



