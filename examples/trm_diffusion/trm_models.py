import torch
import torch.nn as nn
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from trm_utils import deep_recursion, get_model_output


@dataclass
class TRMOutput(BaseOutput):
    """Simple dataclass to mimic Diffusers model outputs."""

    sample: torch.FloatTensor


class BaseIterativeModel(nn.Module):
    """
    Abstract base class that handles state initialization and recursive hyperparameters.
    """

    def __init__(self, core_model, y_channels, z_channels, resolution, n=6, T=3, n_sup=1):
        super().__init__()
        self.core_model = core_model
        self.n = n
        self.T = T
        self.n_sup = n_sup

        # 1. INJECT INTO DIFFUSERS CONFIG:
        # We stamp our TRM arguments into the core model's config so they are safely saved in config.json!
        if hasattr(self.core_model, "register_to_config"):
            self.core_model.register_to_config(
                trm_n=n,
                trm_T=T,
                trm_n_sup=n_sup,
                trm_y_channels=y_channels,
                trm_z_channels=z_channels,
                trm_resolution=resolution,
            )

        # 2. REGISTER BUFFERS:
        # We attach these to the core model so they save seamlessly into the .safetensors file
        self.core_model.register_buffer("y_init", torch.randn(1, y_channels, resolution, resolution))
        self.core_model.register_buffer("z_init", torch.randn(1, z_channels, resolution, resolution))

    # 3. FIX THE DEVICE ERROR:
    # Safely route device requests to the underlying model
    @property
    def device(self):
        return self.core_model.device

    def get_initial_states(self, batch_size):
        """Returns cloned, batch-expanded initial states."""
        y = self.core_model.y_init.expand(batch_size, -1, -1, -1).clone()
        z = self.core_model.z_init.expand(batch_size, -1, -1, -1).clone()
        return y, z

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        raise NotImplementedError

    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, attention_mask=None, **kwargs):
        """
        Standard forward pass for Diffusers pipelines (used in evaluation/inference).
        """
        bsz = sample.shape[0]
        y, z = self.get_initial_states(bsz)
        y, z = y.to(sample.device), z.to(sample.device)

        conditions = class_labels if class_labels is not None else encoder_hidden_states

        model_output = None
        for _ in range(self.n_sup):
            model_output, y, z = self.reasoning_step(sample, y, z, timestep, conditions, attention_mask)

        return TRMOutput(sample=model_output)


class StandardTRM(BaseIterativeModel):
    """
    Your original Tiny Recursive Model.
    """

    def __init__(self, core_model, state_channels, resolution, n=6, T=3, n_sup=1):
        super().__init__(core_model, state_channels, state_channels, resolution, n, T, n_sup)

    def reasoning_step(self, x, y, z, timesteps, conditions=None, masks=None):
        model_output, y_next, z_next = deep_recursion(
            self.core_model, x, y, z, timesteps, conditions, masks, self.n, self.T
        )
        return model_output, y_next, z_next
