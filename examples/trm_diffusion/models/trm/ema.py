import copy
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def _tracked_tensors(self, module):
        """Trainable parameters + floating-point buffers (e.g. BatchNorm's
        running_mean/running_var) — excludes integer buffers like
        num_batches_tracked, which aren't meaningful to exponentially
        average. Without this, BatchNorm running stats never get the same
        long-window smoothing/eval-time snapshot as the weights do, and can
        drift to extreme values under transient training instability.

        A submodule's buffers are only tracked if that submodule's own
        direct parameters (if any) are trainable — mirrors the params-only
        `requires_grad` filter so a frozen submodule (e.g. the frozen
        painter in ThinkerFrozenPainterBase) doesn't get its buffers
        tracked either, even though buffers have no requires_grad of their
        own to check directly."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                yield name, param
        for mod_name, sub in module.named_modules():
            own_params = list(sub.parameters(recurse=False))
            if own_params and not any(p.requires_grad for p in own_params):
                continue
            for buf_name, buf in sub.named_buffers(recurse=False):
                if buf.is_floating_point():
                    yield (f"{mod_name}.{buf_name}" if mod_name else buf_name), buf

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, tensor in self._tracked_tensors(module):
            self.shadow[name] = tensor.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, tensor in self._tracked_tensors(module):
            if name not in self.shadow:
                # e.g. resuming from a checkpoint saved before buffers were
                # tracked — seed from the live value instead of KeyError.
                self.shadow[name] = tensor.data.clone()
                continue
            self.shadow[name].data = (1. - self.mu) * tensor.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, tensor in self._tracked_tensors(module):
            if name in self.shadow:
                tensor.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

