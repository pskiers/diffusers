"""TEMPORARY local wrapper for running the interpretability ablations on a Mac (not part of the repo).

1. adam_atan2 (training-only optimizer) imports a CUDA backend that doesn't exist here: stub it.
2. MPS can't do adaptive_avg_pool2d when the input size isn't divisible by the output size
   (models/utility_models.py:216): run only that op on the CPU, everything else stays on the Mac GPU.

Usage: python run_local.py experiments/run_interpretability.py <hydra overrides...>
"""
import runpy
import sys
import types

sys.modules["adam_atan2_backend"] = types.ModuleType("adam_atan2_backend")

import torch.nn.functional as F

_adaptive_avg_pool2d = F.adaptive_avg_pool2d


def _pool_cpu_fallback(input, output_size):
    if input.device.type == "mps":
        return _adaptive_avg_pool2d(input.cpu(), output_size).to(input.device)
    return _adaptive_avg_pool2d(input, output_size)


F.adaptive_avg_pool2d = _pool_cpu_fallback

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
