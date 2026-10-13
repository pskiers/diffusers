"""Distributed setup, seeding, the learning-rate schedule, EMA updates and atomic saves."""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def init_distributed() -> torch.device:
    """Join the process group when launched by torchrun; return this rank's GPU."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


def learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup from 1e-6 to base_lr, then constant."""
    if step < warmup_steps:
        return 1e-6 + (step + 1) / warmup_steps * (base_lr - 1e-6)
    return base_lr


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = list(ema.parameters())
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, [p.detach() for p in model.parameters()], alpha=1.0 - decay)


def atomic_save(obj: object, path: Path) -> None:
    """torch.save through a temporary file, so a crash never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
