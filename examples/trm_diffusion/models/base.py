from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn
from accelerate import Accelerator

from models.optim_utils import ScheduledOptimizer
from models.trm.ema import EMAHelper

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        pass

    @abstractmethod
    def train_step(
        self,
        micro_batches: dict[str, Any],
        accelerator: Accelerator,
        optimizers: list[ScheduledOptimizer],
        ema: EMAHelper | None,
        global_batch_size: int,
        global_step: int,
        **kwargs,
    ) -> tuple[dict, float, int]:
        pass

    @abstractmethod
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        pass

    def compile_submodules(self):
        pass
