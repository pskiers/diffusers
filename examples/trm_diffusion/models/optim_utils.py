import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def all_reduce_grads(params: Iterable[torch.nn.Parameter], weight: Optional[float] = None) -> float:
    """Average the .grad of `params` across processes, in place.

    train_step always runs on the unwrapped model (train_trm.py,
    train_spatial.py), bypassing DDP's forward — and DDP only arms its
    gradient hooks inside its own forward, so without this each rank
    silently trains its own copy. Call after the last backward and before
    clip_grad_norm_, so clipping sees the global gradient. A plain mean is
    right because each rank backprops the mean loss over its own micro-batches
    (see BaseModel.train_step): the rank average is then the mean over the
    whole effective batch, the same gradient one GPU gets on it.

    Assumes every rank holds grads for the same params (true whenever all
    ranks run the same graph); a mismatch hangs the collective.

    weight: this rank's share of a weighted mean, for when ranks hold
    different amounts of data (see SpatialTRM.train_halt_head_ragged); None
    = plain mean. Returns the global total weight (world_size for a plain
    mean, no host sync) so a caller can skip its step when no rank had data.
    """
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return 1.0 if weight is None else float(weight)
    world_size = dist.get_world_size()
    grads = [p.grad for p in params if p.grad is not None]

    if weight is None:
        total = float(world_size)
    else:
        if grads:
            device = grads[0].device
        else:  # NCCL only takes CUDA tensors, gloo (CPU tests) only CPU ones
            device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
        total_t = torch.tensor([float(weight)], device=device)
        dist.all_reduce(total_t)
        total = total_t.item()

    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for g in grads:
        by_dtype.setdefault(g.dtype, []).append(g)
    for group in by_dtype.values():
        flat = _flatten_dense_tensors(group)
        flat.mul_(1.0 / world_size if weight is None else float(weight))
        dist.all_reduce(flat)
        if weight is not None:
            flat.div_(max(total, 1e-12))
        for g, synced in zip(group, _unflatten_dense_tensors(flat, group)):
            g.copy_(synced)
    return total


class ExponentialLRRamp:
    """LR range-test schedule (Smith 2017): lr grows geometrically from
    base_lr * start_factor at step 0 to base_lr * end_factor at num_steps, so
    every decade of lr gets the same number of steps. Set for all optimizers
    at once by train_trm.py's `lr_schedule` config (configs/lr_schedule/), which
    keeps their lr ratios."""

    def __init__(self, start_factor: float = 1e-2, end_factor: float = 1e2):
        self.start_factor = start_factor
        self.end_factor = end_factor

    def __call__(self, base_lr: float, step: int, num_steps: int) -> float:
        frac = min(step / max(1, num_steps), 1.0)
        return base_lr * self.start_factor * (self.end_factor / self.start_factor) ** frac


class ScheduledOptimizer:
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, num_steps: int, min_ratio: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.min_ratio = min_ratio
        # Replaces the warmup + cosine schedule when set: a callable
        # (base_lr, step, num_steps) -> lr, e.g. ExponentialLRRamp.
        self.schedule = None

    def step(self, global_step: int) -> float:
        lr = self.compute_lr(global_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.step()
        return lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)

    def compute_lr(self, current_step: int, num_cycles: float = 0.5):
        if self.schedule is not None:
            return self.schedule(self.base_lr, current_step, self.num_steps)
        if current_step < self.warmup_steps:
            return self.base_lr * float(current_step) / float(max(1, self.warmup_steps))

        progress = float(current_step - self.warmup_steps) / float(max(1, self.num_steps - self.warmup_steps))
        return self.base_lr * (
            self.min_ratio
            + max(0.0, (1 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        )


class DelayedScheduledOptimizer(ScheduledOptimizer):
    """
    ScheduledOptimizer with a hard-zero phase before the warmup begins.

    lr = 0 for the first `delay_steps` steps; after that the normal
    warmup+cosine schedule runs as if training started at step 0.

    Setting delay_steps=0 is identical to ScheduledOptimizer.
    Setting delay_steps to a very large number (e.g. 999999) keeps lr=0
    for the entire training run, effectively freezing the parameter group.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        num_steps: int,
        min_ratio: float = 0.0,
        delay_steps: int = 0,
    ):
        super().__init__(optimizer, base_lr, warmup_steps, num_steps, min_ratio)
        self.delay_steps = delay_steps

    def compute_lr(self, current_step: int, num_cycles: float = 0.5):
        if current_step < self.delay_steps:
            return 0.0
        return super().compute_lr(current_step - self.delay_steps, num_cycles)


def apply_lr_and_step(optimizers: list[ScheduledOptimizer], global_step: int):
    """Update LR for all optimizers, step them. Returns lr of last optimizer."""
    lr_now = None
    for opt in optimizers:
        lr_now = opt.step(global_step)
        opt.zero_grad()
    return lr_now
