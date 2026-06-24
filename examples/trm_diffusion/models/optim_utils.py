import math


class ScheduledOptimizer:
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, num_steps: int, min_ratio: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.min_ratio = min_ratio

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
