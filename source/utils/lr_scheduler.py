"""LR schedule helpers (missing from the code release).

`configs/model/crag_unified.yaml` names
`LambdaWarmUpCosineFactorScheduler` as the model `lr_scheduler`. The instance
is used as `lr_lambda` in `torch.optim.lr_scheduler.LambdaLR` (see
`source/models/crag_unified.py`), so it is a callable mapping the current
training step to a multiplicative LR factor: linear warm-up from `f_start`
to `f_max`, then cosine decay to `f_min` over `max_decay_steps`.
"""

import math


class LambdaWarmUpCosineFactorScheduler:
    def __init__(
        self,
        optimizer=None,
        max_decay_steps: int = 200000,
        warm_up_steps: int = 1000,
        f_start: float = 1e-6,
        f_min: float = 1e-3,
        f_max: float = 1.0,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.max_decay_steps = max_decay_steps
        self.warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, step: int) -> float:
        if self.warm_up_steps > 0 and step < self.warm_up_steps:
            return self.f_start + (self.f_max - self.f_start) * step / self.warm_up_steps
        progress = (step - self.warm_up_steps) / max(self.max_decay_steps - self.warm_up_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.f_min + 0.5 * (self.f_max - self.f_min) * (1.0 + math.cos(math.pi * progress))
