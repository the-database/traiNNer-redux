import math

from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class CosineAnnealingWarmupLR(LRScheduler):
    """Cosine annealing with linear warmup learning rate scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        total_iters (int): Total number of iterations.
        warmup_iters (int): Number of warmup iterations. Default: 0.
        eta_min_ratio (float): Minimum lr as ratio of initial lr. Default: 0.1.
            Final lr = initial_lr * eta_min_ratio.
        last_epoch (int): Used in LRScheduler. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        warmup_iters: int = 0,
        eta_min_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        if self.last_epoch < self.warmup_iters:
            # Linear warmup: 0 -> 1
            alpha = self.last_epoch / max(1, self.warmup_iters)
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            progress = min(1.0, progress)
            alpha = (
                self.eta_min_ratio
                + (1 - self.eta_min_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )

        return [base_lr * alpha for base_lr in self.base_lrs]
