from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# https://github.com/nikhil-iyer-97/wide-minima-density-hypothesis
class KneeLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        peak_lr: float,
        total_steps: int,
        explore_ratio: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        assert 0.0 <= explore_ratio <= 1.0, "explore_ratio must be between 0 and 1."

        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.explore_ratio = explore_ratio

        self.explore_steps = int(total_steps * explore_ratio)
        self.decay_steps = self.total_steps - self.explore_steps

        assert self.decay_steps >= 0, (
            "Total steps must be at least equal to the number of exploration steps"
        )

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        current_step = self.last_epoch + 1

        if current_step <= self.explore_steps:
            lr = self.peak_lr
        else:
            step_in_decay = current_step - self.explore_steps
            lr = self.peak_lr - (self.peak_lr / self.decay_steps) * step_in_decay
            lr = max(lr, 0.0)

        return [lr for _ in self.optimizer.param_groups]
