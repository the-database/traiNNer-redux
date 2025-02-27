import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

from traiNNer.utils.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class AdanScheduleFree(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float = 0.0,
        warmup_steps: int = 1600,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ) -> None:
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_steps": warmup_steps,
            "r": r,
            "weight_lr_power": weight_lr_power,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "train_mode": True,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def restart_opt(self) -> None:
        for group in self.param_groups:
            group["step"] = 0
            group["weight_sum"] = 0.0
            group["lr_max"] = -1.0
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state["exp_avg_diff"] = torch.zeros_like(p)

                    state["z"] = p.clone(memory_format=torch.preserve_format)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                beta1, _, _ = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                beta1, _, _ = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        global_grad_norm.add_(p.grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + self.param_groups[-1]["eps"]),
                max=1.0,
            ).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            neg_pre_grads = []
            z = []

            beta1, beta2, beta3 = group["betas"]
            group["step"] = group.get("step", 0) + 1
            step = group["step"]

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            bias_correction3 = 1.0 - beta3**step

            warmup_steps = group["warmup_steps"]
            if warmup_steps > 0 and step < warmup_steps:
                sched = step / warmup_steps
            else:
                sched = 1.0
            effective_lr = group["lr"] * sched
            group["lr_max"] = max(effective_lr, group["lr_max"])
            weight = (step ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"]
            )
            group["weight_sum"] += weight
            try:
                ckp1 = weight / group["weight_sum"]
            except ZeroDivisionError:
                ckp1 = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["z"] = p.clone(memory_format=torch.preserve_format)

                if "neg_pre_grad" not in state or step == 1:
                    state["neg_pre_grad"] = p.grad.clone().mul_(-clip_global_grad_norm)

                z.append(state["z"])
                neg_pre_grads.append(state["neg_pre_grad"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                exp_avg_diffs.append(state["exp_avg_diff"])

            if not params_with_grad:
                continue

            _multi_tensor_adan_schedule_free(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                neg_pre_grads=neg_pre_grads,
                z=z,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                clip_global_grad_norm=clip_global_grad_norm,
                ckp1=ckp1,
            )

        return loss


def _multi_tensor_adan_schedule_free(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor],
    neg_pre_grads: list[Tensor],
    z: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    clip_global_grad_norm: float,
    ckp1: float,
) -> None:
    if len(params) == 0:
        return

    torch._foreach_mul_(grads, clip_global_grad_norm)

    # for memory saving, we use `neg_pre_grads`
    # to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(
        exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3
    )  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)  # noqa: SLF001
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size = lr / bias_correction1 * (1 - ckp1)
    step_size_diff = lr * beta2 / bias_correction2 * (1 - ckp1)

    torch._foreach_lerp_(params, z, weight=ckp1)

    torch._foreach_mul_(params, 1 - lr * weight_decay)
    torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
    torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)

    # z step
    torch._foreach_mul_(z, 1 - lr * weight_decay)
    torch._foreach_addcdiv_(z, exp_avgs, denom, value=-lr / bias_correction1)
    torch._foreach_addcdiv_(
        z, exp_avg_diffs, denom, value=-lr * beta2 / bias_correction2
    )

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
