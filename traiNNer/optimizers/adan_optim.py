# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

from traiNNer.utils.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for
        Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay
            (default: False)
        foreach (bool): if True would use torch._foreach implementation.
            It's faster but uses slightly more memory. (default: True)
        fused (bool, optional): whether fused implementation is used.
            (default: False)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
    ) -> None:
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid Max grad norm: {max_grad_norm}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
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
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def restart_opt(self) -> None:
        for group in self.param_groups:
            group["step"] = 0
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

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Performs a single optimization step."""

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
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

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

            beta1, beta2, beta3 = group["betas"]
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]
            bias_correction3 = 1.0 - beta3 ** group["step"]

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

                if "neg_pre_grad" not in state or group["step"] == 1:
                    state["neg_pre_grad"] = p.grad.clone().mul_(-clip_global_grad_norm)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                exp_avg_diffs.append(state["exp_avg_diff"])
                neg_pre_grads.append(state["neg_pre_grad"])

            if not params_with_grad:
                continue

            kwargs = {
                "params": params_with_grad,
                "grads": grads,
                "exp_avgs": exp_avgs,
                "exp_avg_sqs": exp_avg_sqs,
                "exp_avg_diffs": exp_avg_diffs,
                "neg_pre_grads": neg_pre_grads,
                "beta1": beta1,
                "beta2": beta2,
                "beta3": beta3,
                "bias_correction1": bias_correction1,
                "bias_correction2": bias_correction2,
                "bias_correction3_sqrt": math.sqrt(bias_correction3),
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "eps": group["eps"],
                "clip_global_grad_norm": clip_global_grad_norm,
            }

            _multi_tensor_adan(**kwargs)  # pyright: ignore[reportArgumentType]

        return loss


def _multi_tensor_adan(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor],
    neg_pre_grads: list[Tensor],
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
    clip_global_grad_norm: Tensor,
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

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    torch._foreach_mul_(params, 1 - lr * weight_decay)
    torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
    torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
