import numpy as np
import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer


class AdaptiveWeightedDiscriminatorLoss:
    def __init__(
        self,
        alpha1: float = 0.5,
        alpha2: float = 0.75,
        delta: float = 0.05,
        epsilon: float = 0.05,
        normalized_aw: bool = True,
    ) -> None:
        assert alpha1 < alpha2
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._delta = delta
        self._epsilon = epsilon
        self._normalized_aw = normalized_aw

    def aw_loss(
        self,
        d_loss_real: Tensor,
        d_loss_fake: Tensor,
        d_optimizer: Optimizer,
        d_scaler: GradScaler,
        d_net: nn.Module,
        real_validity: Tensor,
        fake_validity: Tensor,
        device_type: str,
    ) -> Tensor:
        # resetting gradient back to zero
        d_optimizer.zero_grad()

        # computing real batch gradient
        with torch.autocast(enabled=False, device_type=device_type):
            d_scaler.scale(d_loss_real).backward(retain_graph=True)
        # tensor with real gradients
        grad_real_tensor = [param.grad.clone() for _, param in d_net.named_parameters()]
        grad_real_list = torch.cat(
            [grad.reshape(-1) for grad in grad_real_tensor], dim=0
        )
        # calculating the norm of the real gradient
        rdotr = (
            torch.dot(grad_real_list, grad_real_list).item() + 1e-4
        )  # 1e-4 added to avoid division by zero
        r_norm = np.sqrt(rdotr)

        # resetting gradient back to zero
        d_optimizer.zero_grad()

        # computing fake batch gradient
        with torch.autocast(enabled=False, device_type=device_type):
            d_scaler.scale(d_loss_fake).backward()  # (retain_graph=True)
        # tensor with real gradients
        grad_fake_tensor = [param.grad.clone() for _, param in d_net.named_parameters()]
        grad_fake_list = torch.cat(
            [grad.reshape(-1) for grad in grad_fake_tensor], dim=0
        )
        # calculating the norm of the fake gradient
        fdotf = (
            torch.dot(grad_fake_list, grad_fake_list).item() + 1e-4
        )  # 1e-4 added to avoid division by zero
        f_norm = np.sqrt(fdotf)

        # resetting gradient back to zero
        d_optimizer.zero_grad()

        # dot product between real and fake gradients
        rdotf = torch.dot(grad_real_list, grad_fake_list).item()
        fdotr = rdotf

        # Real and Fake scores
        rs = torch.mean(torch.sigmoid(real_validity))
        fs = torch.mean(torch.sigmoid(fake_validity))

        if self._normalized_aw:
            # Implementation of normalized version of aw-method, i.e. Algorithm 1
            if rs < self._alpha1 or rs < (fs - self._delta):
                if rdotf <= 0:
                    # Case 1:
                    w_r = (1 / r_norm) + self._epsilon
                    w_f = (-fdotr / (fdotf * r_norm)) + self._epsilon
                else:
                    # Case 2:
                    w_r = (1 / r_norm) + self._epsilon
                    w_f = self._epsilon
            elif rs > self._alpha2 and rs > (fs - self._delta):
                if rdotf <= 0:
                    # Case 3:
                    w_r = (-rdotf / (rdotr * f_norm)) + self._epsilon
                    w_f = (1 / f_norm) + self._epsilon
                else:
                    # Case 4:
                    w_r = self._epsilon
                    w_f = (1 / f_norm) + self._epsilon
            else:
                # Case 5
                w_r = (1 / r_norm) + self._epsilon
                w_f = (1 / f_norm) + self._epsilon
        # Implementation of non-normalized version of aw-method, i.e. Algorithm 2
        elif rs < self._alpha1 or rs < (fs - self._delta):
            if rdotf <= 0:
                # Case 1:
                w_r = 1 + self._epsilon
                w_f = (-fdotr / fdotf) + self._epsilon
            else:
                # Case 2:
                w_r = 1 + self._epsilon
                w_f = self._epsilon
        elif rs > self._alpha2 and rs > (fs - self._delta):
            if rdotf <= 0:
                # Case 3:
                w_r = (-rdotf / rdotr) + self._epsilon
                w_f = 1 + self._epsilon
            else:
                # Case 4:
                w_r = self._epsilon
                w_f = 1 + self._epsilon
        else:
            # Case 5
            w_r = 1 + self._epsilon
            w_f = 1 + self._epsilon

        # calculating aw_loss
        aw_loss = w_r * d_loss_real + w_f * d_loss_fake

        # updating gradient, i.e. getting aw_loss gradient
        for index, (_, param) in enumerate(d_net.named_parameters()):
            param.grad = w_r * grad_real_tensor[index] + w_f * grad_fake_tensor[index]

        return aw_loss
