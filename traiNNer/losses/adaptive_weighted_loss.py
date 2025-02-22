import numpy as np
import torch


class aw_method:
    def __init__(
        self, alpha1=0.5, alpha2=0.75, delta=0.05, epsilon=0.05, normalized_aw=True
    ) -> None:
        assert alpha1 < alpha2
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._delta = delta
        self._epsilon = epsilon
        self._normalized_aw = normalized_aw

    def aw_loss(
        self, Dloss_real, Dloss_fake, Dis_opt, Dis_Net, real_validity, fake_validity
    ):
        # resetting gradient back to zero
        Dis_opt.zero_grad()

        # computing real batch gradient
        Dloss_real.backward(retain_graph=True)
        # tensor with real gradients
        grad_real_tensor = [
            param.grad.clone() for _, param in Dis_Net.named_parameters()
        ]
        grad_real_list = torch.cat(
            [grad.reshape(-1) for grad in grad_real_tensor], dim=0
        )
        # calculating the norm of the real gradient
        rdotr = (
            torch.dot(grad_real_list, grad_real_list).item() + 1e-4
        )  # 1e-4 added to avoid division by zero
        r_norm = np.sqrt(rdotr)

        # resetting gradient back to zero
        Dis_opt.zero_grad()

        # computing fake batch gradient
        Dloss_fake.backward()  # (retain_graph=True)
        # tensor with real gradients
        grad_fake_tensor = [
            param.grad.clone() for _, param in Dis_Net.named_parameters()
        ]
        grad_fake_list = torch.cat(
            [grad.reshape(-1) for grad in grad_fake_tensor], dim=0
        )
        # calculating the norm of the fake gradient
        fdotf = (
            torch.dot(grad_fake_list, grad_fake_list).item() + 1e-4
        )  # 1e-4 added to avoid division by zero
        f_norm = np.sqrt(fdotf)

        # resetting gradient back to zero
        Dis_opt.zero_grad()

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
        aw_loss = w_r * Dloss_real + w_f * Dloss_fake

        # updating gradient, i.e. getting aw_loss gradient
        for index, (_, param) in enumerate(Dis_Net.named_parameters()):
            print(grad_real_tensor[index])
            print(grad_fake_tensor[index])
            param.grad = w_r * grad_real_tensor[index] + w_f * grad_fake_tensor[index]

        return aw_loss
