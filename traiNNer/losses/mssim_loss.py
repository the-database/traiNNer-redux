from collections.abc import Sequence

import torch
from torch import SymInt, Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import CharbonnierLoss
from traiNNer.utils.registry import LOSS_REGISTRY

####################################
# Modified MSSIM Loss with cosine similarity from neosr
# https://github.com/muslll/neosr/blob/master/neosr/losses/ssim_loss.py
####################################


def smoothstep(x: Tensor, min: float = 0, max: float = 1) -> Tensor:
    t = torch.clamp((x - min) / (max - min), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class GaussianFilter2D(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        padding: int | SymInt | Sequence[int | SymInt] | None = None,
    ) -> None:
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(
            name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1)
        )

    def _get_gaussian_window1d(self) -> Tensor:
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d: Tensor) -> Tensor:
        w = torch.matmul(
            gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d
        )
        return w

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(
            input=x,
            weight=self.gaussian_window,
            stride=1,
            padding=self.padding,
            groups=x.shape[1],
        )
        return x


@LOSS_REGISTRY.register()
class MSSIMLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        l: int = 1,
        padding: int | SymInt | Sequence[int | SymInt] | None = None,
        cosim: bool = True,
        cosim_lambda: int = 5,
        loss_weight: float = 1.0,
    ) -> None:
        """Adapted from 'A better pytorch-based implementation for the mean structural
            similarity. Differentiable simpler SSIM and MS-SSIM.':
                https://github.com/lartpang/mssim.pytorch

            Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float): The sigma of the gaussian filter. Defaults to 1.5.
            k1 (float): k1 of MSSIM. Defaults to 0.01.
            k2 (float): k2 of MSSIM. Defaults to 0.03.
            L (int): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None,
                the filter will use window_size//2 as the padding. Another common setting is 0.
            cosim (bool): Enables CosineSimilary on final loss, to keep better color consistency.
            cosim_lambda (float): Lambda value to increase CosineSimilarity weight.
            loss_weight (float): Weight of final loss value.
        """
        super().__init__()

        self.window_size = window_size
        self.C1 = (k1 * l) ** 2  # equ 7 in ref1
        self.C2 = (k2 * l) ** 2  # equ 7 in ref1
        self.cosim = cosim
        self.cosim_lambda = cosim_lambda
        self.loss_weight = loss_weight
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.charbonnier = CharbonnierLoss()

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        charbonnier = 0
        charbonnier_weight = torch.mean(torch.abs(x - x.clamp(1e-12, 1))).clamp(0, 1)
        charbonnier_weight = smoothstep(charbonnier_weight, 0.1, 0.9)
        if charbonnier_weight > 0:
            charbonnier = self.charbonnier(x, y)
            if charbonnier_weight >= 1:  # skip mssim
                return charbonnier

        loss = 1 - self.msssim(x, y)

        loss = loss * (1 - charbonnier_weight) + charbonnier * charbonnier_weight

        if self.cosim:
            loss += self.cosim_penalty(x, y)

        return self.loss_weight * loss

    def msssim(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.clamp(x, 1e-12, 1)
        y = torch.clamp(y, 1e-12, 1)

        msssim = torch.tensor(1.0, device=x.device)

        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            ssim = ssim.mean()
            cs = cs.mean()

            if i == 4:
                msssim *= ssim**w
            else:
                msssim *= cs**w
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)

        return msssim

    def cosim_penalty(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.where(torch.abs(x) < 1e-12, 1e-12, x)
        y = torch.where(torch.abs(y) < 1e-12, 1e-12, y)

        distance = 1 - torch.round(self.similarity(x, y), decimals=20).mean()
        return self.cosim_lambda * distance

    def _ssim(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        a1 = 2 * mu_x * mu_y + self.C1
        a2 = 2 * sigma_xy + self.C2
        b1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        b2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l = a1 / b1
        cs = a2 / b2
        ssim = l * cs

        return ssim, cs
