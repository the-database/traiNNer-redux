# https://github.com/psyrocloud/MS-SSIM_L1_LOSS/blob/main/MS_SSIM_L1_loss.py
"""
Created on Thu Dec  3 00:28:15 2020

@author: Yunpeng Li, Tianjin University
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MSSSIML1Loss(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(
        self,
        gaussian_sigmas: list[float] | None = None,
        data_range: float = 1.0,
        k: tuple[float, float] = (0.01, 0.03),
        alpha: float = 0.025,
        compensation: float = 200.0,
        cuda_dev: int = 0,
    ) -> None:
        if gaussian_sigmas is None:
            gaussian_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
        super().__init__()
        self.DR = data_range
        self.C1 = (k[0] * data_range) ** 2
        self.C2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size: int, sigma: float) -> Tensor:
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size: int, sigma: float) -> Tensor:
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lm = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        pics = cs.prod(dim=1)

        loss_ms_ssim = 1 - lm * pics  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction="none")  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-3, length=3),
            groups=3,
            padding=self.pad,
        ).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
