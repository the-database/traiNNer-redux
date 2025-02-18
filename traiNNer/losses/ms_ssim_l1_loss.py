import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


@LOSS_REGISTRY.register()
class MSSSIML1Loss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        gaussian_sigmas: list[float] | None = None,
        data_range: float = 1.0,
        k: tuple[float, float] = (0.01, 0.03),
        alpha: float = 0.05,
    ) -> None:
        if gaussian_sigmas is None:
            gaussian_sigmas = [0.1, 1.0, 2.0, 4.0, 8.0]
        super().__init__()
        self.DR = data_range
        self.C1 = (k[0] * data_range) ** 2
        self.C2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.padt = (self.pad,) * 4
        self.alpha = alpha
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.register_buffer("_g_masks", g_masks)

        self.ms_weights = torch.tensor(MS_WEIGHTS).repeat_interleave(3).view(-1, 1, 1)
        self.loss_weight = loss_weight

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

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        g_masks: Tensor = torch.jit.annotate(Tensor, self._g_masks)
        mode = "replicate"
        ch = x.shape[-3]
        mux = F.conv2d(F.pad(x, self.padt, mode=mode), g_masks, groups=ch)
        muy = F.conv2d(F.pad(y, self.padt, mode=mode), g_masks, groups=ch)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = (
            F.conv2d(F.pad(x * x, self.padt, mode=mode), g_masks, groups=ch) - mux2
        )
        sigmay2 = (
            F.conv2d(F.pad(y * y, self.padt, mode=mode), g_masks, groups=ch) - muy2
        )
        sigmaxy = (
            F.conv2d(F.pad(x * y, self.padt, mode=mode), g_masks, groups=ch) - muxy
        )

        l = (2 * muxy[:, -3:, :, :] + self.C1) / (
            mux2[:, -3:, :, :] + muy2[:, -3:, :, :] + self.C1
        )  # [B, 3, H, W]
        lm = l.prod(dim=1)  # [B, H, W]
        cs = F.relu(
            (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        )  # [B, 15, H, W]

        cs_weighted = cs ** self.ms_weights.to(cs.device)  # [B, 15, H, W]
        pics = cs_weighted.prod(dim=1)  # [B, H, W]

        loss_ms_ssim = 1 - lm * pics  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction="none")  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(
            F.pad(loss_l1, self.padt, mode=mode),
            g_masks[-ch:],
            groups=ch,
        ).mean(1)  # [B, H, W]
        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR

        return self.loss_weight * loss_mix.mean()
