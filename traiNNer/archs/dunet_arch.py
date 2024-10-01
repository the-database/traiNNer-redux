from spandrel.architectures.__arch_helpers.dysample import DySample
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


class Down(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(spectral_norm(nn.Conv2d(dim, dim * 2, 3, 2, 1)), nn.Mish(True))


class Up(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            DySample(dim, dim, 2, 4, False),
            spectral_norm(nn.Conv2d(dim, dim // 2, 3, 1, 1)),
        )


@ARCH_REGISTRY.register()
class DUnet(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64) -> None:
        super().__init__()
        self.in_to_dim = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # encode x
        self.e_x1 = Down(num_feat)
        self.e_x2 = Down(num_feat * 2)
        self.e_x3 = Down(num_feat * 4)
        # up
        self.up1 = Up(num_feat * 8)
        self.up2 = Up(num_feat * 4)
        self.up3 = Up(num_feat * 2)
        # end conv
        self.end_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.Mish(True),
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.Mish(True),
            nn.Conv2d(num_feat, 1, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.in_to_dim(x)
        x1 = self.e_x1(x0)
        x2 = self.e_x2(x1)
        x3 = self.e_x3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.up3(x) + x0
        return self.end_conv(x)
