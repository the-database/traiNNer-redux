# https://github.com/umzi2
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


def sconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Module:
    return spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


class InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class DilatedContextBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim
        )
        self.conv4 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=4, dilation=4, groups=dim
        )
        self.conv8 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=8, dilation=8, groups=dim
        )
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x2 = self.conv2(x)
        x4 = self.conv4(x)
        x8 = self.conv8(x)
        return self.fuse(torch.cat([x2, x4, x8], dim=1))


class ShiftConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(dim * 5, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(
            [
                x,
                F.pad(x[:, :, :, 1:], (0, 1)),
                F.pad(x[:, :, :, :-1], (1, 0)),
                F.pad(x[:, :, 1:, :], (0, 0, 0, 1)),
                F.pad(x[:, :, :-1, :], (0, 0, 1, 0)),
            ],
            dim=1,
        )
        return self.fuse(x)


class GatedConvBlock(nn.Module):
    def __init__(self, dim: int, conv_ratio: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        hidden = dim * 2
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden - conv_channels, conv_channels]
        self.conv1 = spectral_norm(nn.Conv2d(dim, hidden, 1))
        self.depthwise = InceptionDWConv2d(conv_channels)
        self.conv2 = spectral_norm(nn.Conv2d(hidden, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.conv1(x)
        i, c = torch.split(x, self.split_indices, dim=1)
        c = self.depthwise(c)
        x = self.conv2(torch.cat((i, c), dim=1))
        return shortcut + x


class DualPathBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local = GatedConvBlock(dim)
        self.global_context = DilatedContextBlock(dim)
        self.shift = ShiftConv(dim)
        self.fusion = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x_local = self.local(x)
        x_global = self.global_context(x)
        x_shift = self.shift(x)
        x = torch.cat([x_local, x_global, x_shift], dim=1)
        return self.fusion(x)


class Stem(nn.Module):
    def __init__(self, in_chs: int = 3, out_chs: int = 64) -> None:
        super().__init__()
        self.conv1 = sconv(in_chs, out_chs // 2, 3, stride=2, padding=1)
        self.act = nn.SiLU()
        self.conv2 = sconv(out_chs // 2, out_chs, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(1, out_chs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.norm(self.conv2(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, num_blocks: int) -> None:
        super().__init__()
        self.down = sconv(in_chs, out_chs, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            *[DualPathBlock(out_chs) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        return self.blocks(x)


@ARCH_REGISTRY.register()
class MetaGAN3(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dims: Sequence[int] = (64, 128, 192, 256),
        blocks: Sequence[int] = (2, 3, 5, 2),
    ) -> None:
        super().__init__()
        self.stem = Stem(in_ch, dims[0])
        self.stages = nn.Sequential(
            *[
                DownBlock(dims[i], dims[i + 1], blocks[i])
                for i in range(len(blocks) - 1)
            ]
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x
