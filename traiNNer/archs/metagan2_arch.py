from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


def sconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: str | int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> nn.Module:
    return spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
    )


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
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
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class DConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = InceptionDWConv2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
        self, dim: int, expansion_ratio: float = 8 / 3, conv_ratio: float = 1.0
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = spectral_norm(nn.Linear(dim, hidden * 2))
        self.act = nn.SiLU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = DConv(conv_channels)
        self.fc2 = spectral_norm(nn.Linear(hidden, dim))
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1)) * self.gamma
        return x + shortcut


class Stem(nn.Module):
    r"""Code modified from InternImage:
    https://github.com/OpenGVLab/InternImage
    """

    def __init__(
        self,
        in_chs: int = 3,
        out_chs: int = 96,
    ) -> None:
        super().__init__()
        self.conv1 = sconv(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1)
        self.act = nn.SiLU()
        self.conv2 = sconv(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.RMSNorm(out_chs, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleNormFirst(nn.Module):
    def __init__(
        self,
        in_chs: int = 96,
        out_chs: int = 198,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(in_chs)
        self.conv = sconv(in_chs, out_chs, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Blocks(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, blocks: int, scale: int) -> None:
        super().__init__()
        self.down = (
            DownsampleNormFirst(in_dim, out_dim)
            if scale == 2
            else Stem(in_dim, out_dim)
        )
        self.blocks = nn.Sequential(*[GatedCNNBlock(out_dim) for _ in range(blocks)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.blocks(x)
        return x


@ARCH_REGISTRY.register()
class MetaGan2(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 1,
        dims: Sequence[int] = (32, 64, 128, 192),
        blocks: Sequence[int] = (3, 3, 15, 3),
        downs: Sequence[int] = (4, 2, 2, 2),
    ) -> None:
        super().__init__()
        dims = [in_ch, *list(dims)]
        self.stages = nn.Sequential(
            *[
                Blocks(
                    dims[index],
                    dims[index + 1],
                    blocks[index],
                    downs[index],
                )
                for index in range(len(blocks))
            ]
        )
        self.head = nn.Sequential(
            *[
                spectral_norm(nn.Linear(dims[-1], dims[-1] * 4)),
                nn.Mish(True),
                nn.Linear(dims[-1] * 4, dims[-1]),
            ]
        )

    def perceptual(self, x: Tensor) -> list[Tensor]:
        list_dict = []
        for layer in self.stages:
            x = layer(x)
            list_dict.append(x)
        list_dict.append(self.head(x))
        return list_dict

    def forward(self, x: Tensor) -> Tensor:
        x = self.stages(x)
        return self.head(x)
