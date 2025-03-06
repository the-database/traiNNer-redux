import math
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.architectures.__arch_helpers.dysample import DySample
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import _IncompatibleKeys

from traiNNer.utils.registry import ARCH_REGISTRY

SampleMods = Literal[
    "conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample"
]


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle and DySample
        group: int = 4,  # Only DySample
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
                dys_dim = mid_dim
            else:
                dys_dim = in_dim
            m.append(DySample(dys_dim, out_dim, scale, group))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    2,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self,
        in_channels: int = 64,
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.offset = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self, dim: int = 64, expansion_ratio: float = 8 / 3, rms_norm: bool = True
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim) if rms_norm else LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = dim
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x * self.gamma + shortcut


@ARCH_REGISTRY.register()
class MoSRv2(nn.Module):
    """Mamba Out Super-Resolution"""

    def __init__(
        self,
        in_ch: int = 3,
        scale: int = 4,
        n_block: int = 24,
        dim: int = 64,
        upsampler: SampleMods = "pixelshuffledirect",
        expansion_ratio: float = 1.5,
        mid_dim: int = 32,
        unshuffle_mod: bool = True,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        self.short = nn.Upsample(scale_factor=scale, mode="bilinear")
        self.scale = scale
        self.pad = 1
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            in_to_dim = [
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            ]
            self.pad = unshuffle
            scale = 4
        else:
            in_to_dim = [nn.Conv2d(in_ch, dim, 3, 1, 1)]

        self.gblocks = nn.Sequential(
            *in_to_dim
            + [
                GatedCNNBlock(
                    dim=dim, expansion_ratio=expansion_ratio, rms_norm=rms_norm
                )
                for _ in range(n_block)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.Mish(True),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Mish(True),
                nn.Conv2d(dim, dim, 1, 1),
            ]
        )
        self.to_img = UniUpsample(upsampler, scale, dim, in_ch, mid_dim)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        assert isinstance(state_dict, dict)
        state_dict["to_img.MetaUpsample"] = self.to_img.MetaUpsample
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, h, w)
        x = self.to_img(self.gblocks(x)) + self.short(x)
        return x[:, :, : h * self.scale, : w * self.scale]
