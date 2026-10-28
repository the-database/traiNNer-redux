# from spandrel.architectures.SPAN import SPAN

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal, Self

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.util import store_hyperparameters
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


def _make_pair(value: Any) -> Any:
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size_t: tuple[int, int] = _make_pair(kernel_size)
    padding = (int((kernel_size_t[0] - 1) / 2), int((kernel_size_t[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def sequential(*args: nn.Module) -> nn.Module:
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):  # pyright: ignore[reportUnnecessaryIsInstance]
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3
) -> nn.Module:
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        gain2: int = 0,
        s: int = 1,
        bias: Literal[True] = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False  # pyright: ignore[reportOptionalMemberAccess]
        self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]
        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat.contiguous()
        self.eval_conv.bias.data = self.bias_concat.contiguous()  # pyright: ignore[reportOptionalMemberAccess]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


@store_hyperparameters()
class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        num_in_ch: int,
        num_out_ch: int,
        feature_channels: int = 48,
        num_blocks: int = 6,
        upscale: int = 4,
        bias: bool = True,
        norm: bool = True,
        img_range: float = 255.0,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
    ) -> None:
        super().__init__()

        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.img_range = img_range
        self.num_blocks = num_blocks

        self.mean: Tensor
        self.register_buffer(
            "mean", torch.Tensor(rgb_mean).view(1, 3, 1, 1), persistent=False
        )

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.conv_1 = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)
        for i in range(1, num_blocks + 1):
            setattr(self, f"block_{i}", SPAB(feature_channels, bias=bias))

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(
            feature_channels, self.out_channels, upscale_factor=upscale
        )

    @property
    def is_norm(self) -> bool:
        return self.no_norm is None

    def forward(self, x: Tensor) -> Tensor:
        if self.is_norm:
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out = out_feature
        out_b1 = out_feature
        last_mid = out_feature

        for i in range(1, self.num_blocks + 1):
            block: SPAB = getattr(self, f"block_{i}")
            out, mid, _att = block(out)
            if i == 1:
                out_b1 = out
            last_mid = mid

        out = self.conv_2(out)
        out = self.conv_cat(torch.cat([out_feature, out, out_b1, last_mid], 1))
        return self.upsampler(out)


@SPANDREL_REGISTRY.register()
def span(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 52,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@SPANDREL_REGISTRY.register()
def span_s(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 48,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_f32(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 32,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_f64(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 64,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_f96(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 96,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )
