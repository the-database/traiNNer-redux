# from spandrel.architectures.SPAN import SPAN

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.util import store_hyperparameters
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


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


def activation(
    act_type: str, inplace: bool = True, neg_slope: float = 0.05, n_prelu: int = 1
) -> nn.Module:
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"activation layer [{act_type:s}] is not found")
    return layer


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
            for submodule in module.children():
                modules.append(submodule)
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

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
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
        self.act2 = activation("lrelu", neg_slope=0.1, inplace=True)

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
        upscale: int = 4,
        bias: bool = True,
        norm: bool = True,
        img_range: float = 255.0,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        learn_residual: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.scale = upscale

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.learn_residual: torch.Tensor | None
        if learn_residual:
            self.register_buffer("learn_residual", torch.zeros(1))
        else:
            self.learn_residual = None

        self.conv_1 = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(
            feature_channels, self.out_channels, upscale_factor=upscale
        )

        if learn_residual:
            for m in self.upsampler.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                    m.weight.data.mul_(1e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @property
    def is_norm(self) -> bool:
        return self.no_norm is None

    @property
    def is_learn_residual(self) -> bool:
        return self.learn_residual is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)

        out_b4, _, _att4 = self.block_4(out_b3)
        out_b5, _, _att5 = self.block_5(out_b4)
        out_b6, out_b5_2, _att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        if self.is_learn_residual:
            # add the bilinear upsampled image, so that the network learns the residual
            base = F.interpolate(x, scale_factor=self.scale, mode="bilinear")
            output += base
        return output


@ARCH_REGISTRY.register()
def span(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 52,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
    learn_residual: bool = False,
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
        learn_residual=learn_residual,
    )


@ARCH_REGISTRY.register()
def span_s(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 48,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
    learn_residual: bool = False,
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
        learn_residual=learn_residual,
    )


@ARCH_REGISTRY.register()
def span_f32(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 32,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
    learn_residual: bool = False,
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
        learn_residual=learn_residual,
    )
