# ruff: noqa
# type: ignore
import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as f

from traiNNer.utils.registry import ARCH_REGISTRY


class _Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):  # noqa
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    r"""A memory-efficient implementation of Swish. The original code is from
    https://github.com/zudi-lin/rcan-it/blob/main/ptsr/model/_utils.py.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Swish.apply(x)


class Conv2d1x1(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride,
            padding=(0, 0),
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )


class Conv2d3x3(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )


class MeanShift(nn.Conv2d):
    r"""

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    """

    def __init__(
        self, rgb_range: int, sign: int = -1, data_type: str = "DIV2K"
    ) -> None:
        super().__init__(3, 3, kernel_size=(1, 1))

        rgb_std = (1.0, 1.0, 1.0)
        if data_type == "DIV2K":
            # RGB mean for DIV2K 1-800
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == "DF2K":
            # RGB mean for DF2K 1-3450
            rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f"Unknown data type for MeanShift: {data_type}.")

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d1x1(nn.Conv2d):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        bias: bool = True,
        shift_mode: str = "+",
        val: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride,
            padding=(0, 0),
            dilation=dilation,
            groups=1,
            bias=bias,
            **kwargs,
        )

        assert in_channels % 5 == 0, f"{in_channels} % 5 != 0."

        channel_per_group = in_channels // 5
        self.mask = nn.Parameter(
            torch.zeros((in_channels, 1, 3, 3)), requires_grad=False
        )
        if shift_mode == "+":
            self.mask[0 * channel_per_group : 1 * channel_per_group, 0, 1, 2] = val
            self.mask[1 * channel_per_group : 2 * channel_per_group, 0, 1, 0] = val
            self.mask[2 * channel_per_group : 3 * channel_per_group, 0, 2, 1] = val
            self.mask[3 * channel_per_group : 4 * channel_per_group, 0, 0, 1] = val
            self.mask[4 * channel_per_group :, 0, 1, 1] = val
        elif shift_mode == "x":
            self.mask[0 * channel_per_group : 1 * channel_per_group, 0, 0, 0] = val
            self.mask[1 * channel_per_group : 2 * channel_per_group, 0, 0, 2] = val
            self.mask[2 * channel_per_group : 3 * channel_per_group, 0, 2, 0] = val
            self.mask[3 * channel_per_group : 4 * channel_per_group, 0, 2, 2] = val
            self.mask[4 * channel_per_group :, 0, 1, 1] = val
        else:
            raise NotImplementedError(
                f"Unknown shift mode for ShiftConv2d1x1: {shift_mode}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.conv2d(
            input=x,
            weight=self.mask,
            bias=None,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=self.in_channels,
        )
        x = f.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        return x


class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:
        conv_list:

    """

    def __init__(
        self, sa_list: list, mlp_list: list, conv_list: list | None = None
    ) -> None:
        super().__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)
        self.conv = nn.Sequential(
            *conv_list if conv_list is not None else [nn.Identity()]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list, strict=False):
            x = x + sa(x)
            x = x + mlp(x)
        return self.conv(x)


class _EncoderTail(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=planes,
                out_channels=2 * planes,
                kernel_size=(2, 2),
                stride=(2, 2),
                bias=False,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderHead(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=planes,
                out_channels=2 * planes,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(
        self,
        upscale: int,
        in_channels: int,
        out_channels: int,
        upsample_mode: str = "pixelshuffle",
    ) -> None:
        layer_list = []
        if upsample_mode == "pixelshuffle":  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f"Upscale {upscale} is not supported.")
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == "pixelshuffledirect":  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale**2)))
            layer_list.append(nn.PixelShuffle(upscale))
        else:
            raise ValueError(f"Upscale mode {upscale} is not supported.")

        super().__init__(*layer_list)


class PixelMixer(nn.Module):
    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super().__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin  # 像素的偏移量
        self.mask = nn.Parameter(
            torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
            requires_grad=False,
        )

        self.mask[3::5, 0, 0, mix_margin] = 1.0
        self.mask[2::5, 0, -1, mix_margin] = 1.0
        self.mask[1::5, 0, mix_margin, 0] = 1.0
        self.mask[0::5, 0, mix_margin, -1] = 1.0
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        x = f.conv2d(
            input=f.pad(x, pad=(m, m, m, m), mode="circular"),
            weight=self.mask,
            bias=None,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=self.planes,
        )
        return x


class SWSA(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        attn_layer (list): Layers used to calculate attn
        proj_layer (list): Layers used to proj output
        window_list (tuple): List of window sizes. Input will be equally divided
            by channel to use different windows sizes
        shift_list (tuple): list of shift sizes
        return_attns (bool): Returns attns or not

    Returns:
        b c h w -> b c h w
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_layer: list | None = None,
        proj_layer: list | None = None,
        window_list: tuple = ((8, 8),),
        shift_list: tuple | None = None,
        return_attns: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.return_attns = return_attns

        self.window_list = window_list
        if shift_list is not None:
            assert len(shift_list) == len(window_list)
            self.shift_list = shift_list
        else:
            self.shift_list = ((0, 0),) * len(window_list)

        self.attn = nn.Sequential(
            *attn_layer if attn_layer is not None else [nn.Identity()]
        )
        self.proj = nn.Sequential(
            *proj_layer if proj_layer is not None else [nn.Identity()]
        )

    @staticmethod
    def check_image_size(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
        _, _, h, w = x.size()
        windows_num_h = math.ceil(h / window_size[0])
        windows_num_w = math.ceil(w / window_size[1])
        mod_pad_h = windows_num_h * window_size[0] - h
        mod_pad_w = windows_num_w * window_size[1] - w
        return f.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor or tuple:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        # calculate qkv
        qkv = self.attn(x)
        _, C, _, _ = qkv.size()

        # split channels
        qkv_list = torch.split(
            qkv, [C // len(self.window_list)] * len(self.window_list), dim=1
        )

        output_list = []
        if self.return_attns:
            attn_list = []

        for attn_slice, window_size, shift_size in zip(
            qkv_list, self.window_list, self.shift_list, strict=False
        ):
            _, _, h, w = attn_slice.size()
            attn_slice = self.check_image_size(attn_slice, window_size)

            # roooll!
            if shift_size != (0, 0):
                attn_slice = torch.roll(attn_slice, shifts=shift_size, dims=(2, 3))

            # cal attn
            _, _, H, W = attn_slice.size()
            q, v = rearrange(
                attn_slice,
                "b (qv head c) (nh ws1) (nw ws2) -> qv (b head nh nw) (ws1 ws2) c",
                qv=2,
                head=self.num_heads,
                ws1=window_size[0],
                ws2=window_size[1],
            )
            attn = q @ q.transpose(-2, -1)
            attn = f.softmax(attn, dim=-1)
            if self.return_attns:
                attn_list.append(
                    attn.reshape(
                        self.num_heads,
                        -1,
                        window_size[0] * window_size[1],
                        window_size[0] * window_size[1],
                    )
                )
            output = rearrange(
                attn @ v,
                "(b head nh nw) (ws1 ws2) c -> b (head c) (nh ws1) (nw ws2)",
                head=self.num_heads,
                nh=H // window_size[0],
                nw=W // window_size[1],
                ws1=window_size[0],
                ws2=window_size[1],
            )

            # roooll back!
            if shift_size != (0, 0):
                output = torch.roll(
                    output, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3)
                )

            output_list.append(output[:, :, :h, :w])

        # proj output
        output = self.proj(torch.cat(output_list, dim=1))

        if self.return_attns:
            return output, attn_list
        else:
            return output


class Mlp(nn.Module):
    r"""Multi-layer perceptron.

    Args:
        in_features: Number of input channels
        hidden_features:
        out_features: Number of output channels
        act_layer:

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = ShiftConv2d1x1(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = ShiftConv2d1x1(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.token_mixer = PixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.token_mixer(x) - x)


class MixedTransformerBlock(TransformerGroup):
    def __init__(
        self,
        dim: int,
        num_layer: int,
        num_heads: int,
        num_GTLs: int,
        window_list: tuple | None = None,
        shift_list: tuple | None = None,
        mlp_ratio: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        token_mixer_list = [
            TokenMixer(dim)
            if _ > (num_GTLs - 1)
            else SWSA(
                dim=dim,
                num_heads=num_heads,
                attn_layer=[Conv2d1x1(dim, dim * 2), nn.BatchNorm2d(dim * 2)],
                proj_layer=[Conv2d1x1(dim, dim)],
                window_list=window_list,
                shift_list=shift_list if (_ + 1) % 2 == 0 else None,
            )
            for _ in range(num_layer)
        ]

        mlp_list = [
            Mlp(dim, dim * mlp_ratio, act_layer=act_layer) for _ in range(num_layer)
        ]

        super().__init__(sa_list=token_mixer_list, mlp_list=mlp_list, conv_list=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list, strict=False):
            x = x + sa(x)
            x = x + mlp(x)

        return self.conv(x)


@ARCH_REGISTRY.register()
class EMT(nn.Module):
    r"""Efficient Mixed Transformer Super-Resolution Network!"""

    def __init__(
        self,
        scale: int = 4,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        upsampler: Literal["pixelshuffle", "pixelshuffledirect"] = "pixelshuffle",
        dim: int = 60,
        n_blocks: int = 6,
        n_layers: int = 6,
        num_heads: int = 3,
        mlp_ratio: int = 2,
        n_GTLs: int = 2,
        window_list: tuple = ([32, 8], [8, 32]),
        shift_list: tuple = ([16, 4], [4, 16]),
    ) -> None:
        super().__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type="DF2K")
        self.add_mean = MeanShift(255, sign=1, data_type="DF2K")

        self.head = Conv2d3x3(num_in_ch, dim)

        self.body = nn.Sequential(
            *[
                MixedTransformerBlock(
                    dim=dim,
                    num_layer=n_layers,
                    num_heads=num_heads,
                    num_GTLs=n_GTLs,
                    window_list=window_list,
                    shift_list=shift_list,
                    mlp_ratio=mlp_ratio,
                    act_layer=Swish,
                )
                for _ in range(n_blocks)
            ]
        )

        self.tail = Upsampler(
            upscale=scale,
            in_channels=dim,
            out_channels=num_out_ch,
            upsample_mode=upsampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x
