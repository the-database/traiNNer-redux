# ruff: noqa
# type: ignore
import math
from collections.abc import Callable
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
        super().__init__(*m)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction: int = 16) -> None:
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ) -> None:
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks
    ) -> None:
        super().__init__()
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class MultyScaleResidualGroup(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        act,
        res_scale,
        n_resblocks,
        pixel_shuffle: bool = False,
    ):
        super().__init__()
        self.down = (
            nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, 3, 1, 1), nn.PixelUnshuffle(2))
            if pixel_shuffle
            else nn.Conv2d(n_feat, n_feat, 3, 2, 1)
        )
        self.body = nn.Sequential(
            *[
                RCAB(
                    conv,
                    n_feat,
                    kernel_size,
                    reduction,
                    bias=True,
                    bn=False,
                    act=act,
                    res_scale=res_scale,
                )
                for _ in range(n_resblocks)
            ]
            + [conv(n_feat, n_feat, kernel_size)]
        )
        self.act = nn.LeakyReLU(0.1, True)
        self.up = (
            nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1), nn.PixelShuffle(2))
            if pixel_shuffle
            else nn.ConvTranspose2d(n_feat, n_feat, 4, 2, 1)
        )

    def forward(self, x):
        out = self.act(self.down(x))
        out = self.body(out)
        out = self.act(self.up(out))
        return out + x


@ARCH_REGISTRY.register()
## Residual Channel Attention Network (RCAN)
class RCANMSRB(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        n_colors: int = 3,
        rgb_range: int = 255,  # TODO
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 8,
        mean_shift: Sequence[float] = (
            0.4488,
            0.4371,
            0.4040,
        ),  # div2k: (0.4488, 0.4371, 0.4040) df2k: (0.4690, 0.4490, 0.4036) normal_dataset: (0, 0, 0)
        pixel_shuffle_msrb: bool = True,
        conv: Callable[..., nn.Conv2d] = default_conv,
    ) -> None:
        super().__init__()

        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.rgb_range = rgb_range  # meh
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                mean_shift,
                dtype=torch.float32,
            ),
        )
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = (
            MeanShift(rgb_range, mean_shift, rgb_std)
            if mean_shift != (0, 0, 0)
            else nn.Identity()
        )

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body: list[nn.Module] = [
            ResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act=act,
                res_scale=res_scale,
                n_resblocks=n_resblocks,
            )
            if index % 2 == 1
            else MultyScaleResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act=act,
                res_scale=res_scale,
                n_resblocks=n_resblocks,
                pixel_shuffle=pixel_shuffle_msrb,
            )
            for index in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.add_mean = (
            MeanShift(rgb_range, mean_shift, rgb_std, 1)
            if mean_shift != (0, 0, 0)
            else nn.Identity()
        )
        self.scale = scale
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def check_img_size(self, x, resolution):
        h, w = resolution
        scaled_size = 2
        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.check_img_size(x, (h, w))
        x *= self.rgb_range
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        return (x / self.rgb_range)[:, :, : h * self.scale, : w * self.scale]

    def load_state_dict(self, state_dict, strict=False) -> None:
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") >= 0:
                        print("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(
                            f"While copying the parameter named {name}, "
                            f"whose dimensions in the model are {own_state[name].size()} and "
                            f"whose dimensions in the checkpoint are {param.size()}."
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')
