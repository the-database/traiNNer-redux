import math
from collections.abc import Callable

from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    # print(
    #     "default_conv", in_channels, out_channels, kernel_size, kernel_size // 2, bias
    # )
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


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
        # print("m0", m)
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
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


@ARCH_REGISTRY.register()
## Residual Channel Attention Network (RCAN)
class RCANMod(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        n_colors: int = 3,
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 1,
        conv: Callable[..., nn.Conv2d] = default_conv,
    ) -> None:
        super().__init__()

        act = nn.ReLU(True)

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
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
