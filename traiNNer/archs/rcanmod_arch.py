import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

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
                    m.append(act)
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act)
        else:
            raise NotImplementedError
        # print("m0", m)
        super().__init__(*m)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, dim, expansion_esa=0.25):
        super().__init__()
        f = int(dim * expansion_esa)
        self.conv1 = nn.Conv2d(dim, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, dim, kernel_size=1)
        self.sigmoid = nn.Hardsigmoid(True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _B, _C, H, W = x.shape
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (H, W), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class CSELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Hardsigmoid(True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAMBlock(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
        )
        self.sigmoid_channel = nn.Hardsigmoid(True)

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid_spatial = nn.Hardsigmoid(True)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)

    def forward(self, x):
        short_cut = x
        # Channel Attention
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        # Spatial Attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(
            self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_attention

        return (x * self.gamma) + short_cut


class SSELayer(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.squeezing = nn.Sequential(nn.Conv2d(dim, 1, 1, 1, 0), nn.Hardsigmoid(True))

    def forward(self, x):
        return x * self.squeezing(x)


class CSSELayer(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.sse = SSELayer(dim)
        self.cse = CSELayer(dim)

    def forward(self, x):
        return torch.max(self.sse(x), self.cse(x))


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
        modules_body.append(SSELayer(n_feat))
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
