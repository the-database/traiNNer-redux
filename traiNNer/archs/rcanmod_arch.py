# ruff: noqa
# type: ignore
import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.init import trunc_normal_


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


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels) -> None:
        """

        :param num_channels: No of input channels
        """
        super().__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print("before", input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        # print("after", input_tensor.size(), squeeze_tensor.size())
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class SSELayer(nn.Module):
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.squeezing = nn.Sequential(nn.Conv2d(dim, 1, 1, 1, 0), nn.Hardsigmoid(True))

    def forward(self, x):
        return x * self.squeezing(x)


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
        modules_body.append(MSG(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class MSG(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.PixelUnshuffle(2),
            nn.LeakyReLU(0.1, True),
        )
        self.gated = nn.Sequential(
            *[GatedCNNBlock(dim, expansion_ratio=1.5) for _ in range(3)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        out = self.down(x)
        out = self.gated(out)
        return self.up(out) + x


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return (x * self.gamma) + shortcut


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


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


# @ARCH_REGISTRY.register()
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
