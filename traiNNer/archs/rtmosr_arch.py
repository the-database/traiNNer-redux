from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from traiNNer.utils.registry import ARCH_REGISTRY


# channels_first
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution"""

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


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 2, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

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

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

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
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        W_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        out = self.conv(x_pad) + self.sk(x)
        return out


class RepConv(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.randn(3), requires_grad=True)
        self.forward_module = self.train_forward

    def fuse(self) -> None:
        self.conv3.update_params()
        pad_conv_1x1 = F.pad(self.conv1.weight, (1, 1, 1, 1))
        weight_3x3 = self.conv2.weight
        bias_3x3 = self.conv2.bias
        conv_1x1_bias = self.conv1.bias
        device = self.conv_3x3_rep.weight.device
        sum_weight = (
            self.alpha[0] * pad_conv_1x1
            + self.alpha[1] * weight_3x3
            + self.conv3.eval_conv.weight * self.alpha[2]
        ).to(device)
        sum_bias = (
            self.alpha[0] * conv_1x1_bias
            + self.alpha[1] * bias_3x3
            + self.alpha[2] * self.conv3.eval_conv.bias
        ).to(device)
        self.conv_3x3_rep.weight = nn.Parameter(sum_weight)
        self.conv_3x3_rep.bias = nn.Parameter(sum_bias)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if mode:
            self.forward_module = self.train_forward
        else:
            self.fuse()
            self.forward_module = self.conv_3x3_rep
        return self

    def train_forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_module(x)


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = RepConv(dim, hidden * 2)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(dim)
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1)
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
        return x + shortcut


@ARCH_REGISTRY.register()
class RTMoSR(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        dim: int = 32,
        ffn_expansion: float = 1.5,
        n_blocks: int = 2,
        unshuffle_mod: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale

        unshuffle = 0
        if scale < 4 and unshuffle_mod:
            if scale == 3:
                raise ValueError("Unshuffle_mod does not support 3x")
            unshuffle = 4 // scale
            scale = 4
        self.unshuffle = unshuffle
        self.to_feat = (
            nn.Conv2d(3, dim, 3, 1, 1)
            if not unshuffle
            else nn.Sequential(
                nn.PixelUnshuffle(unshuffle), nn.Conv2d(3 * unshuffle**2, dim, 3, 1, 1)
            )
        )
        self.body = nn.Sequential(
            *[GatedCNNBlock(dim, ffn_expansion) for _ in range(n_blocks)]
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        scaled_size = self.unshuffle
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        short = x
        _, _, h, w = x.shape
        if self.unshuffle:
            x = self.check_img_size(x, (h, w))
        x = self.to_feat(x)
        x = self.body(x)
        return self.to_img(x)[:, :, : h * self.scale, : w * self.scale] + F.interpolate(
            short, scale_factor=self.scale
        )


@ARCH_REGISTRY.register()
def rtmosr_s(
    scale: int = 4,
    dim: int = 32,
    ffn_expansion: float = 1.5,
    n_blocks: int = 2,
    unshuffle_mod: bool = True,
) -> RTMoSR:
    return RTMoSR(
        scale=scale,
        dim=dim,
        ffn_expansion=ffn_expansion,
        n_blocks=n_blocks,
        unshuffle_mod=unshuffle_mod,
    )
