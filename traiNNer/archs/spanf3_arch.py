from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.archs.conv3xc import Conv3XC
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


class PixelUnshuffle(nn.Module):
    """Drop-in replacement for nn.PixelUnshuffle that exports correctly under
    torch.onnx.export(dynamo=True).

    nn.PixelUnshuffle is rewritten to ONNX SpaceToDepth by the dynamo exporter.
    SpaceToDepth has no `mode` attribute (onnx#3739) and defaults to DCR
    channel ordering, while PyTorch's PixelUnshuffle uses CRD ordering — so
    output channels come out permuted. Expressing the op as reshape+permute
    avoids the rewrite and keeps ordering correct on every opset.
    """

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.downscale_factor
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // r, r, w // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.reshape(b, c * r * r, h // r, w // r)


class SPABV2(nn.Module):
    """SPAB with learned 1x1 conv attention and ReLU activation. The 1x1 conv
    enables cross-channel gating with C² params per block, wrapped in tanh to
    bound attention magnitude to [-1, 1] for training stability under aggressive
    losses (GAN, perceptual)."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
        bias: bool = False,
        fused_training: bool = False,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(
            in_channels, mid_channels, gain1=2, s=1, fused_training=fused_training
        )
        self.c2_r = Conv3XC(
            mid_channels, mid_channels, gain1=2, s=1, fused_training=fused_training
        )
        self.c3_r = Conv3XC(
            mid_channels, out_channels, gain1=2, s=1, fused_training=fused_training
        )
        self.act1 = nn.ReLU(inplace=True)
        self.att_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.att_conv.weight)
        nn.init.constant_(self.att_conv.bias, 3.0)  # pyright: ignore[reportArgumentType]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        m = torch.tanh(self.att_conv(out3))
        out = (x + out3) * m

        return out, out1, m


class SPANF3(nn.Module):
    """SPANF2 with learned 1x1 conv attention (SPABV2) and ReLU activation.
    The learned attention enables cross-channel gating while ReLU fuses with
    preceding convolutions in TRT for faster inference."""

    def __init__(
        self,
        *,
        num_in_ch: int,
        num_out_ch: int,
        feature_channels: int = 48,
        num_blocks: int = 6,
        upscale: int = 4,
        bias: bool = True,
        unshuffle_mod: bool = False,
        fused_training: bool = False,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.upscale = upscale
        self.pad = 0

        # Head
        if unshuffle_mod and upscale < 3:
            unshuffle = 4 // upscale
            upscale = 4
            self.conv_1: nn.Module = nn.Sequential(
                PixelUnshuffle(unshuffle),
                Conv3XC(
                    num_in_ch * unshuffle**2,
                    feature_channels,
                    gain1=2,
                    s=1,
                    fused_training=fused_training,
                ),
            )
            self.pad = unshuffle
        else:
            self.conv_1 = Conv3XC(
                num_in_ch, feature_channels, gain1=2, s=1, fused_training=fused_training
            )

        # Body
        for i in range(1, num_blocks + 1):
            setattr(
                self,
                f"block_{i}",
                SPABV2(feature_channels, bias=bias, fused_training=fused_training),
            )

        # Tail (2 convs instead of SPAN's 3)
        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.to_pixel = Conv3XC(
            feature_channels,
            num_out_ch * upscale**2,
            gain1=2,
            s=1,
            fused_training=fused_training,
        )
        self.upsampler = nn.PixelShuffle(upscale)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if isinstance(m, Conv3XC):
                m.switch_to_deploy()

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        if self.pad == 0:
            return x
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, h, w)

        out_feature = self.conv_1(x)

        out = out_feature
        out_b1 = out_feature
        last_mid = out_feature

        for i in range(1, self.num_blocks + 1):
            block: SPABV2 = getattr(self, f"block_{i}")
            out, mid, _att = block(out)
            if i == 1:
                out_b1 = out
            last_mid = mid

        out = self.conv_cat(torch.cat([out_feature, out, out_b1, last_mid], 1))
        out = self.to_pixel(out)
        out = self.upsampler(out)
        return out[:, :, : h * self.upscale, : w * self.upscale]


@ARCH_REGISTRY.register()
def spanf3(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 48,
    num_blocks: int = 6,
    scale: int = 4,
    bias: bool = True,
    unshuffle_mod: bool = True,
    fused_training: bool = False,
) -> SPANF3:
    return SPANF3(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_blocks=num_blocks,
        upscale=scale,
        bias=bias,
        unshuffle_mod=unshuffle_mod,
        fused_training=fused_training,
    )
