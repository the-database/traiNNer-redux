# ruff: noqa
# type: ignore
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from traiNNer.archs.arch_util import default_init_weights
from traiNNer.utils.registry import ARCH_REGISTRY


class PLKB(nn.Module):
    """
    Partial Large Kernel Block (PLKB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.channels = channels
        self.split_channels = int(channels * split_factor)
        self.DWConv_Kx1 = nn.Conv2d(
            self.split_channels,
            self.split_channels,
            kernel_size=(large_kernel, 1),
            stride=1,
            padding=(large_kernel // 2, 0),
            groups=self.split_channels,
        )
        self.DWConv_1xK = nn.Conv2d(
            self.split_channels,
            self.split_channels,
            kernel_size=(1, large_kernel),
            stride=1,
            padding=(0, large_kernel // 2),
            groups=self.split_channels,
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1, x2 = torch.split(
            x, (self.split_channels, self.channels - self.split_channels), dim=1
        )
        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out


class FFB(nn.Module):
    """
    Feature Fusion Block (FFB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.PLKB(x)
        out = self.act(self.conv1(x1 + x2))
        return out


class FDB(nn.Module):
    """
    Feature Distillation Block (FDB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = FFB(channels, large_kernel, split_factor)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = FFB(channels, large_kernel, split_factor)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = FFB(channels, large_kernel, split_factor)
        self.c4 = nn.Conv2d(channels, channels // 2, 1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        return out


class SFMB(nn.Module):
    """
    Spatial Feature Modulation Block (SFMB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv_3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1_1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x)

        x2_1 = self.sigmoid(self.AdaptiveAvgPool(x))
        x2_2 = F.max_pool2d(x, kernel_size=8, stride=8)
        x2_2 = self.act(self.conv1_1(self.DWConv_3(x2_2)))
        x2_2 = F.interpolate(
            x2_2, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        x2 = x2_1 * x2_2

        out = self.act(self.conv1_2(x1 + x2))
        return out


class FSB(nn.Module):
    """
    Feature Selective Block (FSB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv_3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.conv1_1 = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x)
        x2 = self.DWConv_3(x)
        x_fused = self.act(self.conv1_1(torch.cat((x1, x2), dim=1)))
        weight = self.sigmoid(x_fused)
        out = x1 * weight + x2 * (1 - weight)
        return out


class FMB(nn.Module):
    """
    Feature Modulation Block (FMB)
    """

    def __init__(self, channels, large_kernel, split_factor) -> None:
        super().__init__()
        self.FDB = FDB(channels, large_kernel, split_factor)
        self.SFMB = SFMB(channels, large_kernel, split_factor)
        self.FSB = FSB(channels, large_kernel, split_factor)

    def forward(self, input):
        out = self.FDB(input)
        out = self.SFMB(out)
        out = self.FSB(out)
        out = out + input
        return out


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None) -> None:
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super().__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None) -> None:
        super().__init__()
        self.upsampleOneStep = UpsampleOneStep(
            scale, num_feat, num_out_ch, input_resolution=None
        )

    def forward(self, x):
        return self.upsampleOneStep(x)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Layers(nn.Module):
    def __init__(self, channels, num_block, large_kernel, split_factor) -> None:
        super().__init__()
        self.layers = make_layer(
            basic_block=FMB,
            num_basic_block=num_block,
            channels=channels,
            large_kernel=large_kernel,
            split_factor=split_factor,
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class LKFMixer(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        upscale,
        num_block,
        large_kernel,
        split_factor,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.layers = Layers(
            channels, num_block, large_kernel=large_kernel, split_factor=split_factor
        )
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.upsampler = PixelShuffleDirect(
            scale=upscale, num_feat=channels, num_out_ch=out_channels
        )
        self.act = nn.GELU()
        self.scale = upscale

    @staticmethod
    def check_img_size(x, resolution: tuple[int, int]):
        mod_pad_h = (8 - resolution[0] % 8) % 8
        mod_pad_w = (8 - resolution[1] % 8) % 8
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, input):
        b, c, h, w = input.shape
        input = self.check_img_size(input, (h, w))
        out_fea = self.conv_first(input)
        out = self.layers(out_fea)
        out = self.act(self.conv(out))
        output = self.upsampler(out + out_fea)[:, :, : h * self.scale, : w * self.scale]
        return output


@ARCH_REGISTRY.register()
def lkfmixer_t(
    in_channels: int = 3,
    channels: int = 40,
    out_channels: int = 3,
    scale: int = 4,
    num_block: int = 6,
    large_kernel: int = 31,
    split_factor: float = 0.25,
) -> LKFMixer:
    return LKFMixer(
        in_channels=in_channels,
        channels=channels,
        out_channels=out_channels,
        upscale=scale,
        num_block=num_block,
        large_kernel=large_kernel,
        split_factor=split_factor,
    )


@ARCH_REGISTRY.register()
def lkfmixer_b(
    in_channels: int = 3,
    channels: int = 48,
    out_channels: int = 3,
    scale: int = 4,
    num_block: int = 8,
    large_kernel: int = 31,
    split_factor: float = 0.25,
) -> LKFMixer:
    return LKFMixer(
        in_channels=in_channels,
        channels=channels,
        out_channels=out_channels,
        upscale=scale,
        num_block=num_block,
        large_kernel=large_kernel,
        split_factor=split_factor,
    )


@ARCH_REGISTRY.register()
def lkfmixer_l(
    in_channels: int = 3,
    channels: int = 64,
    out_channels: int = 3,
    scale: int = 4,
    num_block: int = 12,
    large_kernel: int = 31,
    split_factor: float = 0.25,
) -> LKFMixer:
    return LKFMixer(
        in_channels=in_channels,
        channels=channels,
        out_channels=out_channels,
        upscale=scale,
        num_block=num_block,
        large_kernel=large_kernel,
        split_factor=split_factor,
    )
