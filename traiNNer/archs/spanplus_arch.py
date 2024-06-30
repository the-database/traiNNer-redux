# https://github.com/umzi2/SPANPlus/blob/master/neosr/archs/spanplus_arch.py

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.init import trunc_normal_
from traiNNer.utils.registry import ARCH_REGISTRY


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()

        try:
            assert in_channels >= groups and in_channels % groups == 0
        except:  # noqa: E722
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)  # noqa: B904

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):  # noqa: ANN202
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x):  # noqa: ANN201, ANN001
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape  # noqa: N806
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 1, s: int = 1, bias: bool = True
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
        if self.training:
            trunc_normal_(self.sk.weight, std=0.02)
            trunc_normal_(self.eval_conv.weight, std=0.02)

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # type: ignore
            self.update_params()

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

    def forward(self, x):  # noqa: ANN201, ANN001
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        return out


class SPAB(nn.Module):
    def __init__(self, in_channels: int, end: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c2_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c3_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.act1 = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.end = end

    def forward(self, x):  # noqa: ANN201, ANN001
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPABS(nn.Module):
    def __init__(
        self, feature_channels: int, n_blocks: int = 4, drop: float = 0.0
    ) -> None:
        super().__init__()
        self.block_1 = SPAB(feature_channels)

        self.block_n = nn.Sequential(*[SPAB(feature_channels) for _ in range(n_blocks)])
        self.block_end = SPAB(feature_channels, True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)
        self.conv_cat = nn.Conv2d(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout2d(drop)
        if self.training:
            trunc_normal_(self.conv_cat.weight, std=0.02)

    def forward(self, x):  # noqa: ANN201, ANN001
        out_b1 = self.block_1(x)
        out_x = self.block_n(out_b1)
        out_end, out_x_2 = self.block_end(out_x)
        out_end = self.dropout(self.conv_2(out_end))
        return self.conv_cat(torch.cat([x, out_end, out_b1, out_x_2], 1))


class SpanPlus(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        blocks: list | None = None,
        feature_channels: int = 48,
        upscale: int = 4,
        drop_rate: float = 0.0,
        upsampler: str = "dys",  # "lp", "ps", "conv"- only 1x
    ) -> None:
        if blocks is None:
            blocks = [4]
        super().__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch if upsampler == "dys" else num_in_ch

        if not self.training:
            drop_rate = 0
        self.feats = nn.Sequential(
            *[Conv3XC(in_channels, feature_channels, gain=2, s=1)]
            + [SPABS(feature_channels, n_blocks, drop_rate) for n_blocks in blocks]
        )
        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, out_channels * (upscale**2), 3, padding=1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == "dys":
            self.upsampler = DySample(feature_channels, out_channels, upscale)
        elif upsampler == "conv":
            if upscale != 1:
                msg = "conv supports only 1x"
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(feature_channels, out_channels, 3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x'
            )

    def forward(self, x):  # noqa: ANN201, ANN001
        out = self.feats(x)
        return self.upsampler(out)


@ARCH_REGISTRY.register()
def spanplus(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    blocks: list | None = None,
    feature_channels: int = 48,
    drop_rate: float = 0.0,
    upsampler: str = "dys",  # "lp", "ps", "conv"- only 1x
    **kwargs,
) -> SpanPlus:
    return SpanPlus(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        blocks=blocks,
        feature_channels=feature_channels,
        upscale=scale,
        drop_rate=drop_rate,
        upsampler=upsampler,
        **kwargs,
    )


@ARCH_REGISTRY.register()
def spanplus_sts(scale: int = 4, **kwargs) -> SpanPlus:
    return SpanPlus(
        upscale=scale, blocks=[2], feature_channels=32, upsampler="ps", **kwargs
    )


@ARCH_REGISTRY.register()
def spanplus_s(scale: int = 4, **kwargs) -> SpanPlus:
    return SpanPlus(upscale=scale, blocks=[2], feature_channels=32, **kwargs)


@ARCH_REGISTRY.register()
def spanplus_st(scale: int = 4, **kwargs) -> SpanPlus:
    return SpanPlus(upscale=scale, upsampler="ps", **kwargs)
