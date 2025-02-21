import math
from collections.abc import Callable, Sequence
from typing import Literal, Self, get_args

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from spandrel.architectures.__arch_helpers.dysample import DySample
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 4, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class InterpolateUpsampler(nn.Sequential):
    def __init__(self, dim: int = 64, out_ch: int = 3, scale: int = 4) -> None:
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(dim, dim, 3, 1, 1))
                m.append(Interpolate(2))
                m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            m.append(nn.Conv2d(dim, dim, 3, 1, 1))
            m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(dim, dim, 3, 1, 1))
            m.append(Interpolate(scale))
            m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            m.append(nn.Conv2d(dim, dim, 3, 1, 1))
            m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        m.append(nn.Conv2d(dim, out_ch, 3, 1, 1))
        super().__init__(*m)


class ConvBlock(nn.Module):
    r"""https://github.com/joshyZhou/AST/blob/main/model.py#L22"""

    def __init__(
        self, in_channel: int = 3, out_channel: int = 48, strides: int = 1
    ) -> None:
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.Mish(),
            nn.Conv2d(
                out_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.Mish(),
        )
        self.conv11 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=strides, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.block(x)
        out2 = self.conv11(x)
        return out1 + out2


class OmniShift(nn.Module):
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )

    def forward_train(self, x: Tensor) -> Tensor:
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self) -> None:
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.reparam_5x5()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            out = self.forward_train(x)
        else:
            out = self.conv5x5_reparam(x)
        return out


class MatMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = a @ b
        return out


class LMLTVIT(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        window_size: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.scale = dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.omni_shift = OmniShift(dim)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x: Tensor, window_size: int) -> Tensor:
        B, H, W, C = x.shape  # noqa: N806
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )

    def window_reverse(
        self, windows: Tensor, window_size: int, h: int, w: int
    ) -> Tensor:
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(
            b, h // window_size, w // window_size, window_size, window_size, -1
        )
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    def get_lepe(
        self, x: Tensor, func: Callable[[Tensor], Tensor]
    ) -> tuple[Tensor, Tensor]:
        B, _N, C = x.shape  # noqa: N806
        H = W = 8  # noqa: N806
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.window_size, self.window_size  # noqa: N806
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = (
            x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        )  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        _B, _N, C = x.shape  # noqa: N806
        H, W = resolution  # noqa: N806
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b h w c")
        ################################
        # 1. window partition
        ################################
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(
            -1, self.window_size * self.window_size, C
        )

        ################################
        # 2. make qkv
        ################################
        qkv = self.qkv(x_window)
        # qkv = qkv.permute(0,2,3,1)
        # qkv = qkv.reshape(-1, self.window_size * self.window_size, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ################################
        # 3. attn and PE
        ################################
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        # x = x.reshape(-1, self.window_size, self.window_size, C)
        # x = x.permute(0,3,1,2)

        ################################
        # 4. proj and drop
        ################################
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)
        x = rearrange(x, "b h w c-> b (h w) c")
        return x


class ChannelMix(nn.Module):
    def __init__(
        self, n_embd: int = 48, hidden_rate: int = 4, key_norm: bool = False
    ) -> None:
        super().__init__()
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = OmniShift(dim=n_embd)

        if key_norm:
            self.key_norm = nn.RMSNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        h, w = resolution

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.rn1 = nn.RMSNorm(dim)
        self.rn2 = nn.RMSNorm(dim)
        self.att = LMLTVIT(dim, window_size, attn_drop, proj_drop)
        self.ffn = ChannelMix(dim, hidden_rate, channel_norm)
        self.gamma1 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        x = x + self.gamma1 * self.att(self.rn1(x), resolution)
        x = x + self.gamma2 * self.ffn(self.rn2(x), resolution)
        return x


class LBlock(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        n_block: int = 1,
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, window_size, hidden_rate, channel_norm, attn_drop, proj_drop
                )
                for _ in range(n_block)
            ]
        )
        self.conv = ConvBlock(dim * 2, dim)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        H, W = resolution  # noqa: N806
        shortcut = x
        for t_block in self.t_blocks:
            x = t_block(x, resolution)
        x = torch.cat([shortcut, x], dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class MBlock(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        n_block: int = 1,
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, window_size, hidden_rate, channel_norm, attn_drop, proj_drop
                )
                for _ in range(n_block)
            ]
        )
        self.conv = ConvBlock(dim * 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        _B, _C, H, W = x.shape  # noqa: N806
        resolution = (H, W)
        x = rearrange(x, "b c h w -> b (h w) c")
        shortcut = x
        for t_block in self.t_blocks:
            x = t_block(x, resolution)
        x = torch.cat([shortcut, x], dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        return x


class LinearPipeline(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.att = nn.ModuleList(
            [
                LBlock(
                    dim,
                    num_block,
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
                for num_block in num_blocks
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.size()  # noqa: N806
        resolution = (H, W)
        x = rearrange(x, "b c h w -> b (h w) c")
        for attn in self.att:
            x = attn(x, resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat: int = 48) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int = 48) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class MetaPipeline(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc0 = nn.Sequential(
            *[
                MBlock(
                    dim,
                    num_blocks[0],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )
        self.enc1 = nn.Sequential(
            *[
                MBlock(
                    dim * 2,
                    num_blocks[1],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )
        self.enc2 = nn.Sequential(
            *[
                MBlock(
                    dim * 4,
                    num_blocks[2],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )
        self.enc3 = nn.Sequential(
            *[
                MBlock(
                    dim * 8,
                    num_blocks[3],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )

        self.down1 = Downsample(dim)
        self.down2 = Downsample(dim * 2)
        self.down3 = Downsample(dim * 4)

        self.up1 = Upsample(dim * 16)
        self.up2 = Upsample(dim * 8)
        self.up3 = Upsample(dim * 4)

        self.dec0 = nn.Sequential(
            *[
                MBlock(
                    dim * 4,
                    num_blocks[2],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )
        self.dec1 = nn.Sequential(
            *[
                MBlock(
                    dim * 2,
                    num_blocks[1],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )
        self.dec2 = nn.Sequential(
            *[
                MBlock(
                    dim,
                    num_blocks[0],
                    window_size,
                    hidden_rate,
                    channel_norm,
                    attn_drop,
                    proj_drop,
                )
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        enc0 = self.enc0(x)
        enc0 = self.down1(enc0)

        enc1 = self.enc1(enc0)
        enc1 = self.down2(enc1)

        enc2 = self.enc2(enc1)
        enc2 = self.down3(enc2)

        enc3 = self.enc3(enc2)
        enc3 = torch.cat([enc3, enc2], dim=1)

        x = self.up1(enc3)
        x = self.dec0(x)
        x = torch.cat([x, enc1], dim=1)

        x = self.up2(x)
        x = self.dec1(x)
        x = torch.cat([x, enc0], dim=1)

        x = self.up3(x)
        x = self.dec2(x)
        return x


T_upsampler = Literal["pixelshuffle", "nearest+conv", "dysample"]


@ARCH_REGISTRY.register()
class FlexNet(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        scale: int = 4,
        dim: int = 64,
        num_blocks: Sequence[int] = (
            6,
            6,
            6,
            6,
            6,
            6,
        ),  # meta = (8,8,8,8), # linear = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        hidden_rate: int = 4,
        channel_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pipeline_type: Literal["meta", "linear"] = "linear",
        upsampler: T_upsampler = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "window_size", torch.tensor(window_size, dtype=torch.uint8)
        )
        self.pipeline_type = pipeline_type
        self.scale = scale
        self.short_cut = ConvBlock(inp_channels, dim)
        self.in_to_feat = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        self.pipeline = (
            LinearPipeline(
                dim,
                num_blocks,
                window_size,
                hidden_rate,
                channel_norm,
                attn_drop,
                proj_drop,
            )
            if pipeline_type == "linear"
            else MetaPipeline(
                dim,
                num_blocks,
                window_size,
                hidden_rate,
                channel_norm,
                attn_drop,
                proj_drop,
            )
        )
        if upsampler == "nearest+conv":
            self.register_buffer("scale_factor", torch.tensor(scale, dtype=torch.uint8))
            self.to_img = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                InterpolateUpsampler(dim, out_channels, scale),
            )
        elif upsampler == "dysample":
            self.to_img = DySample(dim * 2, out_channels, scale)
        elif upsampler == "pixelshuffle":
            self.to_img = nn.Sequential(
                nn.Conv2d(dim * 2, out_channels * (scale**2), 3, 1, 1),
                nn.PixelShuffle(scale),
            )
        else:
            raise ValueError(
                f"upsampler {upsampler} not supported, choose one of these options: {get_args(T_upsampler)}"
            )

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        assert isinstance(self.window_size, Tensor)
        h, w = resolution
        scaled_size = self.window_size.item()
        assert isinstance(scaled_size, int)
        if self.pipeline_type == "meta":
            scaled_size *= 8
        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.size()
        x = self.check_img_size(x, (h, w))
        short_cut = self.short_cut(x)
        x = self.in_to_feat(x)
        x = self.pipeline(x)
        x = torch.cat([x, short_cut], dim=1)
        x = self.to_img(x)
        return x[:, :, : h * self.scale, : w * self.scale]


@ARCH_REGISTRY.register()
def metaflexnet(
    inp_channels: int = 3,
    out_channels: int = 3,
    scale: int = 4,
    dim: int = 64,
    num_blocks: Sequence[int] = (
        4,
        6,
        6,
        8,
    ),  # meta = (8,8,8,8), # linear = (6, 6, 6, 6, 6, 6),
    window_size: int = 8,
    hidden_rate: int = 4,
    channel_norm: bool = False,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    pipeline_type: Literal["meta", "linear"] = "meta",
    upsampler: T_upsampler = "pixelshuffle",
) -> FlexNet:
    return FlexNet(
        scale=scale,
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        window_size=window_size,
        hidden_rate=hidden_rate,
        channel_norm=channel_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        pipeline_type=pipeline_type,
        upsampler=upsampler,
    )
