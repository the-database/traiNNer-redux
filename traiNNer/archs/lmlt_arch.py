from collections.abc import Callable, Iterable
from itertools import repeat
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, shrinkage_rate: float = 0.25) -> None:
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ccm(x)


T = TypeVar("T")


def _ntuple(n: int) -> Callable[[T], Iterable[T]]:
    def parse(x: T) -> Iterable[T]:
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple: Callable[[int], tuple[int, int]] = _ntuple(2)  # type: ignore
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information."""

    def __init__(self, dim: int, k: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            dim, dim, to_2tuple(k), to_2tuple(1), to_2tuple(k // 2), groups=dim
        )

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = size
        assert n == h * w

        feat = x.transpose(1, 2).view(b, c, h, w)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


##############################################################
## Downsample ViT
class DownsampleViT(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        down_scale: int = 2,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.scale = dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x: Tensor, window_size: int) -> Tensor:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        return (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, c)
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
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def get_lepe(self, x: Tensor, func: nn.Conv2d) -> tuple[Tensor, nn.Conv2d]:
        b, n, c = x.shape
        h = w = int(np.sqrt(n))
        x = x.transpose(-2, -1).contiguous().view(b, c, h, w)

        h_sp, w_sp = self.window_size, self.window_size
        x = x.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp)
        x = (
            x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, c, h_sp, w_sp)
        )  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, c, h_sp * w_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, c, h_sp * w_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x: Tensor) -> Tensor:
        _, c, h, w = x.shape

        ################################
        # 1. window partition
        ################################
        x = x.permute(0, 2, 3, 1)
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(
            -1, self.window_size * self.window_size, c
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

        x = x.reshape(-1, self.window_size, self.window_size, c)
        x = self.window_reverse(x, self.window_size, h, w)

        return x.permute(0, 3, 1, 2)


##############################################################
## LHSB - split dim and define 4 attn blocks
class LHSB(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        n_levels: int = 4,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [
                DownsampleViT(
                    dim // 4,
                    window_size=8,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    down_scale=2**i,
                )
                for i in range(self.n_levels)
            ]
        )

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []

        downsampled_feat = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                downsampled_feat.append(s)

            else:
                downsampled_feat.append(xc[i])

        for i in reversed(range(self.n_levels)):
            s = self.mfr[i](downsampled_feat[i])
            s_upsample = F.interpolate(
                s, size=(s.shape[2] * 2, s.shape[3] * 2), mode="nearest"
            )

            if i > 0:
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample

            s_original_shape = F.interpolate(s, size=(h, w), mode="nearest")
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


##############################################################
## Block
class AttBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_scale: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.lhsb = LHSB(dim, attn_drop=attn_drop, proj_drop=drop)

        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lhsb(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


##############################################################
## Overall Architecture
@ARCH_REGISTRY.register()
class LMLT(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int = 8,
        ffn_scale: float = 2.0,
        scale: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.window_size = 8

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)
        ]  # stochastic depth decay rule

        self.feats = nn.Sequential(
            *[
                AttBlock(
                    dim,
                    ffn_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(n_blocks)
            ]
        )

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def check_img_size(self, x: Tensor) -> Tensor:
        _, _, h, w = x.size()
        downsample_scale = 8
        scaled_size = self.window_size * downsample_scale

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape

        # check image size
        x = self.check_img_size(x)

        # patch embed
        x = self.to_feat(x)

        # module, and return to original shape
        x = self.feats(x) + x
        x = x[:, :, :h, :w]

        # reconstruction
        x = self.to_img(x)
        return x


@ARCH_REGISTRY.register()
def lmlt_base(scale: int = 4, **kwargs) -> LMLT:
    return LMLT(60, 8, 2.0, scale, **kwargs)


@ARCH_REGISTRY.register()
def lmlt_large(scale: int = 4, **kwargs) -> LMLT:
    return LMLT(84, 8, 2.0, scale, **kwargs)


@ARCH_REGISTRY.register()
def lmlt_tiny(scale: int = 4, **kwargs) -> LMLT:
    return LMLT(36, 8, 2.0, scale, **kwargs)
