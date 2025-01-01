# https://github.com/muslll/neosr/blob/master/neosr/archs/realplksr_arch.py
# With modifications by umzi2 to support dysample and layer norm
# from spandrel.architectures.PLKSR import PLKSR, RealPLKSR
from functools import partial
from typing import Literal

import torch
from spandrel.architectures.__arch_helpers.dysample import DySample
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from traiNNer.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    "Partial Large Kernel Convolutional Layer"

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    "Element-wise Attention"

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        use_ea: bool = True,
        norm_groups: int = 4,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        # Layer Norm
        self.layer_norm = LayerNorm(dim) if use_layer_norm else nn.Identity()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)
        if not use_layer_norm:
            self.norm = nn.GroupNorm(norm_groups, dim)
            nn.init.constant_(self.norm.bias, 0)
            nn.init.constant_(self.norm.weight, 1.0)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_skip = x
        x = self.layer_norm(x)
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


class RealPLKSR(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution:
    https://arxiv.org/abs/2404.11848
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 4,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0,
        upsampler: str = "pixelshuffle",  # dysample, pixelshuffle
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.upscale = upscaling_factor
        if not self.training:
            dropout = 0
        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, use_ea, norm_groups, layer_norm)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, out_ch * upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

        if upsampler == "dysample":
            groups = out_ch if upscaling_factor % 2 != 0 else 4
            self.to_img = DySample(
                in_ch * upscaling_factor**2,
                out_ch,
                upscaling_factor,
                groups=groups,
                end_convolution=True if upscaling_factor != 1 else False,
            )

        elif upsampler == "pixelshuffle":
            self.to_img = nn.PixelShuffle(upscaling_factor)
        else:
            raise ValueError(f"Invalid upsampler: {upsampler}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.feats(x) + self.repeat_op(x)
        return self.to_img(x)


@ARCH_REGISTRY.register()
def realplksr(
    in_ch: int = 3,
    out_ch: int = 3,
    dim: int = 64,
    n_blocks: int = 28,
    scale: int = 4,
    kernel_size: int = 17,
    split_ratio: float = 0.25,
    use_ea: bool = True,
    norm_groups: int = 4,
    dropout: float = 0,
    upsampler: Literal[
        "dysample", "pixelshuffle"
    ] = "pixelshuffle",  # dysample, pixelshuffle
    layer_norm: bool = True,
) -> RealPLKSR:
    return RealPLKSR(
        upscaling_factor=scale,
        in_ch=in_ch,
        out_ch=out_ch,
        dim=dim,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        use_ea=use_ea,
        norm_groups=norm_groups,
        dropout=dropout,
        upsampler=upsampler,
        layer_norm=layer_norm,
    )
