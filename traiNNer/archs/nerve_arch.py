# NERVE — Norm-free Efficient Restoration for Various Edge devices.
#
# A lightweight super-resolution network designed for ease of use rather than
# benchmark chasing: pure convolution (Conv-ReLU-Conv residual blocks, no
# normalization, no attention, no gating), a single readable file, and standard
# ops only (Conv / Add / ReLU / PixelShuffle / bicubic Resize) so the checkpoint
# exports to dynamic-shape ONNX and converts to NCNN/CoreML without custom ops
# or fused/unfused checkpoint pairs.
#
# A 5x5 stem opens the receptive field for real-world degradations; a global
# residual and a bicubic input residual let the net learn only what bicubic
# upscaling misses. Default dim=64 / n_blocks=24 (~1.8M params at 4x).

from __future__ import annotations

import math

import torch
from spandrel.util import store_hyperparameters
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


def _icnr_init(conv: nn.Conv2d, upscale: int) -> None:
    """ICNR initialization for a Conv2d that feeds a PixelShuffle(upscale).

    Every sub-pixel phase starts from the same base kernel, so the upsampling
    head cannot begin with a checkerboard bias. Init-only: no graph or runtime
    cost, and no effect once a checkpoint is loaded.
    """
    out_c = conv.out_channels
    sub = upscale * upscale
    assert out_c % sub == 0
    base_c = out_c // sub
    base = nn.init.kaiming_uniform_(conv.weight[:base_c].clone(), a=math.sqrt(5))
    with torch.no_grad():
        conv.weight.data.zero_()
        for i in range(sub):
            conv.weight.data[i::sub] = base
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class NerveBlock(nn.Module):
    """Conv-ReLU-Conv residual block. No normalization, no shortcuts."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(F.relu(self.conv1(x)))


@store_hyperparameters()
class NERVE(nn.Module):
    """5x5 stem -> n_blocks residual blocks -> PixelShuffle head -> + bicubic(x)."""

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 24,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale

        self.stem = nn.Conv2d(in_ch, dim, 5, padding=2, bias=False)
        self.blocks = nn.Sequential(*[NerveBlock(dim) for _ in range(n_blocks)])
        head = nn.Conv2d(dim, out_ch * upscale**2, 3, padding=1, bias=False)
        self.upsampler = nn.Sequential(head, nn.PixelShuffle(upscale))
        _icnr_init(head, upscale)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.stem(x)
        out = feat + self.blocks(feat)
        out = self.upsampler(out)
        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bicubic", align_corners=False
        )
        return out + base


@SPANDREL_REGISTRY.register()
@ARCH_REGISTRY.register()
def nerve(
    scale: int = 4,
    *,
    in_ch: int = 3,
    out_ch: int = 3,
    dim: int = 64,
    n_blocks: int = 24,
) -> NERVE:
    return NERVE(in_ch=in_ch, out_ch=out_ch, dim=dim, n_blocks=n_blocks, upscale=scale)
