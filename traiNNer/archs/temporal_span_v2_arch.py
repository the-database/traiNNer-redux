from __future__ import annotations

import math
from collections import OrderedDict
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.checkpoint import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY


def _make_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    bias: bool = True,
) -> nn.Conv2d:
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(
    act_type: str, inplace: bool = True, neg_slope: float = 0.05, n_prelu: int = 1
) -> nn.Module:
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    """
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"activation layer [{act_type:s}] is not found")
    return layer


def sequential(*args: nn.Module) -> nn.Sequential | nn.Module:
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules: list[nn.Module] = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        else:
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3
) -> nn.Sequential | nn.Module:
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    """Efficient reparameterizable convolution block from SPAN"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        gain2: int = 0,
        s: int = 1,
        bias: bool = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.params_updated = False  # Cache flag to avoid redundant updates
        self.stride = s
        self.has_relu = relu
        gain = gain1

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
        self.eval_conv.weight.requires_grad = False
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        assert self.conv[0].bias is not None
        b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        assert self.conv[1].bias is not None
        b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        assert self.conv[2].bias is not None
        b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]

        w = (
            F.conv2d((w1.flip(2, 3)).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d((w.flip(2, 3)).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        assert self.sk.bias is not None
        sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        assert self.weight_concat is not None
        self.eval_conv.weight.data = self.weight_concat
        assert self.bias_concat is not None
        assert self.eval_conv.bias is not None
        self.eval_conv.bias.data = self.bias_concat
        self.params_updated = True

    def train(self, mode: bool = True) -> Self:
        """Override train to trigger weight update when switching to eval mode."""
        super().train(mode)
        if not mode:  # Switching to eval mode
            self.params_updated = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            # Only update params once per eval session
            if not self.params_updated:
                self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block from SPAN"""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation("lrelu", neg_slope=0.1, inplace=True)

    def forward(
        self, x: torch.Tensor, return_intermediate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        if return_intermediate:
            return out, out1, sim_att
        return out


class TemporalSPANBlock(nn.Module):
    """
    Temporal processing block that processes 3 concatenated frames using SPAB blocks.
    Similar to TSCUNetBlock but uses SPAN's efficient SPAB architecture.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        dim: int = 48,
        num_blocks: int = 6,
        bias: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Head: process concatenated 3 frames (in_nc * 3) -> dim
        self.m_head = Conv3XC(in_nc, dim, gain1=2, s=1)

        # SPAB blocks for feature extraction
        self.blocks = nn.ModuleList([SPAB(dim, bias=bias) for _ in range(num_blocks)])

        # Feature concatenation from multiple blocks (similar to SPAN)
        # We'll concatenate features from blocks at positions that give us 4 features total
        self.conv_cat = conv_layer(dim * 4, dim, kernel_size=1, bias=True)
        self.conv_post = Conv3XC(dim, dim, gain1=2, s=1)

        # Tail: output features
        self.m_tail = nn.Sequential(
            nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, True)
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation for checkpointing."""
        out_feature = self.m_head(x)

        # Process through SPAB blocks, only keeping necessary intermediates
        out_b1 = self.blocks[0](out_feature)
        out_b2 = self.blocks[1](out_b1)
        out_b3 = self.blocks[2](out_b2)

        out_b4 = self.blocks[3](out_b3)
        out_b5 = self.blocks[4](out_b4)
        # Only get intermediate from last block
        out_b6, out_b5_2, _ = self.blocks[5](out_b5, return_intermediate=True)

        out_b6 = self.conv_post(out_b6)

        # Concatenate features from different blocks
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        out = self.m_tail(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C*3, H, W) - concatenated 3 frames
        Returns:
            out: (B, dim, H, W)
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)  # type: ignore[return-value]
        return self._forward_impl(x)


class TSPANv2(nn.Module):
    """
    TSPANv2: Video Super-Resolution using SPAN architecture
    Combines SPAN's efficient SPAB blocks with TSCUNet's temporal processing

    Args:
        in_nc: Number of input channels (default: 3 for RGB)
        out_nc: Number of output channels (default: 3 for RGB)
        clip_size: Number of frames in input clip (must be odd, default: 5)
        dim: Feature dimension (default: 48)
        num_blocks: Number of SPAB blocks per temporal layer (default: 6)
        upscale: Upscaling factor (default: 4)
        bias: Use bias in SPAB blocks (default: False)
        residual: Add residual connection from center frame (default: True)
        img_range: Image range for normalization (default: 255.0)
        rgb_mean: RGB mean values for normalization (default: (0.4488, 0.4371, 0.4040))
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        clip_size: int = 5,
        dim: int = 48,
        num_blocks: int = 6,
        upscale: int = 4,
        bias: bool = False,
        residual: bool = True,
        img_range: float = 255.0,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        if clip_size % 2 == 0:
            raise ValueError("TSPANv2 clip_size must be odd")

        self.clip_size = clip_size
        self.dim = dim
        self.upscale = upscale
        self.residual = residual
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.use_checkpoint = use_checkpoint

        # Initial conv to feature space for each frame
        self.m_head = nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)

        # Temporal processing layers - similar to TSCUNet
        # Each layer processes 3-frame windows and reduces temporal dimension
        num_temporal_layers = (clip_size - 1) // 2
        self.m_layers = nn.ModuleList(
            [
                TemporalSPANBlock(dim * 3, dim, dim, num_blocks, bias, use_checkpoint)
                for _ in range(num_temporal_layers)
            ]
        )

        # Residual connection from center frame
        if self.residual:
            self.m_res = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

        # Upsampling using PixelShuffle
        self.m_upsample = pixelshuffle_block(dim, dim, upscale_factor=upscale)

        # Final output conv
        self.m_tail = nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) where T = clip_size
        Returns:
            out: (B, C, H*scale, W*scale)
        """
        b, t, c, h, w = x.size()

        if t != self.clip_size:
            raise ValueError(
                f"Input clip size {t} does not match model clip size {self.clip_size}"
            )

        # Normalize input
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # Padding for spatial dimensions (divisible by 64)
        padding_h = int(np.ceil(h / 64) * 64 - h)
        padding_w = int(np.ceil(w / 64) * 64 - w)

        padding_left = math.ceil(padding_w / 2)
        padding_right = math.floor(padding_w / 2)
        padding_top = math.ceil(padding_h / 2)
        padding_bottom = math.floor(padding_h / 2)

        # Apply padding and process all frames through head
        x = x.view(-1, c, h, w)  # (B*T, C, H, W)
        x = nn.ReflectionPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )(x)
        x = self.m_head(x)  # (B*T, dim, H+pad, W+pad)
        x = x.view(
            b, t, self.dim, h + padding_h, w + padding_w
        )  # (B, T, dim, H+pad, W+pad)

        # Store center frame for residual
        x_center = x[:, self.clip_size // 2, ...]  # (B, dim, H+pad, W+pad)

        # Temporal processing: slide 3-frame windows through temporal dimension
        # Process more efficiently to reduce memory usage
        for layer in self.m_layers:
            current_t = x.size(1)
            num_windows = current_t - 2

            # Pre-allocate output tensor for better memory efficiency
            out = torch.empty(
                b,
                num_windows,
                self.dim,
                h + padding_h,
                w + padding_w,
                device=x.device,
                dtype=x.dtype,
            )

            # Slide 3-frame window
            for i in range(num_windows):
                # Concatenate 3 consecutive frames: [i, i+1, i+2]
                x_window = x[:, i : i + 3, ...].reshape(
                    b, -1, h + padding_h, w + padding_w
                )
                out[:, i] = layer(x_window)

            x = out  # (B, T-2, dim, H+pad, W+pad)

        # After all layers, we should have single frame
        x = x.squeeze(1)  # (B, dim, H+pad, W+pad)

        # Add residual from center frame
        if self.residual:
            x = x + self.m_res(x_center)

        # Upsample and output
        x = self.m_upsample(x)  # (B, dim, H*scale+pad, W*scale+pad)
        x = self.m_tail(x)  # (B, C, H*scale+pad, W*scale+pad)

        # Remove padding
        x = x[
            ...,
            padding_top * self.upscale : padding_top * self.upscale + h * self.upscale,
            padding_left * self.upscale : padding_left * self.upscale
            + w * self.upscale,
        ]

        # Denormalize
        x = x / self.img_range + self.mean

        return x


@ARCH_REGISTRY.register()
def temporalspanv2(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_frames: int = 5,
    feature_channels: int = 48,
    scale: int = 4,
    bias: bool = False,
    num_blocks: int = 6,
    residual: bool = True,
    img_range: float = 255.0,
    use_checkpoint: bool = False,
) -> TSPANv2:
    """TSPANv2 model factory function."""
    return TSPANv2(
        in_nc=num_in_ch,
        out_nc=num_out_ch,
        clip_size=num_frames,
        dim=feature_channels,
        num_blocks=num_blocks,
        upscale=scale,
        bias=bias,
        residual=residual,
        img_range=img_range,
        use_checkpoint=use_checkpoint,
    )
