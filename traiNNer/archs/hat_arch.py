# ruff: noqa
# type: ignore
# HAT from https://github.com/XPixelGroup/HAT/blob/main/hat/archs/hat_arch.py
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from einops import rearrange
from spandrel.architectures.__arch_helpers.padding import pad_to_multiple
from spandrel.util import store_hyperparameters
from spandrel.util.timm import to_2tuple, trunc_normal_
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.utils import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY

# attn_impl: "flex" runs the window attention and the OCA on FlexAttention: the same function as the
# stock path, with the relative position bias and the shifted-window mask applied inside a fused
# kernel, so the [windows, heads, n, n] scores and biases are never materialised (much less training
# memory). "auto" (the default) uses it while training on a CUDA GPU (sm_80+),
# no attention dropout and deterministic algorithms off; inference, validation and ONNX export always
# use the stock path. "stock" never uses it. The weights are the same either way.
AttnImpl = Literal["auto", "flex", "stock"]


# One compiled function per attention kind (window, shifted window, OCA), each with its own compile
# cache: a compiled flex_attention called directly would share one cache, of 8 entries, with every
# other flex_attention user in the process. The tensors are arguments with dynamic shapes, so any
# batch, window count and window size reuses one kernel per head dim and dtype: q comes pre-scaled
# (as in the stock path; no float to specialise on) and the bias table as a plain tensor (parameters
# get static shapes). Only a different head dim or dtype compiles again.
@torch.compile(dynamic=True)
def _flex_window_lse(q, k, v, table, rpi):
    def score_mod(score, b, h, qi, ki):
        return score + table[rpi[qi, ki], h]

    return flex_attention(q, k, v, score_mod=score_mod, scale=1.0, return_lse=True)


@torch.compile(dynamic=True)
def _flex_window_shifted_lse(q, k, v, table, rpi, region):
    # plus the shifted-window mask: -100 between tokens of different regions, as the stock mask
    def score_mod(score, b, h, qi, ki):
        w = b % region.shape[0]
        same = region[w, qi] == region[w, ki]
        return score + table[rpi[qi, ki], h] + torch.where(same, 0.0, -100.0)

    return flex_attention(q, k, v, score_mod=score_mod, scale=1.0, return_lse=True)


@torch.compile(dynamic=True)
def _flex_oca_lse(q, k, v, table, rpi):
    def score_mod(score, b, h, qi, ki):
        return score + table[rpi[qi, ki], h]

    return flex_attention(q, k, v, score_mod=score_mod, scale=1.0, return_lse=True)


def _flex_bias_grad(fn, q, k, v, table, rpi, region=None):
    # FlexAttention would add each score's gradient into the small bias table with an atomic add,
    # which took more than half of the attention backward; FlexBiasGrad sums them without atomics
    # and FlexAttention runs on the detached table. The same gradients.
    args = (rpi,) if region is None else (rpi, region)
    if not (table.requires_grad and torch.is_grad_enabled()):
        return fn(q, k, v, table, *args)[0]
    from traiNNer.ops.flex_bias_grad import FlexBiasGrad

    out, lse = fn(q, k, v, table.detach(), *args)
    return FlexBiasGrad.apply(out, table, q, k, v, lse.detach(), rpi, region)


def _flex_window(q, k, v, table, rpi):
    return _flex_bias_grad(_flex_window_lse, q, k, v, table, rpi)


def _flex_window_shifted(q, k, v, table, rpi, region):
    return _flex_bias_grad(_flex_window_shifted_lse, q, k, v, table, rpi, region)


def _flex_oca(q, k, v, table, rpi):
    return _flex_bias_grad(_flex_oca_lse, q, k, v, table, rpi)


def _flex_head_dim(d: int) -> int:
    # FlexAttention wants a power-of-two head dim of at least 16; zero dims change neither q.k nor
    # the outputs that are kept
    return max(16, 1 << (d - 1).bit_length())


def _flex_pad(t: torch.Tensor, d: int) -> torch.Tensor:
    return F.pad(t, (0, d - t.shape[-1])) if d != t.shape[-1] else t


def _flex_index(rpi: torch.Tensor, rows: int) -> torch.Tensor:
    # A relative position index table for score_mod. The OCA table holds negative entries, which the
    # stock path's table[rpi] wraps to the end of the table (trained models rely on it); the kernel's
    # loads don't wrap, so wrap them here. score_mod also runs on the padding of the last 128-token
    # block (masked afterwards): pad to whole blocks so those reads stay in bounds.
    nq, nk = rpi.shape
    rpi = torch.where(rpi < 0, rpi + rows, rpi)
    return F.pad(rpi, (0, -nk % 128, 0, -nq % 128))


def _with_attn_impl(net: HAT, attn_impl: AttnImpl) -> HAT:
    # set after construction, so it stays out of the hyperparameters saved with checkpoints
    if attn_impl not in ("auto", "flex", "stock"):
        raise ValueError(f"attn_impl must be auto, flex or stock, not {attn_impl!r}")
    net.attn_impl = attn_impl
    return net


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)  # type: ignore


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor: int | float = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(
        self,
        num_feat,
        compress_ratio: int | float = 3,
        squeeze_factor: int | float = 30,
    ) -> None:
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(windows, window_size, h, w):
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


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None, flex=False):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            flex: run on FlexAttention; rpi and mask are then HAT.forward_features' flex forms
        """
        if flex:
            return self.forward_flex(x, rpi, mask)
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_flex(self, x, rpi, region):
        """forward() on FlexAttention. rpi: the index table as _flex_index made it; region: the
        shifted-window mask as a region id per window token (num_windows, n'), int32, or None."""
        b_, n, c = x.shape
        d = c // self.num_heads
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, d).permute(2, 0, 3, 1, 4)
        dp = _flex_head_dim(d)
        q, k, v = (_flex_pad(t, dp) for t in (qkv[0] * self.scale, qkv[1], qkv[2]))
        table = (
            self.relative_position_bias_table * 1
        )  # a plain tensor: dynamic shape (see _flex_window)
        if region is None:
            out = _flex_window(q, k, v, table, rpi)
        else:
            out = _flex_window_shifted(q, k, v, table, rpi, region)
        out = out[..., :d]
        x = out.transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class HAB(nn.Module):
    r"""Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        compress_ratio: int | float = 3,
        squeeze_factor: int | float = 30,
        conv_scale=0.01,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must in 0-window_size"
        )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.conv_scale = conv_scale
        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, x_size, rpi_sa, attn_mask, flex=False):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, c
        )  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask, flex=flex)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        overlap_ratio,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1)
                * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU
        )

    def forward(self, x, x_size, rpi, flex=False):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c)
        q = qkv[..., 0, :]

        # partition windows
        q_windows = window_partition(
            q, self.window_size
        )  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(
            -1, self.window_size * self.window_size, c
        )  # nw*b, window_size*window_size, c

        ow, stride = self.overlap_win_size, self.window_size
        if self.training:
            pad = (ow - stride) // 2
            kv = F.pad(qkv[..., 1:, :], (0, 0, 0, 0, pad, pad, pad, pad))
            kv_windows = (
                kv.unfold(1, ow, stride)
                .unfold(2, ow, stride)
                .permute(3, 0, 1, 2, 5, 6, 4)
                .reshape(2, -1, ow * ow, c)
            )
        else:
            kv = qkv[..., 1:, :].permute(0, 3, 4, 1, 2).reshape(b, 2 * c, h, w)
            kv_windows = rearrange(
                self.unfold(kv),
                "b (nc ch oh ow) nw -> nc (b nw) (oh ow) ch",
                nc=2,
                ch=c,
                oh=ow,
                ow=ow,
            ).contiguous()

        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d

        if flex:  # rpi: the index table as _flex_index made it
            dp = _flex_head_dim(d)
            out = _flex_oca(
                _flex_pad(q * self.scale, dp),
                _flex_pad(k, dp),
                _flex_pad(v, dp),
                self.relative_position_bias_table
                * 1,  # a plain tensor (see _flex_window)
                rpi,
            )
            attn_windows = out[..., :d].transpose(1, 2).reshape(b_, nq, self.dim)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[
                rpi.view(-1)
            ].view(
                self.window_size * self.window_size,
                self.overlap_win_size * self.overlap_win_size,
                -1,
            )  # ws*ws, wse*wse, nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, ws*ws, wse*wse
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, self.dim
        )
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):
    """A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        compress_ratio: int | float,
        squeeze_factor: int | float,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # OCAB
        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        use_chk = self.use_checkpoint and self.training

        for blk in self.blocks:
            if use_chk:
                x = checkpoint.checkpoint(
                    blk,
                    x,
                    x_size,
                    params["rpi_sa"],
                    params["attn_mask"],
                    params["flex"],
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    x_size,
                    params["rpi_sa"],
                    params["attn_mask"],
                    params["flex"],
                )

        if use_chk:
            x = checkpoint.checkpoint(
                self.overlap_attn,
                x,
                x_size,
                params["rpi_oca"],
                params["flex"],
                use_reentrant=False,
            )
        else:
            x = self.overlap_attn(x, x_size, params["rpi_oca"], params["flex"])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        compress_ratio: int | float,
        squeeze_factor: int | float,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ) -> None:
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size, params):
        return (
            self.patch_embed(
                self.conv(
                    self.patch_unembed(self.residual_group(x, x_size, params), x_size)
                )
            )
            + x
        )


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],  # type: ignore
            img_size[1] // patch_size[1],  # type: ignore
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],  # type: ignore
            img_size[1] // patch_size[1],  # type: ignore
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        )  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super().__init__(*m)


@store_hyperparameters()
class HAT(nn.Module):
    r"""Hybrid Attention Transformer
        A PyTorch implementation of : `Activating More Pixels in Image Super-Resolution Transformer`.
        Some codes are based on SwinIR.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        compress_ratio: int | float = 3,
        squeeze_factor: int | float = 30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=1,
        img_range=1.0,
        upsampler: Literal["pixelshuffle"] = "pixelshuffle",
        resi_connection="1conv",
        num_feat=64,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.attn_impl: AttnImpl = "auto"  # see AttnImpl; the factories set it
        self._training_mask = None
        self.attn_drop_rate = attn_drop_rate

        num_in_ch = in_chans
        num_out_ch = in_chans
        # num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)
        self.register_buffer("relative_position_index_OCA", relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Hybrid Attention Groups (RHAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(
                        depths[: i_layer + 1]
                    )  # type: ignore
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = (
            coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        )  # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += (
            window_size_ori - window_size_ext + 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size, dtype, device=None):
        # calculate attention mask for SW-MSA
        mask_windows = self.calculate_mask_windows(x_size, dtype, device)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

        return attn_mask

    def calculate_mask_windows(self, x_size, dtype, device=None):
        # the region of each token of each shifted window (num_windows, window_size*window_size): the
        # SW-MSA mask is -100 between tokens of different regions
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1), dtype=dtype, device=device)  # 1 h w 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nw, window_size, window_size, 1
        return mask_windows.view(-1, self.window_size * self.window_size)

    def check_image_size(self, x):
        return pad_to_multiple(x, self.window_size, mode="reflect")

    def use_flex(self, x) -> bool:
        """Whether this forward runs the attention on FlexAttention (see AttnImpl)."""
        if self.attn_impl == "stock" or torch.onnx.is_in_onnx_export():
            return False
        if self.attn_impl == "flex":
            if self.attn_drop_rate > 0:
                raise ValueError(
                    "attn_impl flex does not support attention dropout (attn_drop_rate > 0)"
                )
            return True
        return (
            self.training
            and x.is_cuda
            and self.attn_drop_rate == 0
            and not torch.are_deterministic_algorithms_enabled()
            and torch.cuda.get_device_capability(x.device)[0] >= 8
        )

    def _shift_mask(self, x, x_size, flex):
        key = (x_size, x.device, x.dtype, flex)
        cache = (
            self.training
            and torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )
        if cache and self._training_mask is not None and self._training_mask[0] == key:
            return self._training_mask[1]
        if flex:
            region = self.calculate_mask_windows(x_size, torch.int32, x.device)
            mask = F.pad(region, (0, -region.shape[1] % 128))
        else:
            mask = self.calculate_mask(x_size, x.dtype, x.device)
        if cache:
            self._training_mask = (key, mask)
        return mask

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-consuming for large window size.
        if self.use_flex(x):
            # the same mask and indices in score_mod's terms: a region id per window token (padded to
            # whole 128-token blocks), the index tables wrapped and padded (_flex_index)
            ws = self.window_size
            ows = ws + int(self.overlap_ratio * ws)
            params = {
                "attn_mask": self._shift_mask(x, x_size, True),
                "rpi_sa": _flex_index(
                    self.relative_position_index_SA, (2 * ws - 1) ** 2
                ),
                "rpi_oca": _flex_index(
                    self.relative_position_index_OCA, (ws + ows - 1) ** 2
                ),
                "flex": True,
            }
        else:
            params = {
                "attn_mask": self._shift_mask(x, x_size, False),
                "rpi_sa": self.relative_position_index_SA,
                "rpi_oca": self.relative_position_index_OCA,
                "flex": False,
            }

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * self.img_range
        x = self.check_image_size(x)

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + mean

        return x[:, :, : H * self.upscale, : W * self.upscale]


@ARCH_REGISTRY.register()
def hat_l(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    attn_impl: AttnImpl = "auto",
) -> HAT:
    net = HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
    return _with_attn_impl(net, attn_impl)


@ARCH_REGISTRY.register()
def hat_m(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    attn_impl: AttnImpl = "auto",
) -> HAT:
    net = HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
    return _with_attn_impl(net, attn_impl)


@ARCH_REGISTRY.register()
def hat_s(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 144,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 24,
    squeeze_factor: float = 24,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    attn_impl: AttnImpl = "auto",
) -> HAT:
    net = HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
    return _with_attn_impl(net, attn_impl)


@ARCH_REGISTRY.register()
def hat_l_aligned(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 192,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    attn_impl: AttnImpl = "auto",
) -> HAT:
    net = HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
    return _with_attn_impl(net, attn_impl)


@ARCH_REGISTRY.register()
def hat_m_aligned(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 192,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    attn_impl: AttnImpl = "auto",
) -> HAT:
    net = HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
    return _with_attn_impl(net, attn_impl)
