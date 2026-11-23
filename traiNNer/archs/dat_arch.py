# ruff: noqa
# type: ignore
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from spandrel.util import store_hyperparameters
from spandrel.util.timm import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from torch.utils import checkpoint

# from spandrel.architectures.DAT import DAT
from traiNNer.ops.flex_bias_grad import FlexBiasGrad
from traiNNer.utils.registry import SPANDREL_REGISTRY

# attn_impl: "flex" runs the spatial window attention on FlexAttention: the same function as the
# stock path, with the dynamic position bias and the shifted-window mask applied inside a fused
# kernel, so the [windows, heads, n, n] scores are never materialised, and the bias table's
# gradient summed by FlexBiasGrad. "auto" (the default) uses it while training on a CUDA GPU
# (sm_80+) with no attention dropout and deterministic algorithms off; inference, validation and
# ONNX export always use the stock path. "stock" never uses it. The weights are the same either way.
AttnImpl = Literal["auto", "flex", "stock"]


# One compiled function per attention kind, each with its own compile cache (see hat_arch): q comes
# pre-scaled and the bias table as a plain tensor, with dynamic shapes, so one kernel serves every
# batch and window count per head dim and dtype.
@torch.compile(dynamic=True)
def _flex_window(q, k, v, table, rpi):
    def score_mod(score, b, h, qi, ki):
        return score + table[rpi[qi, ki], h]

    out, aux = flex_attention(
        q, k, v, score_mod=score_mod, scale=1.0, return_aux=AuxRequest(lse=True)
    )
    return out, aux.lse


@torch.compile(dynamic=True)
def _flex_window_shifted(q, k, v, table, rpi, region):
    # plus the shifted-window mask: -100 between tokens of different regions, as the stock mask
    def score_mod(score, b, h, qi, ki):
        w = b % region.shape[0]
        same = region[w, qi] == region[w, ki]
        return score + table[rpi[qi, ki], h] + torch.where(same, 0.0, -100.0)

    out, aux = flex_attention(
        q, k, v, score_mod=score_mod, scale=1.0, return_aux=AuxRequest(lse=True)
    )
    return out, aux.lse


def _flex_pad_tokens(t: torch.Tensor, *dims: int) -> torch.Tensor:
    # score_mod also runs on the padding of the last 128-token block (masked afterwards): pad the
    # index tables to whole blocks so those reads stay in bounds
    pad = []
    for n in reversed(dims):
        pad += [0, -n % 128]
    return F.pad(t, pad)


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class SpatialGate(nn.Module):
    """Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, _N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).reshape(B, H, W, C // 2).permute(0, 3, 1, 2))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(nn.Module):
    """Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(
        self,
        dim,
        idx,
        split_size=[8, 8],
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            raise ValueError(f"ERROR MODE: {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(
                torch.meshgrid([position_bias_h, position_bias_w], indexing="ij")
            )
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            n = self.H_sp * self.W_sp
            self.register_buffer(
                "flex_rpi",
                _flex_pad_tokens(relative_position_index.to(torch.int32), n, n),
                persistent=False,
            )
        self.flex = False  # set by DAT.forward

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, _N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, qkv, H, W, mask=None, region=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size; on FlexAttention
        the shifted-window mask comes as region, a region id per window token
        Output: x (B, H, W, C)
        """
        _, B, L, C = qkv.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # Pack all three projections directly into [window, head, token, channel].
        wh, ww, heads = self.H_sp, self.W_sp, self.num_heads
        q, k, v = (
            qkv.reshape(3, B, H // wh, wh, W // ww, ww, heads, C // heads)
            .permute(0, 1, 2, 4, 6, 3, 5, 7)
            .reshape(3, -1, heads, wh * ww, C // heads)
            .unbind(0)
        )

        if self.flex and self.position_bias:
            return self.forward_flex(q, k, v, H, W, region)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(
            -1, self.H_sp * self.W_sp, C
        )  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x

    def forward_flex(self, q, k, v, H, W, region):
        """forward() from q, k, v on FlexAttention. region: the shifted-window mask as a region
        id per window token (num_windows, n'), int32, or None."""
        n, d = q.shape[2], q.shape[3]
        dp = max(
            16, 1 << (d - 1).bit_length()
        )  # a power-of-two head dim; zero dims change nothing
        q, k, v = (
            F.pad(t, (0, dp - d)) if dp != d else t for t in (q * self.scale, k, v)
        )
        table = self.pos(self.rpe_biases)  # (2Gh-1) * (2Gw-1), heads
        if table.requires_grad and torch.is_grad_enabled():
            fn_table = table.detach()
        else:
            fn_table = table
        if region is None:
            out, lse = _flex_window(q, k, v, fn_table, self.flex_rpi)
        else:
            out, lse = _flex_window_shifted(q, k, v, fn_table, self.flex_rpi, region)
        if fn_table is not table:
            # FlexAttention would add each score's gradient into the table with an atomic add
            out = FlexBiasGrad.apply(
                out, table, q, k, v, lse.detach(), self.flex_rpi, region
            )
        x = out[..., :d].transpose(1, 2).reshape(-1, n, d * self.num_heads)
        return windows2img(x, self.H_sp, self.W_sp, H, W)


class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        # shift masks for training sizes other than reso, built once per size
        self._train_masks: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # the same as region ids per window token, for FlexAttention
        self._flex_regions: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], (
            "shift_size must in 0-split_size0"
        )
        assert 0 <= self.shift_size[1] < self.split_size[1], (
            "shift_size must in 0-split_size1"
        )

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList(
            [
                Spatial_Attention(
                    dim // 2,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(
                self.patches_resolution, self.patches_resolution
            )
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def calculate_regions(self, H, W, device):
        """calculate_mask's masks as the region id of each token of each window, (nW, n'),
        int32 (-100 between tokens of different regions), padded to whole 128-token blocks."""
        key = (H, W, device)
        regions = self._flex_regions.get(key)
        if regions is None:
            regions = []
            (s0, s1), (t0, t1) = self.split_size, self.shift_size
            # window-0 is s0 x s1 shifted by (t0, t1); window-1 the transpose (calculate_mask)
            for sh, sw, shift_h, shift_w in ((s0, s1, t0, t1), (s1, s0, t1, t0)):
                img = torch.zeros((1, H, W, 1))
                cnt = 0
                for h in (slice(0, -sh), slice(-sh, -shift_h), slice(-shift_h, None)):
                    for w in (
                        slice(0, -sw),
                        slice(-sw, -shift_w),
                        slice(-shift_w, None),
                    ):
                        img[:, h, w, :] = cnt
                        cnt += 1
                win = (
                    img.view(1, H // sh, sh, W // sw, sw, 1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(-1, sh * sw)
                )
                regions.append(
                    _flex_pad_tokens(win.to(device=device, dtype=torch.int32), sh * sw)
                )
            regions = tuple(regions)
            if self.training and len(self._flex_regions) < 4:
                self._flex_regions[key] = regions
        return regions

    def calculate_mask(self, H, W, dtype=None):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1), dtype=dtype)  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1), dtype=dtype)  # 1 H W 1 idx=1
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(
            1,
            H // self.split_size[0],
            self.split_size[0],
            W // self.split_size[1],
            self.split_size[1],
            1,
        )
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, -100.0).masked_fill(
            attn_mask_0 == 0, 0.0
        )

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(
            1,
            H // self.split_size[1],
            self.split_size[1],
            W // self.split_size[0],
            self.split_size[0],
            1,
        )
        img_mask_1 = (
            img_mask_1.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)
        )  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, -100.0).masked_fill(
            attn_mask_1 == 0, 0.0
        )

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        v = qkv[2].permute(0, 3, 1, 2)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        if not self.training or pad_r or pad_b:
            qkv = F.pad(qkv, (0, 0, 0, pad_r, 0, pad_b))
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        qkv = qkv.reshape(3, B, _L, C)

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            )
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            )
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.attns[0].flex:
                regions = self.calculate_regions(_H, _W, x.device)
                x1_shift = self.attns[0](qkv_0, _H, _W, region=regions[0])
                x2_shift = self.attns[1](qkv_1, _H, _W, region=regions[1])
            elif self.patches_resolution != _H or self.patches_resolution != _W:
                key = (_H, _W, x.dtype, x.device)
                masks = self._train_masks.get(key)
                if masks is None:
                    mask_tmp = self.calculate_mask(_H, _W, dtype=x.dtype)
                    masks = (mask_tmp[0].to(x.device), mask_tmp[1].to(x.device))
                    # Training uses a few fixed sizes. Inference sizes vary, and
                    # their masks can be large, so they aren't kept.
                    if self.training and len(self._train_masks) < 4:
                        self._train_masks[key] = masks
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=masks[0])
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=masks[1])
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        # convolution output
        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = (
            self.channel_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, 1, C)
        )
        # S-Map (before sigmoid)
        attention_reshape = attened_x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = (
            self.spatial_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, N, 1)
        )

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DATB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[2, 4],
        shift_size=[1, 2],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if b_idx % 2 == 0:
            # DSTB
            self.attn = Adaptive_Spatial_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )
        else:
            # DCTB
            self.attn = Adaptive_Channel_Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x


class ResidualGroup(nn.Module):
    """ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=[2, 4],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        use_chk=False,
        resi_connection="1conv",
        rg_idx=0,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList(
            [
                DATB(
                    dim=dim,
                    num_heads=num_heads,
                    reso=reso,
                    split_size=split_size,
                    shift_size=[split_size[0] // 2, split_size[1] // 2],
                    expansion_factor=expansion_factor,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_paths[i],  # type: ignore
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rg_idx=rg_idx,
                    b_idx=i,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
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


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution):
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


def _with_attn_impl(net: DAT, attn_impl: AttnImpl) -> DAT:
    # set after construction, so it stays out of the hyperparameters saved with checkpoints
    if attn_impl not in ("auto", "flex", "stock"):
        raise ValueError(f"attn_impl must be auto, flex or stock, not {attn_impl!r}")
    net.attn_impl = attn_impl
    return net


@store_hyperparameters()
class DAT(nn.Module):
    """Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        img_size=64,
        in_chans=3,
        embed_dim=180,
        split_size=[2, 4],
        depth=[2, 2, 2, 2],
        num_heads=[2, 2, 2, 2],
        expansion_factor=4.0,
        qkv_bias=True,
        qk_scale: float | None = None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_chk=False,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
        unshuffle_mod=False,
    ):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.pad = 0
        self.upscale = upscale
        self.upsampler = upsampler
        self.attn_drop_rate = attn_drop_rate
        self.attn_impl: AttnImpl = "auto"  # see AttnImpl; the factories set it

        if unshuffle_mod and upscale < 3:
            unshuffle = 4 // upscale
            upscale = 4
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(num_in_ch * unshuffle**2, embed_dim, 3, 1, 1),
            )
            self.pad = unshuffle
        else:
            # ------------------------- 1, Shallow Feature Extraction ------------------------- #
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, Reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def use_flex(self, x) -> bool:
        """Whether this forward runs the spatial attention on FlexAttention (see AttnImpl)."""
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

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        if self.pad == 0:
            return x
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        _b, _c, h, w = x.shape
        flex = self.use_flex(x)
        if getattr(self, "_flex_on", None) is not flex:
            for m in self.modules():
                if isinstance(m, Spatial_Attention):
                    m.flex = flex
            self._flex_on = flex
        x = self.check_img_size(x, h, w)
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for image SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + mean
        return x[:, :, : h * self.upscale, : w * self.upscale]


@SPANDREL_REGISTRY.register()
def dat(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 32),
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    expansion_factor: int = 4,
    resi_connection: str = "1conv",
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    upsampler: str = "pixelshuffle",
    unshuffle_mod: bool = False,
    attn_impl: AttnImpl = "auto",
) -> DAT:
    net = DAT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
    return _with_attn_impl(net, attn_impl)


@SPANDREL_REGISTRY.register()
def dat_s(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 16),
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    expansion_factor: int = 2,
    resi_connection: str = "1conv",
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    upsampler: str = "pixelshuffle",
    unshuffle_mod: bool = False,
    attn_impl: AttnImpl = "auto",
) -> DAT:
    net = DAT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
    return _with_attn_impl(net, attn_impl)


@SPANDREL_REGISTRY.register()
def dat_2(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 32),
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    expansion_factor: int = 2,
    resi_connection: str = "1conv",
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    upsampler: str = "pixelshuffle",
    unshuffle_mod: bool = False,
    attn_impl: AttnImpl = "auto",
) -> DAT:
    net = DAT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
    return _with_attn_impl(net, attn_impl)


@SPANDREL_REGISTRY.register()
def dat_2_l(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 32),
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    expansion_factor: int = 2,
    resi_connection: str = "1conv",
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    upsampler: str = "pixelshuffle",
    unshuffle_mod: bool = False,
    attn_impl: AttnImpl = "auto",
) -> DAT:
    net = DAT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
    return _with_attn_impl(net, attn_impl)


@SPANDREL_REGISTRY.register()
def dat_light(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 32),
    depth: Sequence[int] = (18,),
    embed_dim: int = 60,
    num_heads: Sequence[int] = (6,),
    expansion_factor: int = 2,
    resi_connection: str = "3conv",
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    upsampler: str = "pixelshuffle",
    unshuffle_mod: bool = False,
    attn_impl: AttnImpl = "auto",
) -> DAT:
    net = DAT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
    return _with_attn_impl(net, attn_impl)
