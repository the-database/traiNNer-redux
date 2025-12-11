import sys
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from spandrel import StateDict
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.modules.module import _IncompatibleKeys

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.utils.misc import require_triton
from traiNNer.utils.registry import ARCH_REGISTRY

ATTN_TYPE = Literal["Naive", "SDPA", "Flex"]
"""
Naive Self-Attention:
    - Numerically stable
    - Choose this for train if you have enough time and GPUs
    - Training ESC with Naive Self-Attention: 33.46dB @Urban100x2

Flex Attention:
    - Fast and memory efficient
    - Choose this for train/test if you are using Linux OS
    - Training ESC with Flex Attention: 33.44dB @Urban100x2

SDPA with memory efficient kernel:
    - Memory efficient (not fast)
    - Choose this for train/test if you are using Windows OS
    - Training ESC with SDPA: 33.43dB @Urban100x2
"""


def attention(q: Tensor, k: Tensor, v: Tensor, bias: Tensor) -> Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(
    table: Tensor, window_size: int
) -> Callable[[Tensor, int, int, int, int], Tensor]:
    def bias_mod(score: Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> Tensor:
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]

    return bias_mod


def feat_to_win(x: Tensor, window_size: Sequence[int], heads: int) -> Tensor:
    return rearrange(
        x,
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c",
        heads=heads,
        wh=window_size[0],
        ww=window_size[1],
        qkv=3,
    )


def win_to_feat(
    x: Tensor, window_size: Sequence[int], h_div: int, w_div: int
) -> Tensor:
    return rearrange(
        x,
        "(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)",
        h=h_div,
        w=w_div,
        wh=window_size[0],
        ww=window_size[1],
    )


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["channels_first", "channels_last"] = "channels_first",
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
        elif self.data_format == "channels_first":
            if self.training:
                return (
                    F.layer_norm(
                        x.permute(0, 2, 3, 1).contiguous(),
                        self.normalized_shape,
                        self.weight,
                        self.bias,
                        self.eps,
                    )
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                return F.layer_norm(
                    x.permute(0, 2, 3, 1),
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int) -> None:
        super().__init__()
        self.pdim = pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0),
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)  # pyright: ignore[reportArgumentType]
        nn.init.zeros_(self.dwc_proj[-1].bias)  # pyright: ignore[reportArgumentType]

    def forward(self, x: Tensor, lk_filter: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x1_ = rearrange(x1, "b c h w -> 1 (b c) h w")
            x1_ = F.conv2d(
                x1_,
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=bs * self.pdim,
            )
            x1_ = rearrange(x1_, "1 (b c) h w -> b c h w", b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = (
                F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2)
                + x1_
            )

            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x[:, : self.pdim] = F.conv2d(
                x[:, : self.pdim], lk_filter, stride=1, padding=13 // 2
            ) + F.conv2d(
                x[:, : self.pdim],
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=self.pdim,
            )

            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x

    def extra_repr(self) -> str:
        return f"pdim={self.pdim}"


class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int) -> None:
        super().__init__()
        self.plk = ConvolutionalAttention(pdim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: Tensor, lk_filter: Tensor) -> Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim * exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(
            int(dim * exp_ratio),
            int(dim * exp_ratio),
            kernel_size,
            1,
            kernel_size // 2,
            groups=int(dim * exp_ratio),
        )
        self.aggr = nn.Conv2d(int(dim * exp_ratio), dim, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int | tuple[int, int],
        num_heads: int,
        attn_func: Callable,
        attn_type: ATTN_TYPE = "Flex",
    ) -> None:
        super().__init__()
        self.dim = dim
        window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_type = attn_type
        self.attn_func = attn_func
        self.relative_position_bias = nn.Parameter(
            torch.randn(
                num_heads, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
            ).to(torch.float32)
            * 0.001
        )
        if self.attn_type == "Flex":
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        self.is_mobile = False

    @staticmethod
    def create_table_idxs(window_size: int, heads: int) -> Tensor:
        # Transposed idxs of original Swin Transformer
        # But much easier to implement and the same relative position distance anyway
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)
        return idxs

    def pad_to_win(self, x: Tensor, h: int, w: int) -> Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def to_mobile(self) -> None:
        bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
        self.rpe_bias = nn.Parameter(  # pyright: ignore[reportUninitializedInstanceVariable]
            bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
        )

        del self.relative_position_bias
        del self.rpe_idxs

        self.is_mobile = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = (
            x.shape[2] // self.window_size[0],
            x.shape[3] // self.window_size[1],
        )

        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_win(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == "Flex":
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        elif self.attn_type == "SDPA":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)
        elif self.attn_type == "Naive":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = self.attn_func(q, k, v, bias)
        else:
            raise NotImplementedError(
                f"Attention type {self.attn_type} is not supported."
            )

        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        pdim: int,
        conv_blocks: int,
        window_size: int,
        num_heads: int,
        exp_ratio: float,
        attn_func: Callable,
        attn_type: ATTN_TYPE = "Flex",
    ) -> None:
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2)

        self.ln_attn = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type)

        self.lns = nn.ModuleList([LayerNorm(dim) for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList(
            [ConvAttnWrapper(dim, pdim) for _ in range(conv_blocks)]
        )
        self.convffns = nn.ModuleList(
            [ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)]
        )

        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: Tensor, plk_filter: Tensor) -> Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for ln, pconv, convffn in zip(
            self.lns, self.pconvs, self.convffns, strict=False
        ):
            x = x + pconv(convffn(ln(x)), plk_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


# To enhance LK's structural inductive bias, we use Feature-level Geometric Re-parameterization
#  as proposed in https://github.com/dslisleedh/IGConv
def _geo_ensemble(k: Tensor) -> Tensor:
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (
        k
        + k_hflip
        + k_vflip
        + k_hvflip
        + k_rot90
        + k_rot90_hflip
        + k_rot90_vflip
        + k_rot90_hvflip
    ) / 8
    return k


@ARCH_REGISTRY.register()
class ESCRealM(nn.Module):
    """
    ESC-Real for multiple upscaling factor and various options for upsampling module.
    For more information, please refer to this issue: https://github.com/dslisleedh/ESC/issues/2
    """

    def __init__(
        self,
        dim: int = 64,
        pdim: int = 16,
        kernel_size: int = 13,
        n_blocks: int = 10,
        conv_blocks: int = 5,
        window_size: int = 32,
        num_heads: int = 4,
        scale: int = 4,
        exp_ratio: float = 2.0,
        attn_type: ATTN_TYPE = "Flex",
        mid_dim: int = 64,
        upsampler: SampleMods3 = "transpose+conv",
        unshuffle_mod: bool = False,
    ) -> None:
        super().__init__()
        self.upscaling_factor = scale

        if attn_type == "Naive":
            attn_func = attention
        elif attn_type == "SDPA":
            attn_func = F.scaled_dot_product_attention
        elif attn_type == "Flex":
            require_triton(
                "The `ESCRealM` architecture requires triton when using Flex attention. "
            )
            attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        self.plk_func = _geo_ensemble

        self.plk_filter = nn.Parameter(
            torch.randn(pdim, pdim, kernel_size, kernel_size)
        )
        torch.nn.init.orthogonal_(self.plk_filter)

        if unshuffle_mod and scale < 3:
            unshuffle_factor = 4 // scale
            scale = 4
            self.proj = nn.Sequential(
                nn.PixelUnshuffle(unshuffle_factor),
                nn.Conv2d(3 * unshuffle_factor**2, dim, 3, 1, 1),
            )
            self.skip = nn.Sequential(
                nn.PixelUnshuffle(unshuffle_factor),
                nn.Conv2d(3 * unshuffle_factor**2, dim * 2, 1, 1, 0),
                nn.Conv2d(
                    dim * 2, dim * 2, 7, 1, 3, groups=dim * 2, padding_mode="reflect"
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * 2, dim, 1, 1, 0),
            )
            self.padding = unshuffle_factor
        else:
            self.proj = nn.Conv2d(3, dim, 3, 1, 1)
            self.skip = nn.Sequential(
                nn.Conv2d(3, dim * 2, 1, 1, 0),
                nn.Conv2d(
                    dim * 2, dim * 2, 7, 1, 3, groups=dim * 2, padding_mode="reflect"
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * 2, dim, 1, 1, 0),
            )
            self.padding = 0
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    pdim,
                    conv_blocks,
                    window_size,
                    num_heads,
                    exp_ratio,
                    attn_func,
                    attn_type,
                )
                for _ in range(n_blocks)
            ]
        )
        self.last = nn.Conv2d(dim, dim, 3, 1, 1)

        self.to_img = UniUpsampleV3(upsampler, scale, dim, 3, mid_dim, 4)

    def load_state_dict(
        self,
        state_dict: StateDict,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> _IncompatibleKeys:
        state_dict["to_img.MetaUpsample"] = self.to_img.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        mod_pad_h = (self.padding - h % self.padding) % self.padding
        mod_pad_w = (self.padding - w % self.padding) % self.padding
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        if self.padding:
            x = self.check_img_size(x, h, w)
        feat = self.proj(x)
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        for block in self.blocks:
            feat = block(feat, plk_filter)
        feat = self.last(feat) + skip + self.skip(x)
        x = self.to_img(feat)
        return x[:, :, : h * self.upscaling_factor, : w * self.upscaling_factor]


@ARCH_REGISTRY.register()
def escrealm_xl(
    dim: int = 128,
    pdim: int = 32,
    kernel_size: int = 13,
    n_blocks: int = 16,
    conv_blocks: int = 5,
    window_size: int = 32,
    num_heads: int = 8,
    scale: int = 4,
    exp_ratio: float = 2.0,
    attn_type: ATTN_TYPE = "Flex",
    mid_dim: int = 64,
    upsampler: SampleMods3 = "pixelshuffle",
    unshuffle_mod: bool = False,
) -> ESCRealM:
    return ESCRealM(
        dim=dim,
        pdim=pdim,
        kernel_size=kernel_size,
        n_blocks=n_blocks,
        conv_blocks=conv_blocks,
        window_size=window_size,
        num_heads=num_heads,
        scale=scale,
        exp_ratio=exp_ratio,
        attn_type=attn_type,
        mid_dim=mid_dim,
        upsampler=upsampler,
        unshuffle_mod=unshuffle_mod,
    )
