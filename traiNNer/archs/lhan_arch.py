# https://github.com/stinkybread/LHAN/blob/main/lhan.py
# ruff: noqa
# type: ignore
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY


# --- Helper Functions ---
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low, up = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1).erfinv_().mul_(
            std * math.sqrt(2.0)
        ).add_(mean).clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# --- Core Components ---
class DropPath(nn.Module):
    def __init__(self, drop_prob=None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor_()


class SimpleLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight, self.bias, self.eps = (
            nn.Parameter(torch.ones(dim)),
            nn.Parameter(torch.zeros(dim)),
            eps,
        )

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


# --- Upsamplers ---
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, c_in, c_out, sf) -> None:
        super().__init__()
        self.conv_pre = nn.Conv2d(c_in, c_out * (sf**2), 3, 1, 1)
        self.ps, self.conv_post = nn.PixelShuffle(sf), nn.Conv2d(c_out, c_out, 3, 1, 1)

    def forward(self, x):
        return self.conv_post(self.ps(self.conv_pre(x)))


class NearestConvUpsampler(nn.Module):
    def __init__(self, c_in, c_out, sf) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=sf, mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1), nn.GELU(), nn.Conv2d(c_in, c_out, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv(self.up(x))


class TransposeConvUpsampler(nn.Module):
    def __init__(self, c_in, c_out, sf) -> None:
        super().__init__()
        if sf == 2:
            self.up = nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)
        elif sf == 3:
            self.up = nn.ConvTranspose2d(c_in, c_out, 3, 3, 0)
        elif sf == 4:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, 4, 2, 1),
                nn.GELU(),
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {sf}")
        self.refine = nn.Conv2d(c_out, c_out, 3, 1, 1)

    def forward(self, x):
        return self.refine(self.up(x))


# --- LHAN Components ---
class FastSpatialWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.dim, self.ws, self.nh = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )
        self.bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        trunc_normal_(self.bias, std=0.02)

    def forward(self, x, H, W):
        B, L, C = x.shape
        pad_r, pad_b = (
            (self.ws - W % self.ws) % self.ws,
            (self.ws - H % self.ws) % self.ws,
        )
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b)).view(B, -1, C)

        H_pad, W_pad = H + pad_b, W + pad_r
        x = (
            x.view(B, H_pad // self.ws, self.ws, W_pad // self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.ws * self.ws, C)
        )
        qkv = (
            self.qkv(x)
            .view(-1, self.ws * self.ws, 3, self.nh, C // self.nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1)) + self.bias
        x = (
            (F.softmax(attn, dim=-1) @ v)
            .transpose(1, 2)
            .reshape(-1, self.ws * self.ws, C)
        )
        x = (
            self.proj(x)
            .view(B, H_pad // self.ws, W_pad // self.ws, self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H_pad, W_pad, C)
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.view(B, L, C)


class FastChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )

    def forward(self, x, H, W):  # H, W are unused but kept for API consistency
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = (
            F.normalize(q.transpose(-2, -1), dim=-1),
            F.normalize(k.transpose(-2, -1), dim=-1),
        )
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        return self.proj(
            (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        )


class SimplifiedAIM(nn.Module):
    def __init__(self, dim, reduction_ratio=8) -> None:
        super().__init__()
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, attn_feat, conv_feat, interaction_type, H, W):
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = (
                self.sg(attn_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, 1, L)
                .transpose(1, 2)
            )
            return attn_feat + (conv_feat * sm)
        else:
            cm = (
                self.cg(conv_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, C, 1)
                .transpose(1, 2)
            )
            return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0) -> None:
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1, self.act, self.fc2 = (
            nn.Linear(dim, hd, False),
            nn.GELU(),
            nn.Linear(hd, dim, False),
        )
        self.drop = nn.Dropout(drop)
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        x_s = (
            self.smix(x.transpose(1, 2).view(B, x.shape[-1], H, W))
            .view(B, x.shape[-1], L)
            .transpose(1, 2)
        )
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    def __init__(self, dim, nh, ws, ffn_exp, aim_re, btype, dp, qkv_b=False) -> None:
        super().__init__()
        self.btype = btype
        self.n1, self.n2 = SimpleLayerNorm(dim), SimpleLayerNorm(dim)
        self.attn = (
            FastSpatialWindowAttention(dim, ws, nh, qkv_b)
            if btype == "spatial"
            else FastChannelAttention(dim, nh, qkv_b)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU()
        )
        self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x, H, W):
        B, L, C = x.shape
        return (
            self.conv(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)
        )

    def forward(self, x, H, W):
        n1 = self.n1(x)
        itype = (
            "channel_modulates_spatial"
            if self.btype == "spatial"
            else "spatial_modulates_channel"
        )
        fused = self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W)
        x = x + self.dp(fused)
        x = x + self.dp(self.ffn(self.n2(x), H, W))
        return x


class SimplifiedResidualGroup(nn.Module):
    def __init__(self, dim, depth, nh, ws, ffn_exp, aim_re, pattern, dp_rates) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SimplifiedDATBlock(
                    dim, nh, ws, ffn_exp, aim_re, pattern[i % len(pattern)], dp_rates[i]
                )
                for i in range(depth)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x, H, W):
        B, C, _, _ = x.shape
        x_seq = x.view(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).view(B, C, H, W)) + x


@ARCH_REGISTRY.register()
class lhan(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        embed_dim: int = 120,
        num_groups: int = 4,
        depth_per_group: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: list[str] | None = None,
        drop_path_rate: float = 0.1,
        upsampler_type: str = "transpose_conv",
        img_range: float = 1.0,
    ) -> None:
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel"]
        super().__init__()
        self.img_range, self.upscale = img_range, scale
        self.mean = torch.zeros(1, 1, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=True)
        ad = depth_per_group * len(group_block_pattern)
        td = num_groups * ad
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, td)]
        self.groups = nn.ModuleList(
            [
                SimplifiedResidualGroup(
                    embed_dim,
                    ad,
                    num_heads,
                    window_size,
                    ffn_expansion_ratio,
                    aim_reduction_ratio,
                    group_block_pattern,
                    dpr[i * ad : (i + 1) * ad],
                )
                for i in range(num_groups)
            ]
        )
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        upsampler_map = {
            "pixelshuffle": PixelShuffleUpsampler,
            "nearest_conv": NearestConvUpsampler,
            "transpose_conv": TransposeConvUpsampler,
        }
        self.upsampler = upsampler_map[upsampler_type](embed_dim, num_out_ch, scale)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, SimpleLayerNorm | nn.LayerNorm | nn.GroupNorm):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        self.mean = self.mean.type_as(x)
        x_norm = (x - self.mean) * self.img_range
        x_shallow = self.conv_first(x_norm)
        x_deep = x_shallow
        for group in self.groups:
            x_deep = group(x_deep, H, W)
        x_deep = self.conv_after(x_deep)
        x_out = self.upsampler(x_deep + x_shallow)
        return x_out / self.img_range + self.mean


@ARCH_REGISTRY.register()
def lhan_tiny(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    embed_dim: int = 96,
    num_groups: int = 2,
    depth_per_group: int = 2,
    num_heads: int = 3,
    window_size: int = 8,
    ffn_expansion_ratio: float = 1.5,
    aim_reduction_ratio: int = 8,
    group_block_pattern: list[str] | None = None,
    drop_path_rate: float = 0.05,
    upsampler_type: str = "pixelshuffle",
    img_range: float = 1.0,
):
    return lhan(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
        img_range=img_range,
    )


@ARCH_REGISTRY.register()
def lhan_light(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    embed_dim: int = 108,
    num_groups: int = 3,
    depth_per_group: int = 2,
    num_heads: int = 4,
    window_size: int = 8,
    ffn_expansion_ratio: float = 2.0,
    aim_reduction_ratio: int = 8,
    group_block_pattern: list[str] | None = None,
    drop_path_rate: float = 0.08,
    upsampler_type: str = "pixelshuffle",
    img_range: float = 1.0,
):
    return lhan(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
        img_range=img_range,
    )


@ARCH_REGISTRY.register()
def lhan_medium(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    embed_dim: int = 120,
    num_groups: int = 4,
    depth_per_group: int = 3,
    num_heads: int = 4,
    window_size: int = 8,
    ffn_expansion_ratio: float = 2.0,
    aim_reduction_ratio: int = 8,
    group_block_pattern: list[str] | None = None,
    drop_path_rate: float = 0.1,
    upsampler_type: str = "transpose_conv",
    img_range: float = 1.0,
):
    return lhan(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
        img_range=img_range,
    )
