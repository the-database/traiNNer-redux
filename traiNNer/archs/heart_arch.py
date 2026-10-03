# ruff: noqa
# type: ignore
# HEART - Hybrid Efficient Attention with Rank-factorized bias Transformer
# based on HAT-iLN (arXiv:2504.06629) and SST's RIB (arXiv:2603.06738)
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from einops import rearrange
from spandrel.util import store_hyperparameters
from spandrel.util.timm import trunc_normal_
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils import checkpoint

from traiNNer.archs.arch_util import iLN
from traiNNer.archs.hat_iln_arch import (
    CAB,
    AffineTransform,
    DropPath,
    Mlp,
    PatchEmbed,
    PatchUnEmbed,
    Upsample,
)
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


def _icnr_init_conv(conv: nn.Conv2d, upscale: int) -> None:
    """ICNR init for a Conv2d feeding a PixelShuffle(upscale).

    Every sub-pixel phase starts from the same base kernel, which keeps the
    upsampling head from beginning with a checkerboard bias. Init-only: no
    graph or runtime cost, and no effect once a checkpoint is loaded.
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


def _icnr_init_upsampler(module: nn.Module) -> None:
    """ICNR-init every Conv2d directly followed by a PixelShuffle in ``module``."""
    for _name, m in module.named_modules():
        if not isinstance(m, nn.Sequential):
            continue
        children = list(m)
        for prev, nxt in zip(children, children[1:], strict=False):
            if isinstance(prev, nn.Conv2d) and isinstance(nxt, nn.PixelShuffle):
                _icnr_init_conv(prev, nxt.upscale_factor)


class RIBWindowAttention(nn.Module):
    """Window attention with Rank-factorized Implicit Neural Bias (RIB).

    RIB (SST, arXiv 2603.06738) replaces the relative position bias table:
    position features from a small implicit net are concatenated onto Q/K,
    turning bias addition into a dot product, so attention runs on
    FlashAttention/SDPA kernels with no bias table and no masks.
    Channels-first (B, C, H, W); shifted windows use non-wrapping
    pad-and-partition (no cyclic wrap).
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        shift: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.rank = rank

        # normalized window coordinates
        wh, ww = self.window_size
        yy, xx = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()  # (N, 2)
        coords[:, 0] = (2.0 * (coords[:, 0] + 0.5) / ww) - 1.0
        coords[:, 1] = (2.0 * (coords[:, 1] + 0.5) / wh) - 1.0

        self.n_freqs = rib_n_freqs
        if self.n_freqs > 0:
            base_coords = coords.clone()
            for i in range(self.n_freqs):
                freq = 2**i
                coords = torch.cat(
                    [
                        coords,
                        torch.sin(base_coords * freq),
                        torch.cos(base_coords * freq),
                    ],
                    dim=-1,
                )
        self.register_buffer("rib_coords", coords, persistent=False)

        n_input = 2 + 4 * self.n_freqs
        hidden_d = rib_hidden_dim
        self.to_hidden = nn.Parameter(torch.empty(n_input, hidden_d))
        self.hidden_b = nn.Parameter(torch.zeros(1, hidden_d))
        self.to_q = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))
        self.to_k = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))

        nn.init.normal_(self.to_hidden, mean=0.0, std=0.05)
        nn.init.normal_(self.to_q, mean=0.0, std=0.05)
        nn.init.normal_(self.to_k, mean=0.0, std=0.05)

        self.shift = shift

    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # inputs are pre-aligned to window multiples by HEART.check_image_size
        # (which owns the replicate fallback), so reflect padding is safe here
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def _flash_cat_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q: torch.Tensor,
        pos_k: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v: (Bwin, N, heads, D); pos_q, pos_k: (Bwin, heads, N, R)
        d = q.shape[-1]
        pos_q = pos_q.transpose(1, 2)  # (Bwin, N, heads, R)
        pos_k = pos_k.transpose(1, 2)
        r = pos_q.shape[-1]

        q = q * (d**-0.5)
        pos_q = pos_q * (r**-0.5)
        q_cat = torch.cat([q, pos_q], dim=-1)
        k_cat = torch.cat([k, pos_k], dim=-1)

        d_total = q_cat.shape[-1]
        pad = (8 - (d_total % 8)) % 8
        if pad:
            q_cat = F.pad(q_cat, (0, pad))
            k_cat = F.pad(k_cat, (0, pad))
        v_cat = F.pad(v, (0, r + pad))

        # q and pos_q are pre-scaled (SST eq. 5); scale=1.0 keeps SDPA from
        # rescaling them again
        out = F.scaled_dot_product_attention(
            q_cat.transpose(1, 2),
            k_cat.transpose(1, 2),
            v_cat.transpose(1, 2),
            scale=1.0,
        ).transpose(1, 2)[:, :, :, :d]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)

        # shifted windows: pad top/left by half a window, partition from the
        # padded origin. x is window-aligned here, so padding (sh, sh) per
        # axis keeps alignment (one extra window row/col).
        sh = sw = 0
        if self.shift:
            sh, sw = self.window_size[0] // 2, self.window_size[1] // 2
            x = F.pad(x, (sw, sw, sh, sh), mode="reflect")

        h_div, w_div = (
            x.shape[2] // self.window_size[0],
            x.shape[3] // self.window_size[1],
        )

        qkv = self.to_qkv(x)
        qkv = rearrange(
            qkv,
            "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) (wh ww) heads c",
            heads=self.num_heads,
            wh=self.window_size[0],
            ww=self.window_size[1],
            qkv=3,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (Bwin, N, heads, D)

        # RIB position features (fp32 for fp16-inference stability)
        intermediate = F.relu(
            self.rib_coords.to(dtype=torch.float32)
            @ self.to_hidden.to(dtype=torch.float32)
            + self.hidden_b.to(dtype=torch.float32)
        )  # (N, hidden_d)
        q_pos = torch.einsum(
            "nd,hdr->hnr", intermediate, self.to_q.to(dtype=torch.float32)
        )  # (heads, N, R)
        k_pos = torch.einsum(
            "nd,hdr->hnr", intermediate, self.to_k.to(dtype=torch.float32)
        )
        bwin = q.shape[0]
        q_pos = (
            q_pos.to(q.dtype).unsqueeze(0).expand(bwin, -1, -1, -1)
        )  # (Bwin, heads, N, R)
        k_pos = k_pos.to(k.dtype).unsqueeze(0).expand(bwin, -1, -1, -1)

        out = self._flash_cat_attn(q, k, v, q_pos, k_pos)  # (Bwin, N, heads, D)

        out = rearrange(
            out,
            "(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)",
            h=h_div,
            w=w_div,
            wh=self.window_size[0],
            ww=self.window_size[1],
        )
        out = out[:, :, sh : sh + h, sw : sw + w]
        return self.to_out(out)


class HAB_RIB(nn.Module):
    """Hybrid Attention Block: RIB window attention + CAB conv branch + i-LN.

    The block unpacks ``(x, std)`` from norm1/norm2, so norm_layer must be
    i-LN-compatible; the parameter exists for HAT symmetry, not as a knob.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 32,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = iLN,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        shift: bool = False,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.conv_scale = conv_scale
        self.use_attention = use_attention

        self.norm1 = norm_layer(dim)
        if use_attention:
            self.attn = RIBWindowAttention(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                shift=shift,
            )

        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x, std1 = self.norm1(x)
        x_2d = x.transpose(1, 2).contiguous().view(b, c, h, w)

        # conv branch
        conv_x = self.conv_block(x_2d).view(b, c, -1).transpose(1, 2)
        residual = conv_x * self.conv_scale

        # RIB window attention (channels-first), optional per attention_freq
        if self.use_attention:
            attn_out = self.attn(x_2d)
            attn_x = attn_out.view(b, c, -1).transpose(1, 2)
            residual = residual + self.drop_path(attn_x)

        # i-LN: rescale by std from norm1
        x = shortcut + std1.view(b, 1, 1) * residual

        # FFN with i-LN rescaling
        x_normed, std2 = self.norm2(x)
        ffn_out = self.mlp(x_normed)
        x = x + std2.view(b, 1, 1) * self.drop_path(ffn_out)

        return x


class AttenBlocksRIB(nn.Module):
    """A stack of HAB_RIB blocks with optional gradient checkpointing."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float,
        drop_path: float | list[float],
        norm_layer: type[nn.Module] = iLN,
        use_checkpoint: bool = False,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blocks: list[HAB_RIB] = []
        attn_idx = 0
        for i in range(depth):
            use_attention = i % attention_freq == 0
            blocks.append(
                HAB_RIB(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    rank=rank,
                    rib_hidden_dim=rib_hidden_dim,
                    rib_n_freqs=rib_n_freqs,
                    shift=(attn_idx % 2 == 1) if use_attention else False,
                    use_attention=use_attention,
                )
            )
            if use_attention:
                attn_idx += 1
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        use_chk = self.use_checkpoint and self.training
        for blk in self.blocks:
            if use_chk:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        return x


class RHAG_RIB(nn.Module):
    """Residual Hybrid Attention Group: AttenBlocksRIB -> 3x3 conv -> residual."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float,
        drop_path: float | list[float],
        norm_layer: type[nn.Module] = iLN,
        use_checkpoint: bool = False,
        img_size: int = 64,
        patch_size: int = 1,
        resi_connection: str = "1conv",
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.residual_group = AttenBlocksRIB(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attention_freq=attention_freq,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

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

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )


@store_hyperparameters()
class HEART(nn.Module):
    """HEART - Hybrid Efficient Attention with Rank-factorized bias Transformer.

    Simplified HAT-iLN: RIB window attention (FlashAttention-friendly) replaces
    the relative position bias table and OCAB is removed. i-LN keeps training
    stable in reduced precision. Pure PyTorch, ONNX-exportable.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
        window_size: int = 32,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = iLN,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 1,
        upsampler: Literal["pixelshuffle"] = "pixelshuffle",
        resi_connection: str = "1conv",
        num_feat: int = 64,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        num_in_ch = in_chans
        num_out_ch = in_chans
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=AffineTransform if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=AffineTransform if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG_RIB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                attention_freq=attention_freq,
            )
            self.layers.append(layer)
        self.norm = AffineTransform(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        # ICNR on the PixelShuffle head: init-only checkerboard prevention.
        self._icnr_init()

    def _icnr_init(self) -> None:
        if self.upsampler == "pixelshuffle":
            _icnr_init_upsampler(self.upsample)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, iLN)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        can_reflect = (pad_h == 0 or pad_h < h) and (pad_w == 0 or pad_w < w)
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x = self.check_image_size(x)

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        return x[:, :, : h * self.upscale, : w * self.upscale]


@SPANDREL_REGISTRY.register()
@ARCH_REGISTRY.register()
def heart(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 32,
    compress_ratio: int = 3,
    squeeze_factor: int = 30,
    conv_scale: float = 0.01,
    mlp_ratio: float = 2.0,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
    rank: int = 8,
    rib_hidden_dim: int = 32,
    rib_n_freqs: int = 10,
    attention_freq: int = 2,
) -> HEART:
    return HEART(
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
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
        rank=rank,
        rib_hidden_dim=rib_hidden_dim,
        rib_n_freqs=rib_n_freqs,
        attention_freq=attention_freq,
    )
