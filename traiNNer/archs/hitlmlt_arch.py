# ruff: noqa
# type: ignore
import numpy as np
import torch
from timm.layers import to_2tuple
from torch import nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        return None


class DFE(nn.Module):
    """Dual Feature Extraction
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0),
        )

        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, H * W).permute(0, 2, 1).contiguous()

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
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


class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.ccm(x)


class SCC(nn.Module):
    """Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        base_win_size,
        window_size,
        num_heads,
        value_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # feature projection
        self.qv = DFE(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.RMSNorm(dim)
        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = nn.Identity()
        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2 * num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(
            self.window_size[0]
            * self.window_size[1]
            // (self.base_win_size[0] * self.base_win_size[1]),
            1,
        )
        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        H, W = self.window_size
        map_H, map_W = self.base_win_size

        x = (
            x.view(B, num_h, map_H, H // map_H, map_W, W // map_W, C)
            .permute(0, 1, 2, 4, 6, 3, 5)
            .contiguous()
            .view(B, num_h, map_H * map_W, C, -1)
        )
        x = self.spatial_linear(x).view(B, num_h, map_H * map_W, C)
        return x

    def spatial_self_correlation(self, q, v):
        B, num_head, L, C = q.shape

        # spatial projection
        v = self.spatial_linear_projection(v)

        # compute correlation map
        corr_map = (q @ v.transpose(-2, -1)) / self.scale

        # add relative position bias
        # generate mother-set
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(
            torch.meshgrid([position_bias_h, position_bias_w], indexing="ij")
        )
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        # select position bias
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.base_win_size[0],
            self.window_size[0] // self.base_win_size[0],
            self.base_win_size[1],
            self.window_size[1] // self.base_win_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = (
            relative_position_bias.permute(0, 1, 3, 5, 2, 4)
            .contiguous()
            .view(
                self.window_size[0] * self.window_size[1],
                self.base_win_size[0] * self.base_win_size[1],
                self.num_heads,
                -1,
            )
            .mean(-1)
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)

        return x

    def channel_self_correlation(self, q, v):
        B, num_head, L, C = q.shape
        # apply single head strategy
        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)

        # compute correlation map
        corr_map = (q.transpose(-2, -1) @ v) / L

        # transformation
        v_drop = self.value_drop(v)
        x = (
            (corr_map @ v_drop.transpose(-2, -1))
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, -1)
        )

        return x

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """

        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x
        xB, xH, xW, xC = x.shape
        qv = self.qv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)

        # window partition
        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0] * self.window_size[1], xC)

        # qv splitting
        B, L, C = qv.shape
        qv = (
            qv.view(B, L, 2, self.num_heads, C // (2 * self.num_heads))
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, v = qv[0], qv[1]  # B, num_heads, L, C//num_heads

        # spatial self-correlation (S-SC)
        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_spatial = window_reverse(
            x_spatial, (self.window_size[0], self.window_size[1]), xH, xW
        )  # xB xH xW xC

        # channel self-correlation (C-SC)
        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_channel = window_reverse(
            x_channel, (self.window_size[0], self.window_size[1]), xH, xW
        )  # xB xH xW xC

        # spatial-channel information fusion
        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))
        x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


##############################################################
# LHSB - split dim and define 4 attn blocks
class LHSB(nn.Module):
    def __init__(self, dim, attn_drop=0.0, proj_drop=0.0, n_levels=4, window_size=8):
        super().__init__()
        self.n_levels = n_levels
        window_log = np.log2(window_size)

        self.scc = nn.ModuleList(
            [
                SCC(
                    dim // n_levels,
                    window_size=to_2tuple(int(2 ** (window_log + n_levels - i - 1))),
                    base_win_size=to_2tuple(int(2**window_log)),
                    num_heads=4,
                    proj_drop=proj_drop,
                )
                for i in range(self.n_levels)
            ]
        )

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.Mish()

    def forward(self, x):
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
            s = self.scc[i](downsampled_feat[i])
            s_upsample = F.interpolate(
                s, size=(s.shape[2] * 2, s.shape[3] * 2), mode="nearest"
            )

            if i > 0:
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample

            s_original_shape = F.interpolate(s, size=(h, w), mode="nearest")
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x


##############################################################
# Block
class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.lhsb = LHSB(dim, attn_drop=attn_drop, proj_drop=drop)

        # Feedforward layer
        self.ccm = CCM(dim)

    def forward(self, x):
        x = self.lhsb(self.norm1(x)) + x
        return self.ccm(self.norm2(x)) + x


@ARCH_REGISTRY.register()
class HiT_LMLT(nn.Module):
    def __init__(
        self,
        dim=64,
        n_blocks=8,
        ffn_scale=2.0,
        scale=4,
        window_size=8,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.window_size = window_size

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

    def check_img_size(self, x):
        _, _, h, w = x.size()
        downsample_scale = 8
        scaled_size = self.window_size * downsample_scale

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        _B, _C, H, W = x.shape

        # check image size
        x = self.check_img_size(x)

        # patch embed
        x = self.to_feat(x)

        # module, and return to original shape
        x = self.feats(x) + x
        x = x[:, :, :H, :W]

        # reconstruction
        return self.to_img(x)
