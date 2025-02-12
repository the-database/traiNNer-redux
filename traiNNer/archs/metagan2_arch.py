from collections.abc import Sequence

import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from traiNNer.utils.registry import ARCH_REGISTRY


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        num_heads: int | None = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, _ = x.shape  # noqa: N806
        N = H * W  # noqa: N806
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=7 // 2, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
        drop_path: float = 0.0,
        att: bool = False,
    ) -> None:
        super().__init__()
        if att:
            expansion_ratio = 1.5
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = Attention(conv_channels) if att else DConv(conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut


class Down(nn.Sequential):
    def __init__(self, dim: int = 48, out_dim: int = 48) -> None:
        super().__init__(nn.Conv2d(dim, out_dim, 3, 2, 1), nn.GroupNorm(4, out_dim))


class Blocks(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        blocks: int,
        scale: int,
        att: bool,
        drop: Sequence[float],
    ) -> None:
        super().__init__()
        self.down = (
            Down(in_dim, out_dim)
            if scale == 2
            else nn.Sequential(
                Down(in_dim, out_dim // 2), nn.Mish(True), Down(out_dim // 2, out_dim)
            )
        )
        self.blocks = nn.Sequential(
            *[
                GatedCNNBlock(out_dim, att=att, drop_path=drop[index])
                for index in range(blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x).permute(0, 2, 3, 1)
        x = self.blocks(x).permute(0, 3, 1, 2)
        return x


@ARCH_REGISTRY.register()
class MetaGan2(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 1,
        dims: Sequence[int] = (48, 96, 192, 288),
        blocks: Sequence[int] = (3, 3, 9, 3),
        downs: Sequence[int] = (4, 4, 2, 2),
        drop_path: float = 0.0,
        end_drop: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_ch, *list(dims)]
        dp_rates = [
            x.tolist() for x in torch.linspace(0, drop_path, sum(blocks)).split(blocks)
        ]
        self.stages = nn.Sequential(
            *[
                Blocks(
                    dims[index],
                    dims[index + 1],
                    blocks[index],
                    downs[index],
                    index > 1,
                    dp_rates[index],
                )
                for index in range(len(blocks))
            ]
            + [
                nn.Conv2d(dims[-1], 100, 1, 1, 0),
                nn.Mish(True),
                nn.Dropout(end_drop),
                nn.Conv2d(100, n_class, 1, 1, 0),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stages(x)
