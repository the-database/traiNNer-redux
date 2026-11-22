import math
from collections.abc import Sequence
from typing import Any

import torch
from spandrel.architectures.ATD import ATD as SpandrelATD  # noqa: N811
from spandrel.architectures.ATD.__arch.atd_arch import AC_MSA, ATDTransformerLayer
from torch import Tensor, nn

from traiNNer.utils.registry import SPANDREL_REGISTRY


class _CategoryAttention(nn.Module):
    """ATD category attention with a batched scatter back into image order."""

    def __init__(self, original: AC_MSA) -> None:
        super().__init__()
        self.num_heads = original.num_heads
        self.category_size = original.category_size
        self.logit_scale = original.logit_scale
        self.proj = original.proj

    def forward(self, qkv: Tensor, sim: Tensor, x_size: tuple[int, int]) -> Tensor:
        b, n, c3 = qkv.shape
        c = c3 // 3
        size = min(n, self.category_size)
        order = sim.argmax(dim=-1).argsort(dim=-1, stable=False).unsqueeze(-1)
        grouped = qkv.gather(1, order.expand_as(qkv))
        pad = -n % size
        if not self.training or pad:
            grouped = torch.cat((grouped, grouped[:, n - pad :].flip((1,))), dim=1)
        q, k, v = (
            grouped.reshape(b, -1, size, 3, self.num_heads, c // self.num_heads)
            .permute(3, 0, 1, 4, 2, 5)
            .unbind(0)
        )
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        weights = ((q @ k.transpose(-1, -2)) * scale).softmax(dim=-1)
        y = (weights @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad, c)[:, :n]
        return self.proj(torch.empty_like(y).scatter_(1, order.expand_as(y), y))


class ATD(SpandrelATD):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._training_mask = None
        for layer in self.modules():
            if isinstance(layer, ATDTransformerLayer):
                layer.add_module("attn_aca", _CategoryAttention(layer.attn_aca))

    def calculate_mask(self, x_size: Sequence[int]) -> Tensor:
        if not self.training or torch.compiler.is_compiling():
            return super().calculate_mask(x_size)
        device = self.get_buffer("relative_position_index_SA").device
        key = (tuple(x_size), device, torch.get_default_dtype())
        if self._training_mask is not None and self._training_mask[0] == key:
            return self._training_mask[1]
        with device:
            mask = super().calculate_mask(x_size)
        if not torch.is_inference_mode_enabled():
            self._training_mask = (key, mask)
        return mask


@SPANDREL_REGISTRY.register()
def atd(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 96,
    embed_dim: int = 210,
    depths: Sequence[int] = (
        6,
        6,
        6,
        6,
        6,
        6,
    ),
    num_heads: Sequence[int] = (
        6,
        6,
        6,
        6,
        6,
        6,
    ),
    window_size: int = 16,
    category_size: int = 256,
    num_tokens: int = 128,
    reducted_dim: int = 20,
    convffn_kernel_size: int = 5,
    img_range: float = 1.0,
    mlp_ratio: int = 2,
    upsampler: str = "pixelshuffle",
    resi_connection: str = "1conv",
) -> ATD:
    return ATD(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        category_size=category_size,
        num_tokens=num_tokens,
        reducted_dim=reducted_dim,
        convffn_kernel_size=convffn_kernel_size,
        img_range=img_range,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )


@SPANDREL_REGISTRY.register()
def atd_light(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    embed_dim: int = 48,
    depths: Sequence[int] = (
        6,
        6,
        6,
        6,
    ),
    num_heads: Sequence[int] = (
        6,
        6,
        6,
        6,
    ),
    window_size: int = 16,
    category_size: int = 128,
    num_tokens: int = 64,
    reducted_dim: int = 8,
    convffn_kernel_size: int = 7,
    img_range: float = 1.0,
    mlp_ratio: int = 1,
    upsampler: str = "pixelshuffledirect",
    resi_connection: str = "1conv",
) -> ATD:
    return ATD(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        category_size=category_size,
        num_tokens=num_tokens,
        reducted_dim=reducted_dim,
        convffn_kernel_size=convffn_kernel_size,
        img_range=img_range,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )
