from collections.abc import Sequence

from spandrel.architectures.CRAFT import CRAFT
from torch import nn

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def craft(
    scale: int = 4,
    window_size: int = 16,
    embed_dim: int = 48,
    depths: Sequence[int] = [2, 2, 2, 2],
    num_heads: Sequence[int] = [6, 6, 6, 6],
    split_size_0: int = 4,
    split_size_1: int = 16,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    norm_layer: type[nn.Module] = nn.LayerNorm,
    img_range: float = 1.0,
    resi_connection: str = "1conv",
) -> CRAFT:
    return CRAFT(
        upscale=scale,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        split_size_0=split_size_0,
        split_size_1=split_size_1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        norm_layer=norm_layer,  # type: ignore
        img_range=img_range,
        resi_connection=resi_connection,
    )
