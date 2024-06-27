from collections.abc import Sequence

from spandrel_extra_arches.architectures.SRFormer import SRFormer
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def srformer(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 48,
    window_size: int = 24,
    img_range: float = 1.0,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    mlp_ratio: int = 2,
    upsampler: str = "pixelshuffle",
    resi_connection: str = "1conv",
    **kwargs,
) -> SRFormer:
    return SRFormer(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection,
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def srformer_light(scale: int = 4, **kwargs) -> SRFormer:
    return SRFormer(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        **kwargs,
    )
