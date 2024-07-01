from collections.abc import Sequence

from spandrel.architectures.DRCT import DRCT

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def drct(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    window_size: int = 16,
    img_range: float = 1.0,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    mlp_ratio: float = 2.0,
    upsampler: str = "pixelshuffle",
    resi_connection: str = "1conv",
    **kwargs,
) -> DRCT:
    return DRCT(
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
def drct_l(scale: int = 4, **kwargs) -> DRCT:
    return DRCT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def drct_xl(scale: int = 4, **kwargs) -> DRCT:
    return DRCT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )
