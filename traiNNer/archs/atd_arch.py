from collections.abc import Sequence

from spandrel.architectures.ATD import ATD

from traiNNer.utils.registry import SPANDREL_REGISTRY


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
