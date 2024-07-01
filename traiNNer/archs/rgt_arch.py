from collections.abc import Sequence

from spandrel.architectures.RGT import RGT

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def rgt(
    scale: int = 4,
    in_chans: int = 3,
    img_size: int = 64,
    img_range: float = 1.0,
    split_size: Sequence[int] = (8, 32),
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6),
    embed_dim: int = 180,
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6),
    mlp_ratio: float = 2.0,
    resi_connection: str = "1conv",
    c_ratio: float = 0.5,
    **kwargs,
) -> RGT:
    return RGT(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        split_size=split_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        resi_connection=resi_connection,
        c_ratio=c_ratio,
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def rgt_s(scale: int = 4, **kwargs) -> RGT:
    return RGT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        split_size=(8, 32),
        depth=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        resi_connection="1conv",
        c_ratio=0.5,
        **kwargs,
    )
