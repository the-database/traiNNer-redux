from collections.abc import Sequence

from spandrel.architectures.RGT import RGT

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def rgt(
    scale: int = 4,
    img_size: int = 64,
    in_chans: int = 3,
    embed_dim: int = 180,
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6),
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    img_range: float = 1.0,
    resi_connection: str = "1conv",
    split_size: Sequence[int] = [8, 32],
    c_ratio: float = 0.5,
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
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        resi_connection=resi_connection,
        c_ratio=c_ratio,
    )


@SPANDREL_REGISTRY.register()
def rgt_s(
    scale: int = 4,
    img_size: int = 64,
    in_chans: int = 3,
    embed_dim: int = 180,
    depth: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_chk: bool = False,
    img_range: float = 1.0,
    resi_connection: str = "1conv",
    split_size: Sequence[int] = [8, 32],
    c_ratio: float = 0.5,
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
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_chk=use_chk,
        resi_connection=resi_connection,
        c_ratio=c_ratio,
    )
