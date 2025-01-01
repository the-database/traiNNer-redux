from collections.abc import Sequence
from typing import Literal

from spandrel.architectures.SwinIR import SwinIR

from traiNNer.utils.registry import SPANDREL_REGISTRY


def swinir(scale: int = 4, **kwargs) -> SwinIR:
    return SwinIR(upscale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def swinir_l(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 240,
    depths: Sequence[int] = [6, 6, 6, 6, 6, 6, 6, 6, 6],
    num_heads: Sequence[int] = [8, 8, 8, 8, 8, 8, 8, 8, 8],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal[
        "pixelshuffle", "pixelshuffledirect", "nearest+conv", ""
    ] = "nearest+conv",
    resi_connection: str = "3conv",
    start_unshuffle: int = 1,
) -> SwinIR:
    return SwinIR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        start_unshuffle=start_unshuffle,
    )


@SPANDREL_REGISTRY.register()
def swinir_m(
    scale: int = 4,
    img_size: int = 48,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = [6, 6, 6, 6, 6, 6],
    num_heads: Sequence[int] = [6, 6, 6, 6, 6, 6],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal[
        "pixelshuffle", "pixelshuffledirect", "nearest+conv", ""
    ] = "pixelshuffle",
    resi_connection: str = "1conv",
    start_unshuffle: int = 1,
) -> SwinIR:
    return SwinIR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        start_unshuffle=start_unshuffle,
    )


@SPANDREL_REGISTRY.register()
def swinir_s(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 60,
    depths: Sequence[int] = [6, 6, 6, 6],
    num_heads: Sequence[int] = [6, 6, 6, 6],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal[
        "pixelshuffle", "pixelshuffledirect", "nearest+conv", ""
    ] = "pixelshuffledirect",
    resi_connection: str = "1conv",
    start_unshuffle: int = 1,
) -> SwinIR:
    return SwinIR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        start_unshuffle=start_unshuffle,
    )
