from collections.abc import Sequence
from typing import Literal

from spandrel_extra_arches.architectures.SRFormer import SRFormer

from traiNNer.utils.registry import SPANDREL_REGISTRY

upsampler_type = Literal["pixelshuffle", "pixelshuffledirect", "nearest+conv", ""]


@SPANDREL_REGISTRY.register()
def srformer(
    scale: int = 4,
    img_size: int = 48,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 24,
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
    upsampler: upsampler_type = "pixelshuffle",
    resi_connection: str = "1conv",
) -> SRFormer:
    return SRFormer(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        ape=ape,
        use_checkpoint=use_checkpoint,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )


@SPANDREL_REGISTRY.register()
def srformer_light(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6),
    window_size: int = 16,
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
    upsampler: upsampler_type = "pixelshuffledirect",
    resi_connection: str = "1conv",
) -> SRFormer:
    return SRFormer(
        upscale=scale,
        in_chans=in_chans,
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        ape=ape,
        use_checkpoint=use_checkpoint,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )
