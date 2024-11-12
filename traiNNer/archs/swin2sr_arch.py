from spandrel.architectures.Swin2SR import Swin2SR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def swin2sr(scale: int = 4, **kwargs) -> Swin2SR:
    return Swin2SR(upscale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def swin2sr_l(scale: int = 4, **kwargs) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        img_size=64,
        embed_dim=240,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        window_size=8,
        mlp_ratio=2,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="3conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def swin2sr_m(scale: int = 4, **kwargs) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def swin2sr_s(scale: int = 4, **kwargs) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        **kwargs,
    )
