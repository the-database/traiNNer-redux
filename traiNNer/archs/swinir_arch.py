from spandrel.architectures.SwinIR import SwinIR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def swinir(scale: int = 4, **kwargs) -> SwinIR:
    return SwinIR(upscale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def swinir_l(scale: int = 4, **kwargs) -> SwinIR:
    return SwinIR(
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
def swinir_m(scale: int = 4, **kwargs) -> SwinIR:
    return SwinIR(
        upscale=scale,
        img_size=48,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def swinir_s(scale: int = 4, **kwargs) -> SwinIR:
    return SwinIR(
        upscale=scale,
        img_size=64,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        img_range=1.0,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        **kwargs,
    )
