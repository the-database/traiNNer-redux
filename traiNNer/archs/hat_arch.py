from spandrel.architectures.HAT import HAT
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def hat(scale: int = 4, **kwargs) -> HAT:
    return HAT(upscale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def hat_l(scale: int = 4, **kwargs) -> HAT:
    return HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def hat_m(scale: int = 4, **kwargs) -> HAT:
    return HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def hat_s(scale: int = 4, **kwargs) -> HAT:
    return HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=144,
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    )
