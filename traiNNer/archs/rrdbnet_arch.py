import math

from spandrel.architectures.ESRGAN import ESRGAN

from traiNNer.utils.registry import SPANDREL_REGISTRY

pixel_unshuffle_scales = (1, 2)


@SPANDREL_REGISTRY.register()
def esrgan(
    scale: int = 4,
    use_pixel_unshuffle: bool = True,
    in_nc: int = 3,
    out_nc: int = 3,
    **kwargs,
) -> ESRGAN:
    if use_pixel_unshuffle:
        if scale in pixel_unshuffle_scales:
            in_nc *= 4 ** (3 - scale)
            shuffle_factor = int(math.sqrt(in_nc / out_nc))
            return ESRGAN(scale=4, in_nc=in_nc, shuffle_factor=shuffle_factor, **kwargs)

    return ESRGAN(scale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def esrgan_lite(scale: int = 4, use_pixel_unshuffle: bool = True, **kwargs) -> ESRGAN:
    return esrgan(
        scale=scale,
        use_pixel_unshuffle=use_pixel_unshuffle,
        num_filters=32,
        num_blocks=12,
        **kwargs,
    )
