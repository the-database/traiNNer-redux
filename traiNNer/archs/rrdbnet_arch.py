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
    num_filters: int = 64,
    num_blocks: int = 23,
) -> ESRGAN:
    if use_pixel_unshuffle:
        if scale in pixel_unshuffle_scales:
            in_nc *= 4 ** (3 - scale)
            shuffle_factor = int(math.sqrt(in_nc / out_nc))
            return ESRGAN(
                scale=4,
                in_nc=in_nc,
                shuffle_factor=shuffle_factor,
                num_blocks=num_blocks,
                num_filters=num_filters,
            )

    return ESRGAN(
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        num_blocks=num_blocks,
        num_filters=num_filters,
    )


@SPANDREL_REGISTRY.register()
def esrgan_lite(
    scale: int = 4,
    use_pixel_unshuffle: bool = True,
    in_nc: int = 3,
    out_nc: int = 3,
    num_filters: int = 32,
    num_blocks: int = 12,
) -> ESRGAN:
    return esrgan(
        scale=scale,
        use_pixel_unshuffle=use_pixel_unshuffle,
        in_nc=in_nc,
        out_nc=out_nc,
        num_filters=num_filters,
        num_blocks=num_blocks,
    )
