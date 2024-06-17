from spandrel.architectures.ESRGAN import RRDBNet
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY
from traiNNer.utils.logger import get_root_logger

pixel_unshuffle_scales = (1, 2)


@SPANDREL_REGISTRY.register()
def esrgan(use_pixel_unshuffle=False, in_nc=3, **kwargs) -> RRDBNet:
    scale = Config.get_scale()

    if use_pixel_unshuffle:
        if scale in pixel_unshuffle_scales:
            in_nc *= 4 ** (3 - scale)
            return RRDBNet(scale=4, in_nc=in_nc, shuffle_factor=scale, **kwargs)
        else:
            logger = get_root_logger()
            logger.warning(f"Pixel unshuffle option is ignored since scale is not in {pixel_unshuffle_scales}")

    return RRDBNet(scale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def esrgan_lite(use_pixel_unshuffle=False, **kwargs) -> RRDBNet:
    return esrgan(use_pixel_unshuffle=use_pixel_unshuffle, num_filters=32, num_blocks=12, **kwargs)
