from spandrel.architectures.RealCUGAN import UpCunet2x, UpCunet2x_fast, UpCunet3x, UpCunet4x
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def realcugan(pro=False, fast=False, **kwargs):
    scale = Config.get_scale()

    if fast and scale != 2:
        raise ValueError(f"Fast is only supported on scale 2, not: {scale}")

    if scale == 4:
        return UpCunet4x(pro=pro, **kwargs)
    elif scale == 3:
        return UpCunet3x(pro=pro, **kwargs)
    elif scale == 2:
        if fast:
            return UpCunet2x_fast(pro=pro, **kwargs)
        return UpCunet2x(pro=pro)

    raise ValueError(f"Scale must be 2, 3, or 4, not: {scale}")
