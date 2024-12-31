from spandrel.architectures.RealCUGAN import (
    UpCunet2x,
    UpCunet2x_fast,
    UpCunet3x,
    UpCunet4x,
)

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def realcugan(
    scale: int = 4,
    pro: bool = False,
    fast: bool = False,
    in_channels: int = 3,
    out_channels: int = 3,
) -> UpCunet2x | UpCunet2x_fast | UpCunet3x | UpCunet4x:
    if fast and scale != 2:
        raise ValueError(f"Fast is only supported on scale 2, not: {scale}")

    if pro and fast:
        raise ValueError(
            "Pro is not supported with fast enabled, disable pro or disable fast."
        )

    if scale == 4:
        return UpCunet4x(pro=pro, in_channels=in_channels, out_channels=out_channels)
    elif scale == 3:
        return UpCunet3x(pro=pro, in_channels=in_channels, out_channels=out_channels)
    elif scale == 2:
        if fast:
            return UpCunet2x_fast(in_channels=in_channels, out_channels=out_channels)
        return UpCunet2x(pro=pro)

    raise ValueError(f"Scale must be 2, 3, or 4, not: {scale}")
