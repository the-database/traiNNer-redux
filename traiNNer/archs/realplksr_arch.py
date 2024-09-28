from typing import Literal

from spandrel.architectures.PLKSR import RealPLKSR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def realplksr(
    scale: int = 4,
    upsampler: Literal["dysample", "pixelshuffle"] = "pixelshuffle",
    **kwargs,
) -> RealPLKSR:
    return RealPLKSR(upscaling_factor=scale, dysample=upsampler == "dysample", **kwargs)
