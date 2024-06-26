from spandrel.architectures.PLKSR import PLKSR, RealPLKSR
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def plksr(scale: int = 4, **kwargs) -> PLKSR:
    return PLKSR(upscaling_factor=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def realplksr(scale: int = 4, **kwargs) -> RealPLKSR:
    return RealPLKSR(upscaling_factor=scale, **kwargs)
