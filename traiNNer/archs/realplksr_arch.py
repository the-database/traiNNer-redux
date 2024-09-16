from spandrel.architectures.PLKSR import RealPLKSR

from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
def realplksr(scale: int = 4, **kwargs) -> RealPLKSR:
    return RealPLKSR(upscaling_factor=scale, **kwargs)
