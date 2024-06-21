from spandrel.architectures.PLKSR import PLKSR, RealPLKSR
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def plksr(**kwargs) -> PLKSR:
    return PLKSR(upscaling_factor=Config.get_scale(), **kwargs)


@SPANDREL_REGISTRY.register()
def realplksr(**kwargs) -> RealPLKSR:
    return RealPLKSR(upscaling_factor=Config.get_scale(), **kwargs)
