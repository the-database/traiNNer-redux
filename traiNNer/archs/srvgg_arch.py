from spandrel.architectures.Compact import Compact

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def compact(scale: int = 4, **kwargs) -> Compact:
    return Compact(upscale=scale, **kwargs)


@SPANDREL_REGISTRY.register()
def ultracompact(scale: int = 4, **kwargs) -> Compact:
    return Compact(upscale=scale, num_feat=64, num_conv=8, **kwargs)


@SPANDREL_REGISTRY.register()
def superultracompact(scale: int = 4, **kwargs) -> Compact:
    return Compact(upscale=scale, num_feat=24, num_conv=8, **kwargs)
