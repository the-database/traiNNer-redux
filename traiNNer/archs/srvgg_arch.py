from spandrel.architectures.Compact import SRVGGNetCompact
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def compact(**kwargs):
    return SRVGGNetCompact(upscale=Config.get_scale(), **kwargs)


@SPANDREL_REGISTRY.register()
def ultracompact(**kwargs):
    return SRVGGNetCompact(upscale=Config.get_scale(), num_feat=64, num_conv=8, **kwargs)


@SPANDREL_REGISTRY.register()
def superultracompact(**kwargs):
    return SRVGGNetCompact(upscale=Config.get_scale(), num_feat=24, num_conv=8, **kwargs)
