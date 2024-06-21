from spandrel.architectures.SPAN import SPAN
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def span(**kwargs) -> SPAN:
    return SPAN(upscale=Config.get_scale(), **kwargs)
