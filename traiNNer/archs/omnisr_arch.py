from spandrel.architectures.OmniSR import OmniSR
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def omnisr(**kwargs):
    return OmniSR(up_scale=Config.get_scale(), **kwargs)
