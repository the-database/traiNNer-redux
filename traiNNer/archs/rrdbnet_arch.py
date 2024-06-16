from spandrel.architectures.ESRGAN import RRDBNet
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def esrgan(**kwargs):
    return RRDBNet(scale=Config.get_scale(), **kwargs)


@SPANDREL_REGISTRY.register()
def esrgan_lite(**kwargs):
    return RRDBNet(scale=Config.get_scale(), num_filters=32, num_blocks=12, **kwargs)
