from spandrel.architectures.ESRGAN import RRDBNet
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def esrgan(**kwargs):
    opts, _ = Config.get_config()
    return RRDBNet(scale=opts['scale'], **kwargs)


@SPANDREL_REGISTRY.register()
def esrganlite(**kwargs):
    opts, _ = Config.get_config()
    return RRDBNet(scale=opts['scale'], num_filters=32, num_blocks=12, **kwargs)
