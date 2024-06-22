from spandrel.architectures.SPAN import SPAN
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def span(num_in_ch: int = 3, num_out_ch: int = 3, **kwargs) -> SPAN:
    return SPAN(
        upscale=Config.get_scale(), num_in_ch=num_in_ch, num_out_ch=num_out_ch, **kwargs
    )
