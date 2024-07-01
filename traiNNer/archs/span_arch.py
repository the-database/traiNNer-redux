from spandrel.architectures.SPAN import SPAN

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def span(scale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3, **kwargs) -> SPAN:
    return SPAN(upscale=scale, num_in_ch=num_in_ch, num_out_ch=num_out_ch, **kwargs)
