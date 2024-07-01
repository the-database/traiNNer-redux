from spandrel.architectures.OmniSR import OmniSR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def omnisr(
    scale: int = 4,
    res_num: int = 5,
    block_num: int = 1,
    bias: bool = True,
    window_size: int = 8,
    pe: bool = True,
    **kwargs,
) -> OmniSR:
    return OmniSR(
        up_scale=scale,
        res_num=res_num,
        block_num=block_num,
        bias=bias,
        window_size=window_size,
        pe=pe,
        **kwargs,
    )
