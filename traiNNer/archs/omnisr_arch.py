from spandrel.architectures.OmniSR import OmniSR
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def omnisr(
    res_num: int = 5,
    block_num: int = 1,
    bias: bool = True,
    window_size: int = 8,
    pe: bool = True,
    ffn_bias: bool = True,
    **kwargs,
) -> OmniSR:
    return OmniSR(
        up_scale=Config.get_scale(),
        res_num=res_num,
        block_num=block_num,
        bias=bias,
        window_size=window_size,
        pe=pe,
        ffn_bias=ffn_bias,
        **kwargs,
    )
