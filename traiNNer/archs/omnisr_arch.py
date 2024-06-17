from spandrel.architectures.OmniSR import OmniSR
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def omnisr(
    res_num=5, block_num=1, bias=True, window_size=8, pe=True, ffn_bias=True, **kwargs
):
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
