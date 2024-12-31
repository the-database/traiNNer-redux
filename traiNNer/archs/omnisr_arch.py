from spandrel.architectures.OmniSR import OmniSR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def omnisr(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    block_num: int = 1,
    pe: bool = True,
    window_size: int = 8,
    res_num: int = 1,
    bias: bool = True,
) -> OmniSR:
    return OmniSR(
        up_scale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        res_num=res_num,
        block_num=block_num,
        bias=bias,
        window_size=window_size,
        pe=pe,
    )
