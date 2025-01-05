from spandrel.architectures.SAFMN import SAFMN

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def safmn(
    scale: int = 4, dim: int = 36, n_blocks: int = 8, ffn_scale: float = 2.0
) -> SAFMN:
    return SAFMN(
        upscaling_factor=scale, dim=dim, n_blocks=n_blocks, ffn_scale=ffn_scale
    )


@SPANDREL_REGISTRY.register()
def safmn_l(
    scale: int = 4, dim: int = 128, n_blocks: int = 16, ffn_scale: float = 2.0
) -> SAFMN:
    return SAFMN(
        upscaling_factor=scale, dim=dim, n_blocks=n_blocks, ffn_scale=ffn_scale
    )
