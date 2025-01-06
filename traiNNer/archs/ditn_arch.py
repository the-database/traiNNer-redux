from spandrel.architectures.DITN import DITN

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def ditn_real(
    scale: int = 4,
    inp_channels: int = 3,
    dim: int = 60,
    ITL_blocks: int = 4,  # noqa: N803
    SAL_blocks: int = 4,  # noqa: N803
    UFONE_blocks: int = 1,  # noqa: N803
    ffn_expansion_factor: int = 2,
    bias: bool = False,
    LayerNorm_type: str = "WithBias",  # noqa: N803
    patch_size: int = 8,
) -> DITN:
    return DITN(
        upscale=scale,
        inp_channels=inp_channels,
        dim=dim,
        ITL_blocks=ITL_blocks,
        SAL_blocks=SAL_blocks,
        UFONE_blocks=UFONE_blocks,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        patch_size=patch_size,
    )
