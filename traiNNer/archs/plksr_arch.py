from typing import Literal

from spandrel.architectures.PLKSR import PLKSR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def plksr(
    scale: int = 4, ccm_type: Literal["CCM", "ICCM", "DCCM"] = "DCCM", **kwargs
) -> PLKSR:
    return PLKSR(upscaling_factor=scale, ccm_type=ccm_type, **kwargs)


@SPANDREL_REGISTRY.register()
def plksr_tiny(scale: int = 4) -> PLKSR:
    return PLKSR(
        upscaling_factor=scale,
        ccm_type="DCCM",
        dim=64,
        n_blocks=12,
        kernel_size=13,
        split_ratio=0.25,
        lk_type="PLK",
        use_ea=False,
    )
