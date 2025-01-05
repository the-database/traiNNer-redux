from spandrel.architectures.DCTLSA import DCTLSA

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def dctlsa(
    scale: int = 4,
    in_nc: int = 3,
    nf: int = 55,
    num_modules: int = 6,
    out_nc: int = 3,
    num_head: int = 5,
) -> DCTLSA:
    return DCTLSA(
        upscale=scale,
        in_nc=in_nc,
        nf=nf,
        num_modules=num_modules,
        out_nc=out_nc,
        num_head=num_head,
    )
