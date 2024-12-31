from collections.abc import Sequence
from typing import Literal

from spandrel.architectures.PLKSR import PLKSR

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def plksr(
    scale: int = 4,
    dim: int = 64,
    n_blocks: int = 28,
    # CCM options
    ccm_type: Literal["CCM", "ICCM", "DCCM"] = "DCCM",
    # LK Options
    kernel_size: int = 17,
    split_ratio: float = 0.25,
    lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK",
    # LK Rep options
    use_max_kernel: bool = False,
    sparse_kernels: Sequence[int] = [5, 5, 5, 5],
    sparse_dilations: Sequence[int] = [1, 2, 3, 4],
    with_idt: bool = False,
    # EA ablation
    use_ea: bool = True,
) -> PLKSR:
    return PLKSR(
        upscaling_factor=scale,
        dim=dim,
        n_blocks=n_blocks,
        ccm_type=ccm_type,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        lk_type=lk_type,
        use_max_kernel=use_max_kernel,
        sparse_kernels=sparse_kernels,
        sparse_dilations=sparse_dilations,
        with_idt=with_idt,
        use_ea=use_ea,
    )


@SPANDREL_REGISTRY.register()
def plksr_tiny(
    scale: int = 4,
    dim: int = 64,
    n_blocks: int = 12,
    # CCM options
    ccm_type: Literal["CCM", "ICCM", "DCCM"] = "DCCM",
    # LK Options
    kernel_size: int = 13,
    split_ratio: float = 0.25,
    lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK",
    # LK Rep options
    use_max_kernel: bool = False,
    sparse_kernels: Sequence[int] = [5, 5, 5, 5],
    sparse_dilations: Sequence[int] = [1, 2, 3, 4],
    with_idt: bool = False,
    # EA ablation
    use_ea: bool = False,
) -> PLKSR:
    return PLKSR(
        upscaling_factor=scale,
        dim=dim,
        n_blocks=n_blocks,
        ccm_type=ccm_type,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        lk_type=lk_type,
        use_max_kernel=use_max_kernel,
        sparse_kernels=sparse_kernels,
        sparse_dilations=sparse_dilations,
        with_idt=with_idt,
        use_ea=use_ea,
    )
