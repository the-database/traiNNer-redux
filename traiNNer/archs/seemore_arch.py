from spandrel.architectures.SeemoRe import LRSpace, SeemoRe

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def seemore_t(
    scale: int = 4,
    in_chans: int = 3,
    num_experts: int = 3,
    num_layers: int = 6,
    embedding_dim: int = 36,
    img_range: float = 1.0,
    use_shuffle: bool = True,
    global_kernel_size: int = 11,
    recursive: int = 2,
    lr_space: LRSpace = "exp",
    topk: int = 1,
) -> SeemoRe:
    return SeemoRe(
        scale=scale,
        in_chans=in_chans,
        num_experts=num_experts,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        img_range=img_range,
        use_shuffle=use_shuffle,
        global_kernel_size=global_kernel_size,
        recursive=recursive,
        lr_space=lr_space,
        topk=topk,
    )
