# https://github.com/umzi2/MoSR/blob/master/neosr/archs/mosr_arch.py
from typing import Literal

from spandrel.architectures.MoSR import MoSR

from traiNNer.utils.registry import ARCH_REGISTRY

upsampler_type = Literal["pixelshuffle", "dysample", "geoensemblepixelshuffle"]

upsampler_map = {
    "pixelshuffle": "ps",
    "dysample": "dys",
    "geoensemblepixelshuffle": "gps",
}


def get_upsampler(s: str) -> str:
    if s in upsampler_map:
        return upsampler_map[s]
    return s


@ARCH_REGISTRY.register()
def mosr(
    scale: int = 4,
    in_ch: int = 3,
    out_ch: int = 3,
    n_block: int = 24,
    dim: int = 64,
    upsampler: upsampler_type = "pixelshuffle",  # "pixelshuffle" "dysample" "geoensemblepixelshuffle"
    drop_path: float = 0.0,
    kernel_size: int = 7,
    expansion_ratio: float = 1.5,
    conv_ratio: float = 1.0,
) -> MoSR:
    return MoSR(
        upscale=scale,
        in_ch=in_ch,
        out_ch=out_ch,
        n_block=n_block,
        dim=dim,
        upsampler=get_upsampler(upsampler),
        drop_path=drop_path,
        kernel_size=kernel_size,
        expansion_ratio=expansion_ratio,
        conv_ratio=conv_ratio,
    )


@ARCH_REGISTRY.register()
def mosr_t(
    scale: int = 4,
    in_ch: int = 3,
    out_ch: int = 3,
    n_block: int = 5,
    dim: int = 48,
    upsampler: upsampler_type = "pixelshuffle",  # "pixelshuffle" "dysample" "geoensemblepixelshuffle"
    drop_path: float = 0.0,
    kernel_size: int = 7,
    expansion_ratio: float = 1.5,
    conv_ratio: float = 1.0,
) -> MoSR:
    return MoSR(
        upscale=scale,
        in_ch=in_ch,
        out_ch=out_ch,
        n_block=n_block,
        dim=dim,
        upsampler=get_upsampler(upsampler),
        drop_path=drop_path,
        kernel_size=kernel_size,
        expansion_ratio=expansion_ratio,
        conv_ratio=conv_ratio,
    )
