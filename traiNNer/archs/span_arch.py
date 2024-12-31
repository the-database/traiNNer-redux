from spandrel.architectures.SPAN import SPAN

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def span(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 48,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPAN:
    return SPAN(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )
