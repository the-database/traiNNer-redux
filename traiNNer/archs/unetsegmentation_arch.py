import segmentation_models_pytorch as smp
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
def unetsegmentation(
    scale: int = 1,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    in_ch: int = 3,
    classes: int = 1,
    activation: str | None = None,
) -> nn.Module:
    """
    Generic U-Net for segmentation.
    - encoder_name: any SMP encoder (resnet34, efficientnet-b7, swin_tiny, etc.)
    - encoder_weights: pretrained weights or None
    - in_ch: input channels (e.g. 3 for RGB)
    - classes: output channels (1 for binary, >1 for multi-class)
    - activation: e.g. "sigmoid" or "softmax2d", or None to get raw logits
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_ch,
        classes=classes,
        activation=activation,
    )
