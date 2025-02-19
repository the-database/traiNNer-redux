import numpy as np
import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

from .robust_loss_pytorch import AdaptiveImageLossFunction


@LOSS_REGISTRY.register()
class AdaptiveLoss(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        super().__init__()

        self.loss_weight = loss_weight

        self.adaptive = AdaptiveImageLossFunction(
            image_size=[3, 256, 256],  # TODO
            color_space="RGB",
            representation="PIXEL",
            float_dtype=np.float32,
            device=0,
        )

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return self.loss_weight * self.adaptive.lossfun(pred - gt).mean()
