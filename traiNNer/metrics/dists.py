import numpy as np
import torch
from torch import Tensor

from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_dists(
    img: np.ndarray, img2: np.ndarray, device: torch.device, **kwargs
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    loss = DISTSLoss(loss_weight=1.0, as_loss=False).to(device)
    with torch.inference_mode():
        return loss(
            img2batchedtensor(img, device, bgr2rgb=False),
            img2batchedtensor(img2, device, bgr2rgb=False),
        )
