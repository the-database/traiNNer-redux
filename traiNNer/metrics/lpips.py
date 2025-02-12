from typing import Literal

import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(
    img: np.ndarray,
    img2: np.ndarray,
    device: torch.device,
    net: Literal["alex", "vgg", "squeeze"] = "alex",
    **kwargs,
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    loss = LPIPS(net=net).to(device)

    with torch.inference_mode():
        return loss(
            img2batchedtensor(img, device, bgr2rgb=False),
            img2batchedtensor(img2, device, bgr2rgb=False),
            normalize=True,
        ).view(())
