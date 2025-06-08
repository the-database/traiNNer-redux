import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.topiq_arch import CFANet
from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_topiq(
    img: np.ndarray, img2: np.ndarray, device: torch.device, **kwargs
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    topiq = CFANet().to(device)
    topiq.eval()
    with torch.inference_mode():
        return topiq(
            img2batchedtensor(img, device, from_bgr=False),
            img2batchedtensor(img2, device, from_bgr=False),
        )
