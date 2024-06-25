import numpy as np
import torch
from torch import Tensor
from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.utils.img_util import imgs2tensors
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_dists(img: np.ndarray, img2: np.ndarray, **kwargs) -> Tensor:
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    # to tensor
    img_t, img2_t = imgs2tensors([img, img2], color=True, bgr2rgb=True, float32=True)
    # normalize to [0, 1]
    img_t, img2_t = img_t / 255, img2_t / 255
    # add dim
    img_t, img2_t = img_t.unsqueeze_(0), img2_t.unsqueeze_(0)
    # to cuda
    device = torch.device("cuda")
    img_t, img2_t = img_t.to(device), img2_t.to(device)

    loss = DISTSLoss(as_loss=False).to(device)
    with torch.no_grad():
        return loss.forward(img_t, img2_t)
