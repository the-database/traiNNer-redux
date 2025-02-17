import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class NCCLoss(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        cc_value = _cc_single_torch(sr, hr)
        return self.loss_weight * (1 - ((cc_value + 1) * 0.5))


def _cc_single_torch(raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
    """
    Compute the Cross-Correlation (CC) metric between two input tensors representing images.

    CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

    Args:
        raw_tensor (Tensor): The image tensor to be compared.
        dst_tensor (Tensor): The reference image tensor.

    Returns:
        CC (Tensor): The Cross-Correlation (CC) metric score.

    """
    n_spectral = raw_tensor.shape[1]

    # Reshaping fused and reference data
    raw_tensor_reshaped = raw_tensor.reshape(n_spectral, -1)
    dst_tensor_reshaped = dst_tensor.reshape(n_spectral, -1)

    # Calculating mean value
    mean_raw = torch.mean(raw_tensor_reshaped, 1).unsqueeze(1)
    mean_dst = torch.mean(dst_tensor_reshaped, 1).unsqueeze(1)

    cc = torch.sum(
        (raw_tensor_reshaped - mean_raw) * (dst_tensor_reshaped - mean_dst), 1
    ) / torch.sqrt(
        torch.sum((raw_tensor_reshaped - mean_raw) ** 2, 1)
        * torch.sum((dst_tensor_reshaped - mean_dst) ** 2, 1)
    )

    cc = torch.mean(cc)

    return cc
