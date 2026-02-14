import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SobelEdgeLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 4.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 4.0,
        )

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred_luma = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        gt_luma = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]

        pred_gx = F.conv2d(pred_luma, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_luma, self.sobel_y, padding=1)
        gt_gx = F.conv2d(gt_luma, self.sobel_x, padding=1)
        gt_gy = F.conv2d(gt_luma, self.sobel_y, padding=1)

        pred_grad = torch.cat([pred_gx, pred_gy], dim=1)
        gt_grad = torch.cat([gt_gx, gt_gy], dim=1)

        loss = F.mse_loss(pred_grad, gt_grad)

        return loss
