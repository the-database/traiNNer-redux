import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


def get_local_weights(residual: Tensor, ksize: int) -> Tensor:
    """Get local weights for generating the artifact map of LDL.

    It is only called by the `get_refined_artifact_map` function.

    Args:
        residual (Tensor): Residual between predicted and ground truth images.
        ksize (Int): size of the local window.

    Returns:
        Tensor: weight for each pixel to be discriminated as an artifact pixel
    """

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = (
        torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
        .squeeze(-1)
        .squeeze(-1)
    )

    return pixel_level_weight


def get_refined_artifact_map(
    img_gt: Tensor, img_output: Tensor, img_ema: Tensor, ksize: int = 7
) -> Tensor:
    """Calculate the artifact map of LDL
    (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022)

    Args:
        img_gt (Tensor): ground truth images.
        img_output (Tensor): output images given by the optimizing model.
        img_ema (Tensor): output images given by the ema model.
        ksize (Int): size of the local window.

    Returns:
        overall_weight: weight for each pixel to be discriminated as an artifact pixel
        (calculated based on both local and global observations).
    """
    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(
        residual_sr.clone(), dim=(-1, -2, -3), keepdim=True
    ) ** (1 / 5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_sr < residual_ema] = 0

    return overall_weight


@LOSS_REGISTRY.register()
class LDLLoss(nn.Module):
    def __init__(
        self, loss_weight: float, criterion: str = "l1", warmup_iter: int = -1
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.warmup_iter = warmup_iter

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, output: Tensor, output_ema: Tensor, gt: Tensor) -> Tensor:
        pixel_weight = get_refined_artifact_map(gt, output, output_ema)
        return self.criterion(
            torch.mul(pixel_weight, output), torch.mul(pixel_weight, gt)
        )
