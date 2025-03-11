import torch
from torch import Tensor

from traiNNer.losses.perceptual_fp16_loss import PerceptualFP16Loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLoss(PerceptualFP16Loss):
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        return super().forward(x, gt)
