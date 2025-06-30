import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class BCEWithLogitsDiceLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        weight_bce: float = 1.0,
        weight_dice: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.bce_weight = weight_bce
        self.dice_weight = weight_dice
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.shape[1] != logits.shape[1]:
            target = target.mean(dim=1, keepdim=True)
        loss_bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum(dim=[1, 2, 3])
        den = probs.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) + self.eps
        loss_dice = 1 - (num / den).mean()

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice
