import math
from collections.abc import Sequence

import torch
from torch import SymInt, nn
from torch.nn import functional as F  # noqa: N812
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RGBCosineSimilarityLoss(nn.Module):
    def __init__(
            self,
            loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.loss_weight = loss_weight
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        loss = 1 - self.cosim(x, y)

        return self.loss_weight * loss

    def cosim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO test without rounding
        return torch.round(self.similarity(x.clamp(1e-12, 1), y.clamp(1e-12, 1)), decimals=20).mean()
