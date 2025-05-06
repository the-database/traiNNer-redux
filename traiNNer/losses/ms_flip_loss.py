from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from traiNNer.losses.flip_loss import FLIPLoss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MSFLIPLoss(nn.Module):
    def __init__(
        self,
        weights: Sequence[float] | None = None,
        ppds: Sequence[float] | None = None,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if ppds is None:
            ppds = [(0.7 * 2**i * 3840 / 0.7) * np.pi / 180 for i in range(-2, 3)]
        if weights is None:
            weights = [1.0] * len(ppds)

        assert len(weights) == len(ppds), "Length of ppds and weights must be the same."

        self.flips = [
            FLIPLoss(weight, ppd) for weight, ppd in zip(weights, ppds, strict=False)
        ]
        self.loss_weight = loss_weight

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, test: Tensor, reference: Tensor) -> dict[str, Tensor]:
        return {str(i): flip(test, reference) for i, flip in enumerate(self.flips)}
