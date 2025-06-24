import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CosimLoss(nn.Module):
    def __init__(
        self, loss_weight: float, cosim_lambda: float = 5, warmup_iter: int = -1
    ) -> None:
        super().__init__()

        self.cosim_lambda = cosim_lambda
        self.loss_weight = loss_weight
        self.warmup_iter = warmup_iter
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.cosim_penalty(x, y)

    def cosim_penalty(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.clamp(x, 1e-12, 1)
        y = torch.clamp(y, 1e-12, 1)

        distance = 1 - torch.round(self.similarity(x, y), decimals=20).mean()
        return self.cosim_lambda * distance
