import torch
from torch import Tensor, nn
from torchvision import transforms

from traiNNer.losses.felix import FelixExtractor, FelixLoss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DinoV2Loss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        weight_ini = 1.0
        weight_end = 1.0
        curve = 1
        verbose = False

        transf = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        assert isinstance(model, nn.Module)
        if torch.cuda.is_available():
            model = model.cuda()

        hook_instance = nn.Linear
        extractor = FelixExtractor(
            model, hook_instance, transf, weight_ini, weight_end, curve, verbose
        )

        self.loss = FelixLoss(extractor)

    def forward(self, x: Tensor, gt: Tensor) -> tuple[Tensor | None, Tensor | None]:
        return self.loss(x, gt) * self.loss_weight
