import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.losses.perceptual_fp16_loss import VGG
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class StructureTextureLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        layer_weights: dict[str, float] | None = None,
        floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.floor = floor

        if layer_weights is None:
            layer_weights = {
                "conv1_2": 0.1,
                "conv2_2": 0.1,
                "conv3_4": 1,
                "conv4_4": 1,
                "conv5_4": 1,
            }

        self.layer_weights = layer_weights
        self.vgg = VGG(list(layer_weights.keys())).to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]

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

    def compute_structure_weight(self, gt: Tensor) -> Tensor:
        luma = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
        gx = F.conv2d(luma, self.sobel_x, padding=1)
        gy = F.conv2d(luma, self.sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        mag_norm = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return self.floor + (1 - self.floor) * mag_norm

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, pred: Tensor, gt: Tensor) -> dict[str, Tensor]:
        gt = gt.detach()
        w_s = self.compute_structure_weight(gt)
        w_t = 1 - w_s

        # pixel loss weighted by structure
        pixel_diff = charbonnier_loss(pred, gt, reduction="none")
        pixel_loss = (w_s * pixel_diff).mean()

        # perceptual loss weighted by texture
        pred_feats = self.vgg(pred)
        gt_feats = self.vgg(gt)

        percep_loss = gt.new_tensor(0.0)
        for k, weight in self.layer_weights.items():
            diff = (pred_feats[k] - gt_feats[k]).abs().mean(dim=1, keepdim=True)
            w_t_down = F.interpolate(
                w_t, size=diff.shape[-2:], mode="bilinear", align_corners=False
            )
            percep_loss = percep_loss + weight * (w_t_down * diff).mean()

        return {
            "structure": self.pixel_weight * pixel_loss,
            "texture": self.perceptual_weight * percep_loss,
        }
