import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.mssim_loss import filter2, fspecial, preprocess_rgb
from traiNNer.losses.perceptual_fp16_loss import VGG
from traiNNer.utils.registry import LOSS_REGISTRY


def weighted_ms_ssim(
    x: Tensor,
    y: Tensor,
    weight_map: Tensor,
    win: Tensor | None = None,
    data_range: float = 1.0,
    include_luminance: bool = False,
) -> Tensor:
    """MS-SSIM with per-scale spatial weighting."""
    if win is None:
        win = fspecial(11, 1.5, x.shape[1]).to(x)

    scale_weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device
    )
    levels = len(scale_weights)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mcs_list: list[Tensor] = []
    w = weight_map

    for i in range(levels):
        filter_shape = "valid" if x.shape[-1] >= win.shape[-1] else "same"

        mu1 = filter2(x, win, filter_shape)
        mu2 = filter2(y, win, filter_shape)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(x * x, win, filter_shape) - mu1_sq
        sigma2_sq = filter2(y * y, win, filter_shape) - mu2_sq
        sigma12 = filter2(x * y, win, filter_shape) - mu1_mu2

        cs_map = F.relu((2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2))

        # downsample weight map to match feature map size
        w_down = F.interpolate(
            w, size=cs_map.shape[-2:], mode="bilinear", align_corners=False
        )

        if i == levels - 1:
            # last level: include luminance only if requested
            if include_luminance:
                l_map = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
                final_map = l_map * cs_map
            else:
                final_map = cs_map
            val = (w_down * final_map).sum(dim=[1, 2, 3]) / (
                w_down.sum(dim=[1, 2, 3]) + 1e-8
            )
        else:
            # intermediate levels: just CS
            val = (w_down * cs_map).sum(dim=[1, 2, 3]) / (
                w_down.sum(dim=[1, 2, 3]) + 1e-8
            )

        mcs_list.append(val)

        # downsample images and weight for next scale
        pad = (x.shape[2] % 2, x.shape[3] % 2)
        x = F.avg_pool2d(x, kernel_size=2, padding=pad)
        y = F.avg_pool2d(y, kernel_size=2, padding=pad)
        w = F.avg_pool2d(w, kernel_size=2, padding=pad)

    mcs = torch.stack(mcs_list, dim=0).clamp(min=1e-8)
    return torch.prod(mcs ** scale_weights.view(-1, 1), dim=0)


@LOSS_REGISTRY.register()
class StructureTextureLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        layer_weights: dict[str, float] | None = None,
        floor: float = 0.1,
        test_y_channel: bool = True,
        color_space: str = "yiq",
        include_luminance: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.floor = floor
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.include_luminance = include_luminance
        self.data_range = 1.0

        if layer_weights is None:
            layer_weights = {
                "conv1_2": 0.1,
                "conv2_2": 0.1,
                "conv3_4": 1,
                "conv4_4": 1,
                "conv5_4": 1,
            }
        self.layer_weights = layer_weights
        self.vgg = VGG(list(layer_weights.keys())).to(memory_format=torch.channels_last)

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

        # compute structure weight on RGB before any conversion
        w_s = self.compute_structure_weight(gt)
        w_t = 1 - w_s

        # preprocess to Y channel for MS-SSIM (matches MSSIMLoss behavior)
        pred_y = preprocess_rgb(
            pred, self.test_y_channel, self.data_range, self.color_space
        )
        gt_y = preprocess_rgb(
            gt, self.test_y_channel, self.data_range, self.color_space
        )

        # structure loss: weighted MS-SSIM on Y channel
        msssim_val = weighted_ms_ssim(
            pred_y,
            gt_y,
            w_s,
            data_range=self.data_range,
            include_luminance=self.include_luminance,
        )
        structure_loss = 1 - msssim_val.mean()

        # texture loss: weighted VGG perceptual on RGB
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
            "structure": self.pixel_weight * structure_loss,
            "texture": self.perceptual_weight * percep_loss,
        }
