import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision.transforms import InterpolationMode, v2

from traiNNer.losses.loss_util import weighted_loss
from traiNNer.utils.color_util import rgb2ycbcr_pt, rgb_to_luma
from traiNNer.utils.hsluv import rgb_to_hsluv
from traiNNer.utils.registry import LOSS_REGISTRY

_reduction_modes = ["none", "mean", "sum"]
VGG_PATCH_SIZE = 256


@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def charbonnier_loss(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt((pred - target) ** 2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight: float, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self, pred: Tensor, target: Tensor, weight: Tensor | None = None, **kwargs
    ) -> Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight: float, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self, pred: Tensor, target: Tensor, weight: Tensor | None = None, **kwargs
    ) -> Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(
        self, loss_weight: float, reduction: str = "mean", eps: float = 1e-12
    ) -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(
        self, pred: Tensor, target: Tensor, weight: Tensor | None = None, **kwargs
    ) -> Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    """Color loss"""

    def __init__(
        self, loss_weight: float, criterion: str = "l1", scale: int = 4
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.scale = scale
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        input_yuv = rgb2ycbcr_pt(x)
        target_yuv = rgb2ycbcr_pt(y)
        # Get just the UV channels
        input_uv = input_yuv[:, 1:, :, :]
        target_uv = target_yuv[:, 1:, :, :]
        input_uv_downscale = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_uv)
        target_uv_downscale = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_uv)
        return (
            self.criterion(input_uv_downscale, target_uv_downscale) * self.loss_weight
        )


@LOSS_REGISTRY.register()
class AverageLoss(nn.Module):
    """Averaging Downscale loss"""

    def __init__(
        self, loss_weight: float, criterion: str = "l1", scale: int = 4
    ) -> None:
        super().__init__()
        self.ds_f = torch.nn.AvgPool2d(kernel_size=int(scale))
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.criterion(self.ds_f(x), self.ds_f(y)) * self.loss_weight


@LOSS_REGISTRY.register()
class BicubicLoss(nn.Module):
    """Bicubic Downscale loss"""

    def __init__(
        self, loss_weight: float, criterion: str = "l1", scale: int = 4
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ds_f = lambda x: torch.nn.Sequential(
            v2.Resize(
                [x.shape[2] // self.scale, x.shape[3] // self.scale],
                InterpolationMode.BICUBIC,
            ),
            v2.GaussianBlur([5, 5], [0.5, 0.5]),
        )(x)
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.criterion(self.ds_f(x), self.ds_f(y)) * self.loss_weight


@LOSS_REGISTRY.register()
class LumaLoss(nn.Module):
    def __init__(self, loss_weight: float, criterion: str = "l1") -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_luma = rgb_to_luma(x)
        y_luma = rgb_to_luma(y)
        loss = self.criterion(x_luma, y_luma) * self.loss_weight
        return loss


@LOSS_REGISTRY.register()
class HSLuvLoss(nn.Module):
    def __init__(
        self, loss_weight: float, criterion: str = "l1", downscale_factor: int = 1
    ) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor
        self.loss_weight = loss_weight
        self.criterion_type = criterion

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward_once(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.downscale_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=1 / self.downscale_factor,
                mode="bicubic",
                antialias=True,
            ).clamp(0, 1)

        x_hsluv = rgb_to_hsluv(x)

        # hue: 0 to 360. normalize
        x_hue = x_hsluv[:, 0, :, :] / 360

        # saturation: 0 to 100. normalize
        x_saturation = torch.round(x_hsluv[:, 1, :, :], decimals=20) / 100

        # lightness: 0 to 100. normalize
        x_lightness = x_hsluv[:, 2, :, :] / 100

        return x_hue, x_saturation, x_lightness

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_hue, x_saturation, x_lightness = self.forward_once(x)
        y_hue, y_saturation, y_lightness = self.forward_once(y)

        eps = 0.1

        # find the shortest distance between angles on a circle. TODO: this is l1, implement other criteria
        # since the max distance is 0.5, multiply by 2 to normalize
        hue_diff = torch.min(torch.abs(x_hue - y_hue), 1 - torch.abs(x_hue - y_hue)) * 2
        # hue diff between two grayscale colors is 0
        hue_diff = torch.where((x_saturation < eps) & (y_saturation < eps), 0, hue_diff)
        # hue diff between grayscale and non-grayscale is maximum, scaled by max saturation
        hue_diff = torch.where(
            ((x_saturation < eps) & (y_saturation > eps))
            | ((x_saturation > eps) & (y_saturation < eps)),
            torch.max(x_saturation, y_saturation),
            hue_diff,
        )
        # hue diff between black or white is 0
        hue_diff = torch.where((x_lightness < eps) & (y_lightness < eps), 0, hue_diff)
        hue_diff = torch.where(
            (x_lightness > 1 - eps) & (y_lightness > eps - 1), 0, hue_diff
        )

        hue_loss = torch.mean(hue_diff) * 1 / 3
        saturation_loss = self.criterion(x_saturation, y_saturation) * 1 / 3
        lightness_loss = self.criterion(x_lightness, y_lightness) * 1 / 3

        return (hue_loss + saturation_loss + lightness_loss) * self.loss_weight
