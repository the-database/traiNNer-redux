import torch
import torchvision
from torch import Tensor, nn
from torchvision.models import VGG19_Weights

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PDLoss(nn.Module):
    def __init__(self, w_lambda: float = 0.01, loss_weight: float = 1) -> None:
        super().__init__()
        self.vgg = VGG().cuda()
        self.loss_weight = loss_weight * w_lambda

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def w_distance(self, x_vgg: Tensor, y_vgg: Tensor) -> Tensor:
        x_vgg = x_vgg / (torch.sum(x_vgg, dim=(2, 3), keepdim=True) + 1e-14)
        y_vgg = y_vgg / (torch.sum(y_vgg, dim=(2, 3), keepdim=True) + 1e-14)

        x_vgg = x_vgg.view(x_vgg.size()[0], x_vgg.size()[1], -1).contiguous()
        y_vgg = y_vgg.view(y_vgg.size()[0], y_vgg.size()[1], -1).contiguous()

        cdf_x_vgg = torch.cumsum(x_vgg, dim=-1)
        cdf_y_vgg = torch.cumsum(y_vgg, dim=-1)

        cdf_distance = torch.sum(torch.abs(cdf_x_vgg - cdf_y_vgg), dim=-1)

        cdf_loss = cdf_distance.mean()

        return cdf_loss

    def forward_once(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        return self.vgg(x)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        x_vgg, gt_vgg = self.forward_once(x), self.forward_once(gt.detach())
        return self.w_distance(x_vgg, gt_vgg) * self.loss_weight


class VGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            weights=VGG19_Weights.DEFAULT
        ).features
        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)
        self.vgg = nn.Sequential()

        for x in range(23):
            self.vgg.add_module(str(x), vgg_pretrained_features[x])

        self.vgg[0] = self._change_padding_mode(self.vgg[0], "replicate")

        for param in self.parameters():
            param.requires_grad = False

        self.vgg.eval()

    @staticmethod
    def _change_padding_mode(conv: nn.Module, padding_mode: str) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            padding_mode=padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            if new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return new_conv

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor) -> Tensor:
        return self.vgg(x)
