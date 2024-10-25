import torch
import torchvision
from torch import Tensor, nn
from torchvision.models import VGG19_Weights

from traiNNer.utils.registry import LOSS_REGISTRY

VGG19_LAYERS = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_4",
    "relu3_4",
    "pool3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_4",
    "relu4_4",
    "pool4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_4",
    "relu5_4",
    "pool5",
]


@LOSS_REGISTRY.register()
class PDLoss(nn.Module):
    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        w_lambda: float = 0.01,
        loss_weight: float = 1,
        alpha: list[float] | None = None,
    ) -> None:
        super().__init__()
        if layer_weights is None:
            layer_weights = {
                "conv1_2": 0.1,
                "relu1_2": 0.1,
                "conv2_2": 0.1,
                "relu2_2": 0.1,
                "conv3_4": 1,
                "relu3_4": 1,
                "conv4_4": 1,
                "relu4_4": 1,
                "conv5_4": 1,
                "relu5_4": 1,
            }
        if alpha is None:
            alpha = []
            for k in layer_weights:
                if k.startswith("conv"):
                    alpha.append(0.0)
                else:
                    alpha.append(1.0)

        self.vgg = VGG(list(layer_weights.keys())).cuda()
        self.loss_weight = loss_weight
        self.w_lambda = w_lambda
        self.layer_weights = layer_weights
        self.alpha = alpha

        self.criterion1 = None
        self.criterion2 = None

        if any(x < 1 for x in self.alpha):
            self.criterion1 = nn.L1Loss()
        if any(x > 0 for x in self.alpha):
            self.criterion2 = self.w_distance

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

    def forward_once(self, x: Tensor) -> dict[str, Tensor]:
        return self.vgg(x)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        x_vgg, gt_vgg = self.forward_once(x), self.forward_once(gt.detach())
        score = torch.tensor(0.0, device=x.device)
        for i, k in enumerate(x_vgg):
            alpha = self.alpha[i]
            s = torch.tensor(0.0, device=score.device)
            if alpha < 1:
                assert self.criterion1 is not None
                temp = self.criterion1(x_vgg[k], gt_vgg[k]) * (1 - alpha)
                s += temp
                # print("l1", k, temp)
            if alpha > 0:
                assert self.criterion2 is not None
                temp = self.criterion2(x_vgg[k], gt_vgg[k]) * alpha * self.w_lambda
                s += temp
                # print("pd", k, temp)

            s *= self.layer_weights[k]
            score += s

        return score * self.loss_weight


class VGG(nn.Module):
    def __init__(self, layer_name_list: list[str]) -> None:
        super().__init__()

        vgg_pretrained_features = torchvision.models.vgg19(
            weights=VGG19_Weights.DEFAULT
        ).features
        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)

        self._disable_inplace_relu(vgg_pretrained_features)

        self.stages: nn.ModuleDict = nn.ModuleDict()
        stage_breakpoints = {}

        for v in layer_name_list:
            stage_breakpoints[v] = VGG19_LAYERS.index(v) + 1

        prev_breakpoint = 0
        for layer_name, idx in stage_breakpoints.items():
            self.stages[layer_name] = nn.Sequential()
            for x in range(prev_breakpoint, idx):
                self.stages[layer_name].add_module(str(x), vgg_pretrained_features[x])
            prev_breakpoint = idx

        for _layer_name, stage in self.stages.items():
            stage[0] = self._change_padding_mode(stage[0], "replicate")  # pyright: ignore[reportIndexIssue]
            break

        for param in self.parameters():
            param.requires_grad = False

        for _, stage in self.stages.items():
            stage.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

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

    @staticmethod
    def _disable_inplace_relu(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = (x - self.mean) / self.std

        feats = {}
        for layer_name, stage in self.stages.items():
            last_h = h if not feats else feats[next(reversed(feats))]
            feats[layer_name] = stage(last_h)

        return feats
