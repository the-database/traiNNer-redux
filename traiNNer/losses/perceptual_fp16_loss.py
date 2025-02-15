from typing import Literal

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models import VGG19_Weights

from traiNNer.losses.basic_loss import charbonnier_loss
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

VGG19_CONV_LAYER_WEIGHTS = {
    "conv1_2": 0.1,
    "conv2_2": 0.1,
    "conv3_4": 1,
    "conv4_4": 1,
    "conv5_4": 1,
}

VGG19_RELU_LAYER_WEIGHTS = {
    "relu1_2": 0.1,
    "relu2_2": 0.1,
    "relu3_4": 1,
    "relu4_4": 1,
    "relu5_4": 1,
}

VGG19_CONV_CRITERION = {"l1", "charbonnier", "pd+l1", "fd+l1"}
VGG19_RELU_CRITERION = {"pd", "fd", "pd+l1", "fd+l1"}

VGG19_CHANNELS = [64, 128, 256, 512, 512]


@LOSS_REGISTRY.register()
class PerceptualFP16Loss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        layer_weights: dict[str, float] | None = None,
        w_lambda: float = 0.01,
        alpha: list[float] | None = None,
        criterion: Literal["pd+l1", "fd+l1", "pd", "fd", "charbonnier", "l1"] = "pd+l1",
        num_proj_fd: int = 256,
        phase_weight_fd: float = 1.0,
        stride_fd: int = 1,
    ) -> None:
        super().__init__()

        use_conv_layers = False
        use_relu_layers = False

        if layer_weights is None:
            layer_weights = {}

            if criterion in VGG19_CONV_CRITERION:
                use_conv_layers = True
                layer_weights |= VGG19_CONV_LAYER_WEIGHTS

            if criterion in VGG19_RELU_CRITERION:
                use_relu_layers = True
                layer_weights |= VGG19_RELU_LAYER_WEIGHTS

        self.vgg = VGG(list(layer_weights.keys())).to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]

        if alpha is None:
            alpha = []
            for k in self.vgg.stages:
                if k.startswith("conv"):
                    alpha.append(0.0)
                else:
                    alpha.append(1.0)

        self.loss_weight = loss_weight
        self.w_lambda = w_lambda
        self.layer_weights = layer_weights
        self.alpha = alpha
        self.phase_weight_fd = phase_weight_fd
        self.stride_fd = stride_fd

        self.criterion1 = None
        self.criterion2 = None

        if any(x < 1 for x in self.alpha) and use_conv_layers:
            if use_relu_layers:
                self.criterion1 = nn.L1Loss()
            elif criterion == "l1":
                self.criterion1 = nn.L1Loss()
            elif criterion == "charbonnier":
                self.criterion1 = charbonnier_loss
            else:
                raise NotImplementedError(
                    f"{criterion} criterion has not been supported."
                )
        if any(x > 0 for x in self.alpha) and use_relu_layers:
            if "pd" in criterion.lower():
                self.criterion2 = self.pd
            elif "fd" in criterion.lower():
                self.criterion2 = self.fd
                self.init_random_projections_fd(num_proj_fd)
            else:
                raise NotImplementedError(
                    f"{criterion} criterion has not been supported."
                )

    def init_random_projections_fd(self, num_proj: int, patch_size: int = 5) -> None:
        for i in range(len(VGG19_CHANNELS)):
            rand = torch.randn(num_proj, VGG19_CHANNELS[i], patch_size, patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)

    def forward_once_fd(self, x: Tensor, y: Tensor, idx: int) -> Tensor:
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        assert isinstance(self.stride_fd, int)
        rand = self.__getattr__(f"rand_{idx}")
        assert isinstance(rand, Tensor)
        projx = F.conv2d(x, rand, stride=self.stride_fd)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride_fd)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1, 2])
        return s

    def fd(self, x_vgg: Tensor, y_vgg: Tensor, i: int) -> Tensor:
        # Transform to Fourier Space
        fft_x = torch.fft.fftn(x_vgg, dim=(-2, -1))
        fft_y = torch.fft.fftn(y_vgg, dim=(-2, -1))

        # get the magnitude and phase of the extracted features
        x_mag = torch.abs(fft_x)
        x_phase = torch.angle(fft_x)
        y_mag = torch.abs(fft_y)
        y_phase = torch.angle(fft_y)

        s_amplitude = self.forward_once_fd(x_mag, y_mag, i)
        s_phase = self.forward_once_fd(x_phase, y_phase, i)

        return s_amplitude + s_phase * self.phase_weight_fd

    def pd(self, x_vgg: Tensor, y_vgg: Tensor, _: int = -1) -> Tensor:
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

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        x_vgg, gt_vgg = self.forward_once(x), self.forward_once(gt.detach())
        score1 = torch.tensor(0.0, device=x.device)
        score2 = None
        criterion2_i = 0
        for i, k in enumerate(x_vgg):
            alpha = self.alpha[i]
            s1 = torch.tensor(0.0, device=x.device)
            s2 = None
            if alpha < 1:
                assert self.criterion1 is not None
                temp = self.criterion1(x_vgg[k], gt_vgg[k]) * (1 - alpha)
                # print("l1", k, temp)
                s1 += temp
            if alpha > 0:
                assert self.criterion2 is not None
                temp = (
                    self.criterion2(x_vgg[k], gt_vgg[k], criterion2_i)
                    * alpha
                    * self.w_lambda
                )
                if score2 is None:
                    score2 = torch.zeros(temp.shape, device=x.device)
                if s2 is None:
                    s2 = torch.zeros(temp.shape, device=x.device)
                # print("fd", k, temp)
                s2 += temp
                criterion2_i += 1

            s1 *= self.layer_weights[k]
            score1 += s1
            if s2 is not None:
                assert score2 is not None
                s2 *= self.layer_weights[k]
                score2 += s2

        score = score1
        if score2 is not None:
            score += score2.mean()

        return score * self.loss_weight


class VGG(nn.Module):
    def __init__(self, layer_name_list: list[str]) -> None:
        super().__init__()

        model = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).to(
            memory_format=torch.channels_last
        )  # pyright: ignore[reportCallIssue]

        vgg_pretrained_features = model.features
        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)

        self._disable_inplace_relu(vgg_pretrained_features)

        self.stages: nn.ModuleDict = nn.ModuleDict()
        stage_breakpoints = {}

        for v in layer_name_list:
            stage_breakpoints[v] = VGG19_LAYERS.index(v) + 1

        prev_breakpoint = 0
        for layer_name, idx in sorted(stage_breakpoints.items(), key=lambda x: x[1]):
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

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @staticmethod
    def _change_padding_mode(conv: nn.Module, padding_mode: str) -> nn.Conv2d:
        assert isinstance(conv.in_channels, int)
        assert isinstance(conv.out_channels, int)
        assert isinstance(conv.kernel_size, int)
        assert isinstance(conv.stride, int)
        assert isinstance(conv.padding, int)
        assert isinstance(conv.weight, Tensor)
        assert isinstance(conv.bias, Tensor)
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

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        assert isinstance(self.mean, Tensor)
        assert isinstance(self.std, Tensor)
        h = (x - self.mean) / self.std

        feats = {}
        for layer_name, stage in self.stages.items():
            last_h = h if not feats else feats[next(reversed(feats))]
            feats[layer_name] = stage(last_h)

        return feats
