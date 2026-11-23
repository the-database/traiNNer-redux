from typing import Any, Literal

import antialiased_cnns
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

VGG19_AA_LAYERS = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool1",
    "blur1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool2",
    "blur2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_4",
    "relu3_4",
    "pool3",
    "blur3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_4",
    "relu4_4",
    "pool4",
    "blur4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_4",
    "relu5_4",
    "pool5",
    "blur5",
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

# Largest tap gradient after _ScaleTapGrads. The fp16 backward through VGG is equally
# accurate for anything from about 2^-15 to 2^3; it underflows below and overflows
# above.
TAP_GRAD_PEAK_LOG2 = -7


class _UnscaleInputGrad(torch.autograd.Function):
    """Identity on x. Also returns a token for _ScaleTapGrads, whose backward sends
    the scale it applied back as the token's gradient; this divides it out."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> tuple[Tensor, Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return x.view_as(x), x.new_zeros(())

    @staticmethod
    def backward(ctx: Any, grad: Tensor, scale: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return grad.float() / scale


class _ScaleTapGrads(torch.autograd.Function):
    """Identity on the tap features. Backward multiplies their gradients by the power
    of 2 that puts the largest at 2^TAP_GRAD_PEAK_LOG2, so the fp16 backward through
    VGG neither underflows nor overflows. Chosen on the GPU, with no sync."""

    @staticmethod
    def forward(ctx: Any, token: Tensor, *feats: Tensor) -> tuple[Tensor, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return tuple(f.view_as(f) for f in feats)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        amax = torch.stack([g.abs().amax() for g in grads]).amax()
        scale = torch.exp2(
            TAP_GRAD_PEAK_LOG2 - torch.ceil(torch.log2(amax.float().clamp_min(1e-30)))
        )
        return (scale, *(g * scale for g in grads))


def _conv_fp16(
    conv: nn.Conv2d, x: Tensor, weight: Tensor, bias: Tensor | None
) -> Tensor:
    x = x.half()
    padding = conv.padding
    if conv.padding_mode != "zeros":
        ph, pw = int(conv.padding[0]), int(conv.padding[1])
        x = F.pad(x, (pw, pw, ph, ph), mode=conv.padding_mode)
        padding = 0
    return F.conv2d(x, weight, bias, conv.stride, padding).float()  # pyright: ignore[reportArgumentType]


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        layer_weights: dict[str, float] | None = None,
        w_lambda: float = 0.01,
        alpha: list[float] | None = None,
        criterion: Literal[
            "pd+l1", "fd+l1", "pd", "fd", "charbonnier", "l1"
        ] = "charbonnier",
        num_proj_fd: int = 256,
        phase_weight_fd: float = 1.0,
        stride_fd: int = 1,
        clip_min: int = 0,
        antialiased: bool = False,
        verbose_logging: bool = False,
    ) -> None:
        super().__init__()

        self.clip_min = clip_min
        self.verbose_logging = verbose_logging
        use_conv_layers = False
        use_relu_layers = False

        if criterion in VGG19_CONV_CRITERION:
            use_conv_layers = True

        if criterion in VGG19_RELU_CRITERION:
            use_relu_layers = True

        if layer_weights is None:
            layer_weights = {}

            if criterion in VGG19_CONV_CRITERION:
                layer_weights |= VGG19_CONV_LAYER_WEIGHTS

            if criterion in VGG19_RELU_CRITERION:
                layer_weights |= VGG19_RELU_LAYER_WEIGHTS

        self.vgg = VGG(list(layer_weights.keys()), antialiased=antialiased).to(
            memory_format=torch.channels_last
        )  # pyright: ignore[reportCallIssue]

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
            if use_relu_layers or criterion == "l1":
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

        return (s_amplitude + s_phase * self.phase_weight_fd).mean()

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

    def _reduce_delta(self, diffs: dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        total = torch.zeros((), device=next(iter(diffs.values())).device)
        for i, (k, d) in enumerate(diffs.items()):
            if self.criterion1 is charbonnier_loss:
                layer_loss = torch.sqrt(d * d + 1e-12).mean()
            else:
                layer_loss = d.abs().mean()
            layer_loss = layer_loss * (1 - self.alpha[i]) * self.layer_weights[k]
            if self.verbose_logging:
                losses[k] = layer_loss
            else:
                total = total + layer_loss
        if self.verbose_logging:
            return losses
        return torch.clamp(total, min=self.clip_min)

    def forward_once(self, x: Tensor) -> dict[str, Tensor]:
        return self.vgg(x)

    def forward_delta(self, x: Tensor, gt: Tensor) -> dict[str, Tensor]:
        """Differences VGG(x) - VGG(gt) at the taps, for L1 and Charbonnier.

        Carries a = VGG(gt) and d = VGG(x) - VGG(gt) instead of the two features: a
        conv maps d to W*d (replicate padding is linear too), and the other layers
        to op(a + d) - op(a), in fp32. Exactly the same loss, but rounding lands on
        the small d rather than on two large features that nearly cancel, which
        matters more the closer x gets to gt. That lets the convs run in fp16 and
        still be more accurate than separate fp32 (TF32) passes."""
        vgg = self.vgg
        with torch.autocast(x.device.type, enabled=False):
            x, gt = x.float(), gt.detach().float()
            d, token = _UnscaleInputGrad.apply((x - gt) / vgg.std)  # pyright: ignore[reportGeneralTypeIssues,reportOperatorIssue]
            with torch.no_grad():
                a = (gt - vgg.mean) / vgg.std  # pyright: ignore[reportOperatorIssue]
            taps = []
            for stage in vgg.stages.values():
                for m in stage.children():
                    if isinstance(m, nn.Conv2d):
                        assert m.bias is not None
                        weight = m.weight.half()
                        with torch.no_grad():
                            a = _conv_fp16(m, a, weight, m.bias.half())
                        d = _conv_fp16(m, d, weight, None)
                    else:
                        with torch.no_grad():
                            a_next = m(a)
                        d = m(a + d) - a_next
                        a = a_next
                taps.append(d)
            taps = _ScaleTapGrads.apply(token, *taps)
        return dict(zip(vgg.stages.keys(), taps, strict=True))  # pyright: ignore[reportArgumentType]

    # pd and fd run VGG on x and gt separately, in fp32
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor | dict[str, Tensor]:
        if self.criterion2 is None and self.criterion1 is not None:
            return self._reduce_delta(self.forward_delta(x, gt))

        x_vgg, gt_vgg = self.forward_once(x), self.forward_once(gt.detach())

        losses: dict[str, Tensor] = {}
        total = torch.zeros((), device=x.device)
        criterion2_i = 0
        for i, k in enumerate(x_vgg):
            alpha = self.alpha[i]
            layer_loss = torch.zeros((), device=x.device)

            if alpha < 1:
                assert self.criterion1 is not None
                loss1 = self.criterion1(x_vgg[k], gt_vgg[k]) * (1 - alpha)
                layer_loss = layer_loss + loss1

            if alpha > 0:
                assert self.criterion2 is not None
                loss2 = (
                    self.criterion2(x_vgg[k], gt_vgg[k], criterion2_i)
                    * alpha
                    * self.w_lambda
                )
                layer_loss = layer_loss + loss2
                criterion2_i += 1

            layer_loss = layer_loss * self.layer_weights[k]

            if self.verbose_logging:
                losses[k] = layer_loss
            else:
                total = total + layer_loss

        if self.verbose_logging:
            return losses
        return torch.clamp(total, min=self.clip_min)


@LOSS_REGISTRY.register()
class PerceptualFP16Loss(PerceptualLoss):
    """Deprecated: the same as perceptualloss, which now runs in fp16 where that's
    as accurate."""

    deprecated = "the same as perceptualloss now, which runs in fp16 where that's as accurate. Use type: perceptualloss."


class VGG(nn.Module):
    def __init__(self, layer_name_list: list[str], antialiased: bool = False) -> None:
        super().__init__()

        self.antialiased = antialiased

        if antialiased:
            vgg_pretrained_features = self._build_antialiased_vgg19()
            layer_list = VGG19_AA_LAYERS
        else:
            model = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).to(
                memory_format=torch.channels_last
            )  # pyright: ignore[reportCallIssue]
            vgg_pretrained_features = model.features
            layer_list = VGG19_LAYERS

        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)

        self._disable_inplace_relu(vgg_pretrained_features)

        self.stages: nn.ModuleDict = nn.ModuleDict()
        stage_breakpoints = {}

        for v in layer_name_list:
            stage_breakpoints[v] = layer_list.index(v) + 1

        prev_breakpoint = 0
        for layer_name, idx in sorted(stage_breakpoints.items(), key=lambda x: x[1]):
            self.stages[layer_name] = nn.Sequential()
            for x in range(prev_breakpoint, idx):
                self.stages[layer_name].add_module(str(x), vgg_pretrained_features[x])
            prev_breakpoint = idx

        for stage in self.stages.values():
            stage[0] = self._change_padding_mode(stage[0], "replicate")  # pyright: ignore[reportIndexIssue]
            break

        for param in self.parameters():
            param.requires_grad = False

        for stage in self.stages.values():
            stage.eval()

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _build_antialiased_vgg19(self) -> nn.Sequential:
        model = antialiased_cnns.vgg19(pretrained=True, filter_size=4)
        features = model.features
        assert isinstance(features, nn.Sequential)

        return features

    @staticmethod
    def _change_padding_mode(
        conv: nn.Module,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
    ) -> nn.Conv2d:
        assert isinstance(conv.in_channels, int)
        assert isinstance(conv.out_channels, int)
        assert isinstance(conv.kernel_size, int | tuple)
        assert isinstance(conv.stride, int | tuple)
        assert isinstance(conv.padding, int | tuple)
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

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = (x - self.mean) / self.std  # pyright: ignore[reportOperatorIssue]

        feats = {}
        for layer_name, stage in self.stages.items():
            last_h = h if not feats else feats[next(reversed(feats))]
            feats[layer_name] = stage(last_h)

        return feats
