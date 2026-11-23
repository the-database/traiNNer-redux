from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.util import store_hyperparameters
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


@torch.compile(fullgraph=True)
def _bias_prelu(x: Tensor, bias: Tensor, slope: Tensor) -> Tensor:
    z = x + bias.to(x.dtype).view(1, -1, 1, 1)
    slope = slope.to(x.dtype).view(1, -1, 1, 1)
    return torch.where(z > 0, z, z * slope)


@store_hyperparameters()
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
        learn_residual: bool = True,
        fused_training: bool = False,
    ) -> None:
        super().__init__()
        self.hyperparameters.pop("fused_training", None)
        self.fused_training = fused_training
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.learn_residual = learn_residual

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)  # type: ignore

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)  # type: ignore

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if (
            self.fused_training
            and self.training
            and x.is_cuda
            and self.act_type == "prelu"
        ):
            for i in range(0, len(self.body) - 1, 2):
                conv = cast(nn.Conv2d, self.body[i])
                activation = cast(nn.PReLU, self.body[i + 1])
                out = F.conv2d(out, conv.weight, padding=1)
                out = _bias_prelu(out, cast(Tensor, conv.bias), activation.weight)
            out = self.body[-1](out)
        else:
            for layer in self.body:
                out = layer(out)

        if self.training and self.learn_residual:
            # Each LR channel is broadcast over its r*r PixelShuffle phases.
            out.unflatten(1, (self.num_out_ch, self.upscale**2)).add_(x.unsqueeze(2))
            return self.upsampler(out)

        out = self.upsampler(out)

        if self.learn_residual:
            # add the nearest upsampled image, so that the network learns the residual
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            out += base

        return out


@ARCH_REGISTRY.register()
def compact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 16,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
    fused_training: bool = False,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
        fused_training=fused_training,
    )


@ARCH_REGISTRY.register()
def ultracompact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 8,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
    fused_training: bool = False,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
        fused_training=fused_training,
    )


@ARCH_REGISTRY.register()
def superultracompact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 24,
    num_conv: int = 8,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
    fused_training: bool = False,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
        fused_training=fused_training,
    )
