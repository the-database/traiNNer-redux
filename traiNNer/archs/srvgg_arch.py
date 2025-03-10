import torch.nn.functional as F  # noqa: N812
from spandrel.util import store_hyperparameters
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


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
    ) -> None:
        super().__init__()
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
        for i in range(len(self.body)):
            out = self.body[i](out)

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
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
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
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
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
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
    )
