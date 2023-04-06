import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights


@ARCH_REGISTRY.register()
class RRDB2C2Net(nn.Module):

    def __init__(
        self,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        act_type="leakyrelu",
    ):
        in_nc = num_in_ch
        out_nc = num_out_ch
        upscale = scale
        nf = num_feat
        nb = num_block
        gc = num_grow_ch

        super(RRDB2C2Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        fea_conv = conv_block(in_nc, nf, act_type=None)
        rb_blocks = [RRDB(
            nf,
            act_type=act_type,
        ) for _ in range(nb)]
        LR_conv = conv_block(
            nf,
            nf,
            act_type=None,
        )

        upsampler = [upconv_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = conv_block(nf, nf, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, act_type=None)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
        )

    def forward(self, x, outm=None):
        x = self.model(x)

        if (
                outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://githucom/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(x) + 1.0) / 2.0
        elif outm == "tanh":  # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm == "clamp":
            return torch.clamp(x, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nf,
        gc=32,
        act_type="leakyrelu",
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):

    def __init__(
        self,
        nf=64,
        gc=32,
        act_type="leakyrelu",
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = conv_block(
            nf,
            gc,
            act_type=act_type,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            act_type=act_type,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            act_type=act_type,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            act_type=act_type,
        )
        last_act = None
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            act_type=last_act,
        )

        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# 2x2x2 Conv Block
def conv_block(
    in_nc,
    out_nc,
    act_type="relu",
):
    return sequential(
        nn.Conv2d(in_nc, out_nc, kernel_size=2, padding=1),
        nn.Conv2d(out_nc, out_nc, kernel_size=2, padding=0),
        act(act_type) if act_type else None,
    )


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    # beta: for swish
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type in ("leakyrelu", "lrelu"):
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "tanh":  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == "sigmoid":  # [0, 1] range output
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError("activation layer [{:s}] is not found".format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        return "ShortcutBlock"


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def upconv_block(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="nearest",
    convtype="Conv2D",
):
    """Upconv layer described in
    https://distill.pub/2016/deconv-checkerboard/.
    Example to replace deconvolutions:
        - from: nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1)
        - to: upconv_block(in_nc, out_nc,kernel_size=3, stride=1, act_type=None)
    """
    upscale_factor = ((1, upscale_factor, upscale_factor) if convtype == "Conv3D" else upscale_factor)
    upsample = Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        act_type=act_type,
    )
    return sequential(upsample, conv)


class Upsample(nn.Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners
        # self.interp = nn.functional.interpolate

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        # return self.interp(x, size=self.size,
        #     scale_factor=self.scale_factor, mode=self.mode,
        #     align_corners=self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info
