import math

from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

AFFINE_LIST = ["basicblock", "bottleneck", "mbconv", "basicblock_dw"]


import torch
from torch.nn.parameter import Parameter

try:
    from detectron2.layers import ModulatedDeformConv

    DF_CONV_READY = True

    class DeformConv(nn.Module):
        def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            groups: int = 1,
            dilation: int = 1,
            bias: bool = False,
        ) -> None:
            super().__init__()
            self.df_conv = ModulatedDeformConv(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
            self.conv_offset = conv3x3(in_planes, 27, bias=True)

        def forward(self, x):
            offset_mask = self.conv_offset(x)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            return self.df_conv(x, offset, mask)
except:
    DF_CONV_READY = False


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
    df_conv: bool = False,
) -> nn.Module:
    """3x3 convolution with padding"""
    conv_op = DeformConv if DF_CONV_READY and df_conv else nn.Conv2d
    return conv_op(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Swish(nn.Module):
    # An ordinary implementation of Swish function
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    # A memory-efficient implementation of Swish function
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def get_activation(activation: str = "relu") -> nn.Module:
    """Get the specified activation layer.
    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in [
        "relu",
        "leaky_relu",
        "elu",
        "silu",
        "gelu",
        "swish",
        "efficient_swish",
        "none",
    ], f"Get unknown activation key {activation}"
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "silu": nn.SiLU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "efficient_swish": MemoryEfficientSwish(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]


def get_num_params(model):
    num_param = sum([param.nelement() for param in model.parameters()])
    return num_param


class ResidualBase(nn.Module):
    def __init__(
        self, stochastic_depth: bool = False, prob: float = 1.0, multFlag: bool = True
    ) -> None:
        super().__init__()
        self.sd = stochastic_depth
        if stochastic_depth:
            self.prob = prob
            self.multFlag = multFlag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        return (
            self._forward_train(x, identity)
            if self.training
            else self._forward_test(x, identity)
        )

    def _forward_train(self, x, identity) -> torch.Tensor:
        if not self.sd:  # no stochastic depth
            res = self._forward_res(x)
            return identity + res

        if torch.rand(1) < self.prob:  # no skip
            for param in self.parameters():
                param.requires_grad = True
            res = self._forward_res(x)
            return identity + res

        # This block is skipped during training
        for param in self.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity) -> torch.Tensor:
        res = self._forward_res(x)
        if self.sd and self.multFlag:
            res *= self.prob

        return identity + res

    def _forward_res(self, _) -> torch.Tensor:
        # Residual forward function should be
        # defined in child classes.
        raise NotImplementedError


class PreActBasicBlock(ResidualBase):
    def __init__(
        self,
        planes: int,
        stochastic_depth: bool = False,
        act_mode: str = "relu",
        prob: float = 1.0,
        multFlag: bool = True,
        zero_inti_residual: bool = False,
        affine_init_w: float = 0.1,
        **_,
    ) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv3x3(planes, planes)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff2.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class PreActBasicBlockDW(ResidualBase):
    def __init__(
        self,
        planes: int,
        stochastic_depth: bool = False,
        act_mode: str = "relu",
        prob: float = 1.0,
        multFlag: bool = True,
        zero_inti_residual: bool = False,
        affine_init_w: float = 0.1,
        reduction: int = 8,
    ) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv3x3(planes, planes, groups=planes)
        self.se1 = SEBlock(planes, reduction, act_mode)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes, groups=planes)
        self.se2 = SEBlock(planes, reduction, act_mode)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff2.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.se1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.se2(x)

        return x


class PreActBottleneck(ResidualBase):
    def __init__(
        self,
        planes: int,
        stochastic_depth: bool = False,
        act_mode: str = "relu",
        prob: float = 1.0,
        multFlag: bool = True,
        zero_inti_residual: bool = False,
        affine_init_w: float = 0.1,
        **_,
    ) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv1x1(planes, planes)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes)

        self.aff3 = Affine2d(planes, affine_init_w)
        self.conv3 = conv1x1(planes, planes)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff3.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.aff3(x)
        x = self.act(x)
        x = self.conv3(x)

        return x


class MBConvBlock(ResidualBase):
    def __init__(
        self,
        planes: int,
        stochastic_depth: bool = False,
        act_mode: str = "relu",
        prob: float = 1.0,
        multFlag: bool = True,
        reduction: int = 8,
        zero_inti_residual: bool = False,
        affine_init_w: float = 0.1,
    ) -> None:
        super().__init__(stochastic_depth, prob, multFlag)

        self.conv1 = conv1x1(planes, planes)
        self.aff1 = Affine2d(planes, affine_init_w)

        self.conv2 = conv3x3(planes, planes, groups=planes)  # depth-wise
        self.aff2 = Affine2d(planes, affine_init_w)
        self.se = SEBlock(planes, reduction, act_mode)

        self.conv3 = conv1x1(planes, planes)
        self.aff3 = Affine2d(planes, affine_init_w)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff3.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.aff1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.aff2(x)
        x = self.act(x)

        x = self.se(x)

        x = self.conv3(x)
        x = self.aff3(x)  # no activation

        return x


class EDSRBlock(ResidualBase):
    def __init__(
        self,
        planes: int,
        bias: bool = True,
        act_mode: str = "relu",
        res_scale: float = 0.1,
        res_scale_learnable: bool = False,
        stochastic_depth: bool = False,
        prob: float = 1.0,
        multFlag: bool = True,
        **_,
    ):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias),
        )

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlock(ResidualBase):
    def __init__(
        self,
        planes: int,
        bias: bool = True,
        act_mode: str = "relu",
        res_scale: float = 0.1,
        reduction: int = 16,
        res_scale_learnable: bool = False,
        stochastic_depth: bool = False,
        prob: float = 1.0,
        multFlag: bool = True,
        normal_init_std: float | None = None,
        **_,
    ):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias),
            SEBlock(planes, reduction, act_mode),
        )

        # normal initialization
        if normal_init_std is not None:
            for idx in [0, 2]:
                nn.init.normal_(self.body[idx].weight, 0.0, normal_init_std)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlockDW(ResidualBase):
    """RCAN building block with depth-wise convolution for the second conv layer."""

    def __init__(
        self,
        planes: int,
        bias: bool = True,
        act_mode: str = "relu",
        res_scale: float = 0.1,
        reduction: int = 16,
        res_scale_learnable: bool = False,
        stochastic_depth: bool = False,
        prob: float = 1.0,
        multFlag: bool = True,
        **_,
    ):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode),
        )

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlockAllDW(ResidualBase):
    """RCAN building block with depth-wise convolution for all conv layers. An
    additional squeeze-and-excitation (SE) block is used for the cross-channel
    communication.
    """

    def __init__(
        self,
        planes: int,
        bias: bool = True,
        act_mode: str = "relu",
        res_scale: float = 0.1,
        reduction: int = 16,
        res_scale_learnable: bool = False,
        stochastic_depth: bool = False,
        prob: float = 1.0,
        multFlag: bool = True,
        **_,
    ):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode),
        )

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class SEBlock(nn.Module):
    def __init__(self, planes: int, reduction: int = 8, act_mode: str = "relu"):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(planes, planes // reduction, kernel_size=1),
            get_activation(act_mode),
            nn.Conv2d(planes // reduction, planes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Affine2d(nn.Module):
    def __init__(self, planes: int, init_w: float = 0.1) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(1, planes, 1, 1))
        self.bias = Parameter(torch.zeros(1, planes, 1, 1))
        nn.init.constant_(self.weight, init_w)

    def forward(self, x):
        return x * self.weight + self.bias


class Upsampler(nn.Sequential):
    def __init__(
        self, scale: int, planes: int, act_mode: str = "relu", use_affine: bool = True
    ):
        m = []
        if (scale & (scale - 1)) == 0:  # is power of 2
            if use_affine:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv3x3(planes, 4 * planes))
                    m.append(nn.PixelShuffle(2))
                    m.append(Affine2d(planes))
                    m.append(get_activation(act_mode))
            else:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv3x3(planes, 4 * planes, bias=True))
                    m.append(nn.PixelShuffle(2))
                    m.append(get_activation(act_mode))

        elif scale == 3:
            if use_affine:
                m.append(conv3x3(planes, 9 * planes))
                m.append(nn.PixelShuffle(3))
                m.append(Affine2d(planes))
                m.append(get_activation(act_mode))
            else:
                m.append(conv3x3(planes, 9 * planes, bias=True))
                m.append(nn.PixelShuffle(3))
                m.append(get_activation(act_mode))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


BLOCK_DICT = {
    "basicblock": PreActBasicBlock,
    "bottleneck": PreActBottleneck,
    "mbconv": MBConvBlock,
    "basicblock_dw": PreActBasicBlockDW,
    "edsr_block": EDSRBlock,
    "rcan_block": RCANBlock,
    "rcan_block_dw": RCANBlockDW,
    "rcan_block_all_dw": RCANBlockAllDW,
}


class ResidualGroup(nn.Module):
    def __init__(
        self,
        block_type: str,
        n_resblocks: int,
        planes: int,
        short_skip: bool = False,
        out_conv: bool = False,
        df_conv: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.short_skip = short_skip

        assert block_type in BLOCK_DICT
        blocks = [BLOCK_DICT[block_type](planes, **kwargs) for _ in range(n_resblocks)]
        if out_conv:
            # the final convolution for each residual block can be deformable
            blocks.append(conv3x3(planes, planes, bias=True, df_conv=df_conv))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        print("rg", res.min(), res.max(), res.mean(), res.shape)
        if self.short_skip:
            res += x
        return res


@ARCH_REGISTRY.register()
class RCANIT(nn.Module):
    def __init__(
        self,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        planes: int = 64,
        scale: int = 4,
        prob: list[float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        block_type: str = "rcan_block",
        channels: int = 3,
        rgb_range: int = 255,
        act_mode: str = "silu",
        out_conv: bool = True,
        short_skip: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.rgb_range = rgb_range
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        modules_head = [conv3x3(channels, planes, bias=True)]

        modules_body = [
            ResidualGroup(
                block_type,
                n_resblocks,
                planes,
                act_mode=act_mode,
                prob=prob[i],
                out_conv=out_conv,
                short_skip=short_skip,
                **kwargs,
            )
            for i in range(n_resgroups)
        ]
        modules_body.append(conv3x3(planes, planes, bias=True))

        modules_tail = [
            Upsampler(scale, planes, act_mode, use_affine=(block_type in AFFINE_LIST)),
            conv3x3(planes, channels, bias=True),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # print("x0", x.min(), x.max(), x.mean(), x.shape)
        x *= self.rgb_range
        x = self.sub_mean(x)
        # print("x1", x.min(), x.max(), x.mean(), x.shape)
        x = self.head(x)
        # print("x2", x.min(), x.max(), x.mean(), x.shape)

        res = self.body(x)
        print("res0", res.min(), res.max(), res.mean(), res.shape)
        res += x  # long skip-connection

        x = self.tail(res)
        x = self.add_mean(x)

        return x / self.rgb_range
