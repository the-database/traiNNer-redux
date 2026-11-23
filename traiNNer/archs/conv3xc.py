"""Trainable reparameterization of the SPAN 1x1 -> 3x3 -> 1x1 block."""

from collections.abc import Iterable
from typing import Literal, Self, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class Conv3XC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        gain2: int = 0,
        s: int = 1,
        bias: Literal[True] = True,
        relu: bool = False,
        *,
        fused_training: bool = False,
    ) -> None:
        super().__init__()
        self.stride = s
        self.has_relu = relu
        self.fused_training = fused_training
        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain1, 1, bias=bias),
            nn.Conv2d(c_in * gain1, c_out * gain1, 3, stride=s, bias=bias),
            nn.Conv2d(c_out * gain1, c_out, 1, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=bias)
        self.eval_conv.requires_grad_(False)
        self.update_params()

    def equivalent_parameters(self) -> tuple[Tensor, Tensor]:
        first, spatial, last = cast(Iterable[nn.Conv2d], self.conv)
        # Keep the weight contractions in parameter precision under AMP. Autocast
        # still chooses the activation dtype for the much larger image convolution.
        with torch.autocast(device_type=self.sk.weight.device.type, enabled=False):
            first_w = first.weight.flatten(1)
            last_w = last.weight.flatten(1)
            projected = (last_w @ spatial.weight.flatten(1)).reshape(
                last_w.shape[0], first_w.shape[0], 9
            )
            kernel = (projected.transpose(1, 2) @ first_w).transpose(1, 2)
            kernel = kernel.reshape(last_w.shape[0], first_w.shape[1], 3, 3)
            kernel = kernel + F.pad(self.sk.weight, (1, 1, 1, 1))
            # In the branch version the image is padded BEFORE the first affine
            # map. Its bias therefore contributes at all nine taps, even at edges.
            assert self.sk.bias is not None
            assert (
                first.bias is not None
                and spatial.bias is not None
                and last.bias is not None
            )
            bias = projected.sum(2) @ first.bias + last_w @ spatial.bias
            bias = bias + last.bias + self.sk.bias
        return kernel, bias

    @torch.no_grad()
    def update_params(self) -> None:
        kernel, bias = self.equivalent_parameters()
        self.eval_conv.weight.copy_(kernel)
        assert self.eval_conv.bias is not None
        self.eval_conv.bias.copy_(bias)

    def _save_to_state_dict(
        self, destination: dict[str, Tensor], prefix: str, keep_vars: bool
    ) -> None:
        # Consumers may load only eval_conv. A checkpoint taken between
        # validations must contain the current deploy weights as well.
        if hasattr(self, "conv"):
            self.update_params()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode and hasattr(self, "conv"):
            self.update_params()
        return self

    def switch_to_deploy(self) -> None:
        if hasattr(self, "conv"):
            self.update_params()
            del self.conv
            del self.sk

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or not hasattr(self, "conv"):
            out = self.eval_conv(x)
        elif self.fused_training:
            kernel, bias = self.equivalent_parameters()
            out = F.conv2d(x, kernel, bias, stride=self.stride, padding=1)
        else:
            out = self.conv(F.pad(x, (1, 1, 1, 1))) + self.sk(x)
        return F.leaky_relu(out, negative_slope=0.05) if self.has_relu else out
