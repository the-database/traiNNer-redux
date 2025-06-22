from collections.abc import Sequence
from typing import Any, Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from traiNNer.archs.arch_util import SampleMods, UniUpsample
from traiNNer.utils.registry import ARCH_REGISTRY


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain1: int = 1, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        nn.init.trunc_normal_(self.sk.weight, std=0.02)
        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # pyright: ignore[reportOptionalMemberAccess]
            self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]

        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b  # pyright: ignore[reportOperatorIssue,reportPossiblyUnboundVariable]
            self.eval_conv.bias.data = self.bias_concat  # pyright: ignore[reportOptionalMemberAccess]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out


class SPAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_dim: int | None = None,
        out_dim: int | None = None,
        bias: bool = False,
        end: bool = False,
    ) -> None:
        super().__init__()
        mid_dim = mid_dim if mid_dim else in_channels
        out_dim = out_dim if out_dim else in_channels
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_dim, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_dim, mid_dim, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_dim, out_dim, gain1=2, s=1, bias=bias)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.end = end

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, flash: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.register_buffer("flash", torch.tensor([flash]))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, 1, dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Преобразуем в (b, num_heads, head_dim, h*w)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)
        if self.flash.item():  # pyright: ignore[reportCallIssue]
            out = F.scaled_dot_product_attention(
                q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3), is_causal=False
            )
            out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        else:
            q = F.normalize(q, dim=3)
            k = F.normalize(k, dim=3)

            attn = (
                torch.matmul(q, k.transpose(2, 3)) * self.temperature
            )  # (b, num_heads, hw, hw)
            attn = attn.softmax(dim=3)

            out = torch.matmul(attn, v)  # (b, num_heads, head_dim, hw)

            # Обратно в (b, c, h, w)
            out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1,
        att: bool = False,
        flash: bool = True,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.token_mix = (
            Attention(conv_channels, 16, flash) if att else InceptionDWConv2d(dim)
        )
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1, 0)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.token_mix(c)
        x = self.act(g) * torch.cat((i, c), dim=1)
        x = self.act(self.fc2(x))
        return x


class SimpleGate(nn.Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MetaGated(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim * 2
        self.local = nn.Sequential(
            RMSNorm(dim),
            nn.Conv2d(dim, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=dim),
            SimpleGate(),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
        )
        self.glob = GatedCNNBlock(dim)
        self.gamma0 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.gamma0.register_hook(lambda grad: grad * 10)
        self.gamma1.register_hook(lambda grad: grad * 10)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        short = x
        x = self.local(x)
        x = x * self.sca(x)
        x = x * self.gamma0 + short
        del short
        x = self.glob(x) * self.gamma1 + x
        return x


class Down(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2)
        )


class Upsample(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2)
        )


class Block(nn.Module):
    def __init__(self, dim: int, num_gated: int, down: bool = True) -> None:
        super().__init__()

        if down:
            self.gated = nn.Sequential(*[MetaGated(dim) for _ in range(num_gated)])
            self.scale = Down(dim)

        else:
            self.scale = Upsample(dim)
            self.gated = nn.Sequential(*[MetaGated(dim // 2) for _ in range(num_gated)])
            self.shor = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.down = down

    def forward(
        self, x: Tensor, short: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.down:
            x = self.gated(x)
            return self.scale(x), x
        else:
            x = torch.cat([self.scale(x), short], dim=1)  # pyright: ignore[reportCallIssue,reportArgumentType]
            x = self.shor(x)
            return self.gated(x)


@ARCH_REGISTRY.register()
class GateRV3(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 32,
        enc_blocks: Sequence[int] = (2, 2, 4, 6),
        dec_blocks: Sequence[int] = (2, 2, 2, 2),
        num_latent: int = 8,
        scale: int = 2,
        upsample: SampleMods = "pixelshuffle",
        upsample_mid_dim: int = 48,
        end_gamma_init: int = 1,
        attention: bool = False,
        sisr_blocks: int = 4,
        flash: bool = True,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.gater_encode = nn.ModuleList(
            [Block(dim * (2**i), enc_blocks[i]) for i in range(len(enc_blocks))]
        )
        self.span_block0 = SPAB(dim, end=False)
        self.span_n_b = nn.Sequential(
            *[SPAB(dim, end=False) for _ in range(sisr_blocks)]
        )
        self.span_end = SPAB(dim, end=True)
        self.sisr_end_conv = Conv3XC(dim, dim, bias=True)
        self.sisr_cat_conv = nn.Conv2d(dim * 4, dim, 1)
        nn.init.trunc_normal_(self.sisr_cat_conv.weight, std=0.02)
        self.latent = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim * (2 ** len(enc_blocks)),
                    expansion_ratio=1.5,
                    conv_ratio=1.00,
                    att=attention,
                    flash=flash,
                )
                for _ in range(num_latent)
            ]
        )
        self.decode = nn.ModuleList(
            [
                Block(
                    dim * (2 ** (len(dec_blocks) - i)),
                    dec_blocks[i],
                    False,
                )
                for i in range(len(dec_blocks))
            ]
        )
        self.pad = 2 ** (len(enc_blocks))

        self.gamma = nn.Parameter(torch.ones(1, in_ch, 1, 1) * end_gamma_init)
        self.gamma.register_hook(lambda grad: grad * 10)

        if scale != 1:
            self.short_to_dim = nn.Upsample(scale_factor=scale)  # ConvBlock(in_ch, dim)
            self.dim_to_in = UniUpsample(upsample, scale, dim, in_ch, upsample_mid_dim)
        else:
            self.dim_to_in = nn.Conv2d(dim, in_ch, 3, 1, 1)
            self.short_to_dim = nn.Identity()

    def load_state_dict(
        self, state_dict: dict[str, Any], *args: Any, **kwargs
    ) -> _IncompatibleKeys:
        if "dim_to_in.MetaUpsample" in state_dict:
            state_dict["dim_to_in.MetaUpsample"] = self.dim_to_in.MetaUpsample
        if "gamma" not in state_dict:
            state_dict["gamma"] = self.gamma
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, inp: Tensor) -> Tensor:
        _b, _c, h, w = inp.shape
        inp = self.check_img_size(inp, (h, w))
        x = self.in_to_dim(inp)
        sisr = self.span_block0(x)
        sisr_short = sisr
        sisr = self.span_n_b(sisr)
        sisr, sisr_out = self.span_end(sisr)
        sisr = self.sisr_end_conv(sisr)
        sisr = self.sisr_cat_conv(torch.cat([x, sisr, sisr_short, sisr_out], dim=1))
        del sisr_short, sisr_out
        shorts = []

        for _index, block in enumerate(self.gater_encode):
            x, short = block(x)
            shorts.append(short)

        x = self.latent(x)
        len_block = len(self.decode)
        shorts.reverse()
        for index in range(len_block):
            x = self.decode[index](x, shorts[index])

        x = self.dim_to_in(x + sisr) + self.gamma * self.short_to_dim(inp)
        return x[:, :, : h * self.scale, : w * self.scale]


@ARCH_REGISTRY.register()
def gaterv3_s(**kwargs) -> GateRV3:
    return GateRV3(enc_blocks=(2, 2, 4), dec_blocks=(2, 2, 2), dim=32, **kwargs)


@ARCH_REGISTRY.register()
def gaterv3_r(**kwargs) -> GateRV3:
    return GateRV3(dim=32, **kwargs)
