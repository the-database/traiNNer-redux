import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.utils.registry import TESTARCH_REGISTRY


class FeedForward(nn.Module):
    def __init__(
        self, dim: int = 64, ffn_expansion_factor: float = 2.66, bias: bool = True
    ) -> None:
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DynamicLocal(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * kernel_size * kernel_size, 1),
        )
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        # генерируем веса для каждого примера
        kernels = self.kernel_gen(x)  # [B, C*k*k, 1, 1]
        kernels = kernels.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        # разворачиваем вход для групповой свёртки
        x_ = x.reshape(1, B * C, H, W)
        out = F.conv2d(x_, kernels, padding=self.padding, groups=B * C)
        out = out.reshape(B, C, H, W)
        return out


class FSAS(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = False,
    ) -> None:
        super().__init__()
        mid = int(dim * 3 * mid_factor)
        self.to_hidden = nn.Conv2d(dim, mid, kernel_size=1, bias=True)
        self.to_hidden_dw = nn.Conv2d(
            mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=True
        )

        self.project_out = nn.Conv2d(
            int(dim * mid_factor), dim, kernel_size=1, bias=True
        )
        self.patch = windowed
        self.norm = LayerNorm(int(dim * mid_factor))

        self.patch_size = window_size

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        if self.patch:
            B, C, H, W = q.shape

            q_patch = q.view(
                B,
                C,
                H // self.patch_size,
                self.patch_size,
                W // self.patch_size,
                self.patch_size,
            ).permute(0, 1, 2, 4, 3, 5)  # -> [B, C, H//p, W//p, p, p]
            k_patch = k.view(
                B,
                C,
                H // self.patch_size,
                self.patch_size,
                W // self.patch_size,
                self.patch_size,
            ).permute(0, 1, 2, 4, 3, 5)

            q_fft = torch.fft.rfft2(q_patch.float())
            k_fft = torch.fft.rfft2(k_patch.float())

            out = q_fft * k_fft
            out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
            out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        else:
            q_fft = torch.fft.rfft2(q.float())
            k_fft = torch.fft.rfft2(k.float())

            out = q_fft * k_fft
            out = torch.fft.irfft2(out)
        out = self.norm(out)

        output = v * out

        output = self.project_out(output)

        return output


class SFSAS(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = True,
    ) -> None:
        super().__init__()
        local = int(split * dim)
        global_dim = dim - local
        self.local = nn.Sequential(DynamicLocal(local), DynamicLocal(local, 5))
        self.att = FSAS(global_dim, t_mid_factor, window_size, windowed)
        self.last = nn.Conv2d(dim, dim, 1)
        self.split = [local, global_dim]

    def forward(self, x):
        x1, x2 = x.split(self.split, dim=1)
        x1 = self.local(x1)
        x2 = self.att(x2)
        return self.last(torch.cat([x1, x2], dim=1))


class MetaBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = True,
        mlp: float = 2.66,
    ) -> None:
        super().__init__()
        self.token_mix = nn.Sequential(
            LayerNorm(dim), SFSAS(dim, split, t_mid_factor, window_size, windowed)
        )
        self.channel_mix1 = nn.Sequential(LayerNorm(dim), FeedForward(dim, mlp, True))

    def forward(self, x):
        x = self.token_mix(x) + x
        x = self.channel_mix1(x) + x
        return x


class ResidualMeta(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        meta_b: int = 6,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        mlp: float = 2.66,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            *[
                MetaBlock(dim, split, t_mid_factor, window_size, bool(i % 2), mlp)
                for i in range(meta_b)
            ]
            + [DynamicLocal(dim)]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.residual(x) + x


@TESTARCH_REGISTRY.register()
class LAWFFT(nn.Module):
    """Local Adaptive Weighted Fourier Feature Transformer"""

    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 60,
        split: float = 0.25,
        scale: int = 4,
        n_rblock: int = 4,
        n_mblock: int = 6,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        mlp_factor: float = 2.66,
        unshuffle_mod: bool = False,
        upsampler: SampleMods3 = "pixelshuffle",
        mid_dim: int = 64,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.pad = 1
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            self.in_to_dim = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            )
            self.pad = unshuffle * window_size
            scale = 4
        else:
            self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
            self.pad = window_size
        self.register_buffer(
            "window_size", torch.tensor(window_size, dtype=torch.uint8)
        )
        self.body = nn.Sequential(
            *[
                ResidualMeta(
                    dim, n_mblock, split, t_mid_factor, window_size, mlp_factor
                )
                for _ in range(n_rblock)
            ]
        )
        self.upscale = UniUpsampleV3(upsampler, scale, dim, in_ch, mid_dim)

        self.scale = scale

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict["upscale.MetaUpsample"] = self.upscale.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, inp):
        b, c, h, w = inp.shape
        x = self.check_img_size(inp, (h, w))
        x = self.in_to_dim(x)
        x = self.body(x) + x
        x = self.upscale(x)[:, :, : h * self.scale, : w * self.scale]
        return x
