# https://github.com/aaf6aa/SCUNet/blob/main/models/network_tscunet.py
# ruff: noqa
# type: ignore
import math
import re

import numpy as np
import torch
from torch import nn

from traiNNer.archs.scunet_aaf6aa_arch import ConvTransBlock, RRDBUpsample, Upconv
from traiNNer.utils.registry import ARCH_REGISTRY


class TSCUNetBlock(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        config=None,
        dim=64,
        drop_path_rate=0.0,
        input_resolution=256,
    ) -> None:
        if config is None:
            config = [2, 2, 2, 2, 2, 2, 2]
        super().__init__()

        self.head_dim = 32
        self.window_size = 8
        self.config = config
        self.dim = dim

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [
            ConvTransBlock(
                dim // 2,
                dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution,
            )
            for i in range(config[0])
        ] + [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [
            ConvTransBlock(
                dim,
                dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 2,
            )
            for i in range(config[1])
        ] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [
            ConvTransBlock(
                2 * dim,
                2 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 4,
            )
            for i in range(config[2])
        ] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [
            ConvTransBlock(
                4 * dim,
                4 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 8,
            )
            for i in range(config[3])
        ]

        begin += config[3]
        self.m_up3 = [
            Upconv(8 * dim, 4 * dim, 2, 2, bias=False),
        ] + [
            ConvTransBlock(
                2 * dim,
                2 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 4,
            )
            for i in range(config[4])
        ]

        begin += config[4]
        self.m_up2 = [
            Upconv(4 * dim, 2 * dim, 2, 2, bias=False),
        ] + [
            ConvTransBlock(
                dim,
                dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 2,
            )
            for i in range(config[5])
        ]

        begin += config[5]
        self.m_up1 = [
            Upconv(2 * dim, dim, 2, 2, bias=False),
        ] + [
            ConvTransBlock(
                dim // 2,
                dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution,
            )
            for i in range(config[6])
        ]

        self.m_res = [nn.Conv2d(dim, dim, 3, 1, 1, bias=False)]
        self.m_tail = [
            nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)

        self.m_res = nn.Sequential(*self.m_res)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):
        x1 = self.m_head(x0)

        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        x = x + self.m_res(x1)
        x = self.m_tail(x)

        return x


@ARCH_REGISTRY.register()
class TSCUNet(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        clip_size=5,
        nb=2,
        dim=64,
        drop_path_rate=0.0,
        input_resolution=256,
        scale=2,
        residual=True,
        sigma=False,
        state=None,
    ) -> None:
        super().__init__()

        if state:
            in_nc = state["m_head.0.weight"].shape[1]
            dim = state["m_head.0.weight"].shape[0]
            out_nc = state["m_tail.0.weight"].shape[0]

            clip_size = (
                len(
                    [
                        k
                        for k in state.keys()
                        if re.match(
                            re.compile(
                                r"m_layers\..\.m_body\.0\.trans_block\.mlp\.0\.weight"
                            ),
                            k,
                        )
                    ]
                )
                * 2
                + 1
            )
            nb = len(
                [
                    k
                    for k in state.keys()
                    if re.match(
                        re.compile(
                            r"m_layers\.0\.m_body\..\.trans_block\.mlp\.0\.weight"
                        ),
                        k,
                    )
                ]
            )

            scale = 2 ** max(
                0,
                len(
                    [
                        k
                        for k in state.keys()
                        if re.match(re.compile(r"m_upsample\.0\.up\.[0-9]+\.weight"), k)
                    ]
                )
                - 1,
            )
            input_resolution = 64 if scale > 1 else 256
            residual = "m_res.0.weight" in state.keys()
            sigma = "m_sigma.0.weight" in state.keys()

        if clip_size % 2 == 0:
            raise ValueError("TSCUNet clip_size must be odd")

        self.clip_size = clip_size
        self.dim = dim
        self.scale = scale
        self.residual = residual
        self.sigma = sigma

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        self.m_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        self.m_layers = [
            TSCUNetBlock(dim * 3, dim, [nb] * 7, dim, drop_path_rate, input_resolution)
            for _ in range((clip_size - 1) // 2)
        ]

        if self.residual:
            self.m_res = [nn.Conv2d(dim, dim, 3, 1, 1, bias=False)]
        self.m_upsample = [RRDBUpsample(dim, nb=2, scale=self.scale)]

        if self.sigma:
            # https://arxiv.org/abs/2201.10084
            # Revisiting L1 Loss in Super-Resolution: A Probabilistic View and Beyond
            self.m_sigma = []
            for _ in range(4):
                self.m_sigma += [nn.Conv2d(dim, dim, 3, 1, 1, bias=True), nn.PReLU()]
            self.m_sigma_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head).to(memory_format=torch.channels_last)
        self.m_tail = nn.Sequential(*self.m_tail).to(memory_format=torch.channels_last)
        self.m_layers = nn.ModuleList(self.m_layers).to(
            memory_format=torch.channels_last
        )

        if self.residual:
            self.m_res = nn.Sequential(*self.m_res)
        self.m_upsample = nn.Sequential(*self.m_upsample).to(
            memory_format=torch.channels_last
        )

        if self.sigma:
            self.m_sigma = nn.Sequential(*self.m_sigma).to(
                memory_format=torch.channels_last
            )
            self.m_sigma_tail = nn.Sequential(*self.m_sigma_tail).to(
                memory_format=torch.channels_last
            )

        if state:
            self.load_state_dict(state, strict=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if t != self.clip_size:
            raise ValueError(
                f"input clip size {t} does not match model clip size {self.clip_size}"
            )

        paddingH = int(np.ceil(h / 64) * 64 - h)
        paddingW = int(np.ceil(w / 64) * 64 - w)

        if not self.training:
            paddingH += 64
            paddingW += 64

        paddingLeft = math.ceil(paddingW / 2)
        paddingRight = math.floor(paddingW / 2)
        paddingTop = math.ceil(paddingH / 2)
        paddingBottom = math.floor(paddingH / 2)

        x = (
            self.m_head(
                nn.ReflectionPad2d(
                    (paddingLeft, paddingRight, paddingTop, paddingBottom)
                )(x.view(-1, c, h, w)).to(memory_format=torch.channels_last)
            )
            .to(memory_format=torch.contiguous_format)
            .view(b, -1, self.dim, h + paddingH, w + paddingW)
        )
        x1 = x

        for layer in self.m_layers:
            temp = [None] * (t - 2)

            for i in range(t - 2):
                temp[i] = layer(
                    x1[:, i : i + 3, ...]
                    .reshape(b, -1, h + paddingH, w + paddingW)
                    .to(memory_format=torch.channels_last)
                ).to(memory_format=torch.contiguous_format)

            x1 = torch.stack(temp, dim=1)
            t = x1.size(1)

        x1 = x1.squeeze(1).to(memory_format=torch.channels_last)

        if self.residual:
            x1 = x1 + self.m_res(
                x[:, self.clip_size // 2, ...].to(memory_format=torch.channels_last)
            )

        x1 = self.m_upsample(x1)

        if self.sigma and self.training:
            sigma = self.m_sigma(x1)
            sigma = self.m_sigma_tail(sigma + x1).to(
                memory_format=torch.contiguous_format
            )
            sigma = sigma[
                ...,
                paddingTop * self.scale : paddingTop * self.scale + h * self.scale,
                paddingLeft * self.scale : paddingLeft * self.scale + w * self.scale,
            ]

        x1 = self.m_tail(x1).to(memory_format=torch.contiguous_format)
        x1 = x1[
            ...,
            paddingTop * self.scale : paddingTop * self.scale + h * self.scale,
            paddingLeft * self.scale : paddingLeft * self.scale + w * self.scale,
        ]

        if self.sigma and self.training:
            return x1, sigma
        return x1
