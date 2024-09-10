# https://github.com/aaf6aa/SCUNet/blob/main/models/network_scunet.py
# ruff: noqa
# type: ignore
import math
import re

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, trunc_normal_
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY


class WMSA(nn.Module):
    """Self-attention module in Swin Transformer"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1, 2 * window_size - 1, self.n_heads
            )
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        """generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(
            h,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        """Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        # Using Attn Mask to distinguish different subwindows.
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                ]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
    ) -> None:
        """SwinTransformer Block"""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.sigma != 0 and self.training:
            noise = torch.randn_like(x, device=x.device, dtype=x.dtype) * self.sigma
            x = ((noise + x).detach() - x).detach() + x
        return x


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        conv_dim,
        trans_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
        noise=True,
    ) -> None:
        """SwinTransformer and Conv Block"""
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ["W", "SW"]
        if self.input_resolution <= self.window_size:
            self.type = "W"

        self.trans_block = Block(
            self.trans_dim,
            self.trans_dim,
            self.head_dim,
            self.window_size,
            self.drop_path,
            self.type,
            self.input_resolution,
        )
        self.conv1_1 = nn.Conv2d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv1_2 = nn.Conv2d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
        )

        self.noise = nn.Identity()
        if noise:
            self.noise = GaussianNoise(0.05)

    def forward(self, x):
        conv_x, trans_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1
        )
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = self.noise(x + res)

        return x


class Upconv(nn.Module):
    def __init__(self, dim, out_dim, scale=2, a=0, b=0, bias=False, blur=False) -> None:
        super().__init__()
        self.scale = scale

        self.up = []
        for _ in range(int(math.log2(self.scale))):
            self.up += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
            ]
            if blur:
                self.up += [
                    nn.ReplicationPad2d((1, 0, 1, 0)),
                    nn.AvgPool2d(2, stride=1),
                ]
        self.up += [nn.Conv2d(dim, out_dim, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*self.up)

    def forward(self, x):
        x = self.up(x)
        return x


class ResidualDenseBlock_(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB_(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock_(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBUpsample(nn.Module):
    def __init__(self, dim, nb=2, scale=2, blur=False) -> None:
        super().__init__()
        self.scale = scale

        self.up = [RRDB_(dim, 32) for _ in range(nb)]
        for _ in range(int(math.log2(self.scale))):
            self.up += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
            ]
            if blur:
                self.up += [
                    nn.ReplicationPad2d((1, 0, 1, 0)),
                    nn.AvgPool2d(2, stride=1),
                ]
        self.up += [nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*self.up)

    def forward(self, x):
        x = self.up(x)
        return x


@ARCH_REGISTRY.register()
class SCUNet_aaf6aa(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        config=None,
        dim=64,
        drop_path_rate=0.0,
        input_resolution=256,
        scale=1,
        residual=True,
        state=None,
    ) -> None:
        if config is None:
            config = [2, 2, 2, 2, 2, 2, 2]
        super().__init__()

        if state:
            in_nc = state["m_head.0.weight"].shape[1]
            dim = state["m_head.0.weight"].shape[0]
            out_nc = state["m_tail.0.weight"].shape[0]

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

            # TODO: obtain this parameter without assuming
            input_resolution = 64 if scale > 1 else 256
            residual = "m_res.0.weight" in state.keys()

            config = []
            for i in range(1, 4):
                config.append(
                    len(
                        [
                            k
                            for k in state.keys()
                            if re.match(
                                re.compile(
                                    rf"m_down{i}\..\.trans_block\.mlp\.0\.weight"
                                ),
                                k,
                            )
                        ]
                    )
                )
            config.append(
                len(
                    [
                        k
                        for k in state.keys()
                        if re.match(
                            re.compile(r"m_body\..\.trans_block\.mlp\.0\.weight"), k
                        )
                    ]
                )
            )
            for i in range(1, 4):
                config.append(
                    len(
                        [
                            k
                            for k in state.keys()
                            if re.match(
                                re.compile(rf"m_up{i}\..\.trans_block\.mlp\.0\.weight"),
                                k,
                            )
                        ]
                    )
                )
            config.append(
                len(
                    [
                        k
                        for k in state.keys()
                        if re.match(
                            re.compile(
                                r"m_upsample\.0\.up\.[0-9]+\.rdb1\.conv1\.weight"
                            ),
                            k,
                        )
                    ]
                )
            )

        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        self.scale = scale
        self.residual = residual

        unet_up = Upconv if scale > 1 else nn.ConvTranspose2d

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
            unet_up(8 * dim, 4 * dim, 2, 2, bias=False),
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
            unet_up(4 * dim, 2 * dim, 2, 2, bias=False),
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
            unet_up(2 * dim, dim, 2, 2, bias=False),
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

        if self.residual:
            self.m_res = [nn.Conv2d(dim, dim, 3, 1, 1, bias=False)]
        if self.scale > 1:
            self.m_upsample = [RRDBUpsample(dim, nb=2, scale=self.scale)]

        self.m_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)

        if self.residual:
            self.m_res = nn.Sequential(*self.m_res)
        if scale > 1:
            self.m_upsample = nn.Sequential(*self.m_upsample)

        self.m_tail = nn.Sequential(*self.m_tail)

        if state:
            self.load_state_dict(state, strict=True)
        # self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.size()[-2:]

        paddingH = int(np.ceil(h / 64) * 64 - h)
        paddingW = int(np.ceil(w / 64) * 64 - w)
        if not self.training:
            paddingH += 64
            paddingW += 64

        paddingLeft = math.ceil(paddingW / 2)
        paddingRight = math.floor(paddingW / 2)
        paddingTop = math.ceil(paddingH / 2)
        paddingBottom = math.floor(paddingH / 2)

        x0 = nn.ReflectionPad2d((paddingLeft, paddingRight, paddingTop, paddingBottom))(
            x0
        )

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        if self.residual:
            x1 = self.m_res(x1)

        x = x + x1
        if self.scale > 1:
            x = self.m_upsample(x)
        x = self.m_tail(x)

        x = x[
            ...,
            paddingTop * self.scale : paddingTop * self.scale + h * self.scale,
            paddingLeft * self.scale : paddingLeft * self.scale + w * self.scale,
        ]

        return x

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
