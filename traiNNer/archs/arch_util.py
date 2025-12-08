# ruff: noqa
# type: ignore
import collections.abc
import math
from collections import OrderedDict
from itertools import repeat
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from spandrel.architectures.__arch_helpers.dysample import DySample as DySampleV1
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

from traiNNer.utils.download_util import load_file_from_url

# --------------------------------------------
# IQA utils
# --------------------------------------------


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA

    Args:
        dist_score (tensor): (*, C), C is the class number

    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


def random_crop(input_list, crop_size, crop_num):
    """
    Randomly crops the input tensor(s) to the specified size and number of crops.

    Args:
        input_list (list or tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crop. If an int is provided, a square crop of that size is used.
        If a tuple is provided, a crop of that size is used.
        crop_num (int): Number of crops to generate.

    Returns:
        tensor or list of tensors: If a single input tensor is provided, a tensor of cropped images is returned.
            If a list of input tensors is provided, a list of tensors of cropped images is returned.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]

    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)

    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [
            F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            for x in input_list
        ]
        b, c, h, w = input_list[0].shape

    crops_list = [[] for i in range(len(input_list))]
    for i in range(crop_num):
        sh = np.random.randint(0, h - ch + 1)
        sw = np.random.randint(0, w - cw + 1)
        for j in range(len(input_list)):
            crops_list[j].append(input_list[j][..., sh : sh + ch, sw : sw + cw])

    for i in range(len(crops_list)):
        crops_list[i] = torch.stack(crops_list[i], dim=1).reshape(
            b * crop_num, c, ch, cw
        )

    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


def uniform_crop(input_list, crop_size, crop_num):
    """
    Crop the input_list of tensors into multiple crops with uniform steps according to input size and crop_num.

    Args:
        input_list (list or torch.Tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crops. If int, the same size will be used for height and width.
            If tuple, should be (height, width).
        crop_num (int): Number of crops to generate.

    Returns:
        torch.Tensor or list of torch.Tensor: Cropped tensors. If input_list is a list, the output will be a list
            of cropped tensors. If input_list is a single tensor, the output will be a single tensor.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]

    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)

    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [
            F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            for x in input_list
        ]
        b, c, h, w = input_list[0].shape

    step_h = (h - ch) // int(np.sqrt(crop_num))
    step_w = (w - cw) // int(np.sqrt(crop_num))

    crops_list = []
    for _idx, inp in enumerate(input_list):
        tmp_list = []
        for i in range(int(np.ceil(np.sqrt(crop_num)))):
            for j in range(int(np.ceil(np.sqrt(crop_num)))):
                sh = i * step_h
                sw = j * step_w
                tmp_list.append(inp[..., sh : sh + ch, sw : sw + cw])
        crops_list.append(
            torch.stack(tmp_list[:crop_num], dim=1).reshape(b * crop_num, c, ch, cw)
        )

    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


# --------------------------------------------
# Common utils
# --------------------------------------------


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_pretrained_network(net, model_path, strict=True, weight_keys=None) -> None:
    if model_path.startswith(("https://", "http://")):
        model_path = load_file_from_url(model_path)
    state_dict = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )
    if weight_keys is not None:
        state_dict = state_dict[weight_keys]
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs) -> None:
    """
    Initializes the weights of the given module(s) using Kaiming Normal initialization.

    Args:
        module_list (list or nn.Module): List of modules or a single module to initialize.
        scale (float, optional): Scaling factor for the weights. Default is 1.
        bias_fill (float, optional): Value to fill the biases with. Default is 0.
        **kwargs: Additional arguments for the Kaiming Normal initialization.

    Returns:
        None

    Example:
        >>> import torch.nn as nn
        >>> from arch_util import default_init_weights
        >>> model = nn.Sequential(
        >>>     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        >>>     nn.ReLU(),
        >>>     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        >>>     nn.ReLU(),
        >>>     nn.Linear(64 * 32 * 32, 10)
        >>> )
        >>> default_init_weights(model, scale=0.1, bias_fill=0.01)
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
]


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle
        group: int = 4,  # Only DySample
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            m.append(DySampleV1(in_dim, out_dim, scale, group))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    1,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


SampleMods3 = Literal[SampleMods, "transpose+conv", "lda", "pa_up"]


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


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


class LDA_AQU(nn.Module):
    def __init__(
        self,
        in_channels=48,
        reduction_factor=4,
        nh=1,
        scale_factor=2.0,
        k_e=3,
        k_u=3,
        n_groups=2,
        range_factor=11,
        rpb=True,
    ) -> None:
        super().__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )
        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset, Hout, Wout):
        B, _, _, _ = offset.shape
        device = offset.device
        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(
            1, Hout, Wout, 2
        )
        offset = rearrange(
            offset, "b (kh kw d) h w -> b kh h kw w d", kh=self.k_u, kw=self.k_u
        )
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)

        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x, offset, ks=3):
        out = nn.functional.grid_sample(
            x,
            offset,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(
            q, (out_H, out_W), mode="bilinear", align_corners=True
        )
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(
            x.dtype
        )

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out


class PA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods3 = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,  # Only DySample
        dysample_end_kernel=1,  # needed only for compatibility with version 2
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )
            # m.append(nn.Conv2d(mid_dim, out_dim, dysample_end_kernel, 1, dysample_end_kernel//2)) # kernel 1 causes chromatic artifacts
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2, 3, 4"
                )
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods3.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )
