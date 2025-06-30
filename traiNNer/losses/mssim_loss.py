r"""SSIM, MS-SSIM, CW-SSIM Metric

Created by:
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MS_SSIM.py
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/CW_SSIM.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    - Official SSIM matlab code from https://www.cns.nyu.edu/~lcv/ssim/;
    - PIQ from https://github.com/photosynthesis-team/piq;
    - BasicSR from https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py;
    - Official MS-SSIM matlab code from https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip;
    - Official CW-SSIM matlab code from
    https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/43017/versions/1/download/zip;

"""

import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.archs.arch_util import to_2tuple
from traiNNer.utils.registry import LOSS_REGISTRY


def rgb2yiq(x: Tensor) -> Tensor:
    r"""Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = (
        torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [0.5959, -0.2746, -0.3213],
                [0.2115, -0.5227, 0.3112],
            ]
        )
        .t()
        .to(x)
    )
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def rgb2lhm(x: Tensor) -> Tensor:
    r"""Convert a batch of RGB images to a batch of LHM images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.

    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = (
        torch.tensor([[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]])
        .t()
        .to(x)
    )
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm


def rgb2ycbcr(x: Tensor) -> Tensor:
    r"""Convert a batch of RGB images to a batch of YCbCr images

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). YCbCr color space.
    """
    weights_rgb_to_ycbcr = torch.tensor(
        [
            [65.481, -37.797, 112.0],
            [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214],
        ]
    ).to(x)
    bias_rgb_to_ycbcr = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(x)
    x_ycbcr = (
        torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_ycbcr).permute(0, 3, 1, 2)
        + bias_rgb_to_ycbcr
    )
    x_ycbcr = x_ycbcr / 255.0
    return x_ycbcr


def to_y_channel(
    img: Tensor, out_data_range: float = 1.0, color_space: str = "yiq"
) -> Tensor:
    r"""Change to Y channel
    Args:
        image tensor: tensor with shape (N, 3, H, W) in range [0, 1].
    Returns:
        image tensor: Y channel of the input tensor
    """
    assert img.ndim == 4 and img.shape[1] == 3, (
        "input image tensor should be RGB image batches with shape (N, 3, H, W)"
    )
    color_space = color_space.lower()
    if color_space == "yiq":
        img = rgb2yiq(img)
    elif color_space == "ycbcr":
        img = rgb2ycbcr(img)
    elif color_space == "lhm":
        img = rgb2lhm(img)
    out_img = img[:, [0], :, :] * out_data_range
    if out_data_range >= 255:
        # differentiable round with pytorch
        out_img = out_img - out_img.detach() + out_img.round()
    return out_img


def preprocess_rgb(
    x: Tensor, test_y_channel: bool, data_range: float = 1, color_space: str = "yiq"
) -> Tensor:
    """
    Preprocesses an RGB image tensor.

    Args:
        - x (torch.Tensor): The input RGB image tensor.
        - test_y_channel (bool): Whether to test the Y channel.
        - data_range (float): The data range of the input tensor. Default is 1.
        - color_space (str): The color space of the input tensor. Default is "yiq".

    Returns:
        torch.Tensor: The preprocessed RGB image tensor.
    """

    x = x.clamp(0, 1)

    if test_y_channel and x.shape[1] == 3:
        x = to_y_channel(x, data_range, color_space)
    else:
        x = x * data_range

    # use rounded uint8 value to make the input image same as MATLAB
    if data_range == 255:
        x = x - x.detach() + x.round()
    return x


def symm_pad(im: Tensor, padding: tuple[int, int, int, int]) -> Tensor:
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x: np.ndarray, minx: float, maxx: float) -> np.ndarray:
        """Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length"""
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def exact_padding_2d(
    x: Tensor,
    kernel: tuple | int,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    mode: str = "same",
) -> Tensor:
    assert len(x.shape) == 4, f"Only support 4D tensor input, but got {x.shape}"
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    _b, _c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (
        pad_col // 2,
        pad_col - pad_col // 2,
        pad_row // 2,
        pad_row - pad_row // 2,
    )

    mode = mode if mode != "same" else "constant"
    if mode != "symmetric":
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == "symmetric":
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))

    return x


class ExactPadding2d(nn.Module):
    r"""This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.

    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')

    """

    def __init__(
        self,
        kernel: tuple,
        stride: int = 1,
        dilation: int = 1,
        mode: str | None = "same",
    ) -> None:
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if self.mode is None:
            return x
        else:
            return exact_padding_2d(
                x, self.kernel, self.stride, self.dilation, self.mode
            )


def imfilter(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: str = "same",
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """imfilter same as matlab.
    Args:
        input (tensor): (b, c, h, w) tensor to be filtered
        weight (tensor): (out_ch, in_ch, kh, kw) filter kernel
        padding (str): padding mode
        dilation (int): dilation of conv
        groups (int): groups of conv
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)

    return F.conv2d(
        pad_func(input), weight, bias, stride, dilation=dilation, groups=groups
    )


def filter2(input: Tensor, weight: Tensor, shape: str = "same") -> Tensor:
    if shape == "same":
        return imfilter(input, weight, groups=input.shape[1])
    elif shape == "valid":
        return F.conv2d(
            input,
            weight,
            stride=1,
            padding=0,
            groups=input.shape[1],
        )
    else:
        raise NotImplementedError(f"Shape type {shape} is not implemented.")


def fspecial(
    size: int | tuple,
    sigma: float,
    channels: int = 1,
    filter_type: str = "gaussian",
) -> Tensor:
    r"""Function same as 'fspecial' in MATLAB, only support gaussian now.
    Args:
        size (int or tuple): size of window
        sigma (float): sigma of gaussian
        channels (int): channels of output
    """
    if filter_type == "gaussian":
        shape = to_2tuple(size)
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
        return h
    else:
        raise NotImplementedError(
            f"Only support gaussian filter now, got {filter_type}"
        )


def ssim(
    x: Tensor,
    y: Tensor,
    win: Tensor | None = None,
    get_ssim_map: bool = False,
    get_cs: bool = False,
    get_weight: bool = False,
    downsample: bool = False,
    data_range: float = 1.0,
    include_luminance: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    if win is None:
        win = fspecial(11, 1.5, x.shape[1]).to(x)

    filter_shape = "valid"
    if x.shape[-1] < win.shape[-1]:
        filter_shape = "same"

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    # Downsample operation is used in official matlab code
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    mu1 = filter2(x, win, filter_shape)
    mu2 = filter2(y, win, filter_shape)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(x * x, win, filter_shape) - mu1_sq
    sigma2_sq = filter2(y * y, win, filter_shape) - mu2_sq
    sigma12 = filter2(x * y, win, filter_shape) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    cs_map = F.relu(
        cs_map
    )  # force the ssim response to be nonnegative to avoid negative results.
    if include_luminance:
        l = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    else:
        l = torch.full_like(cs_map, 1.0)
    ssim_map = l * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / c2) * (1 + sigma2_sq / c2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val


@LOSS_REGISTRY.register()
class SSIMLoss(torch.nn.Module):
    r"""Args:
    - channel: number of channel.
    - downsample: boolean, whether to downsample same as official matlab code.
    - test_y_channel: boolean, whether to use y channel on ycbcr same as official matlab code.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        channels: int = 3,
        downsample: bool = False,
        test_y_channel: bool = True,
        color_space: str = "yiq",
        crop_border: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.crop_border = crop_border
        self.data_range = 1.0
        self.loss_weight = loss_weight

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape, (
            f"Input {x.shape} and reference images should have the same shape"
        )

        if self.crop_border != 0:
            crop_border = self.crop_border
            x = x[..., crop_border:-crop_border, crop_border:-crop_border]
            y = y[..., crop_border:-crop_border, crop_border:-crop_border]

        x = preprocess_rgb(x, self.test_y_channel, self.data_range, self.color_space)
        y = preprocess_rgb(y, self.test_y_channel, self.data_range, self.color_space)

        score = ssim(x, y, data_range=self.data_range, downsample=self.downsample)
        assert isinstance(score, Tensor)
        return score


def ms_ssim(
    x: Tensor,
    y: Tensor,
    win: Tensor | None = None,
    data_range: float = 1.0,
    downsample: bool = False,
    test_y_channel: bool = True,
    is_prod: bool = True,
    color_space: str = "yiq",
    include_luminance: bool = False,
) -> Tensor:
    r"""Compute Multiscale structural similarity for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        win: Window setting.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr.
        is_prod: Boolean, calculate product or sum between mcs and weight.
    Returns:
        Index of similarity between two images. Usually in [0, 1] interval.
    """
    if not x.shape == y.shape:
        raise ValueError("Input images must have the same dimensions.")

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(x)

    levels = weights.shape[0]
    mcs = []

    for _ in range(levels):
        ssim_val, cs = ssim(
            x,
            y,
            win=win,
            get_cs=True,
            downsample=downsample,
            data_range=data_range,
            include_luminance=include_luminance,
        )

        mcs.append(cs)
        padding = (x.shape[2] % 2, x.shape[3] % 2)
        x = F.avg_pool2d(x, kernel_size=2, padding=padding)
        y = F.avg_pool2d(y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)

    if is_prod:
        msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)), dim=0) * (
            ssim_val ** weights[-1]  # pyright: ignore[reportPossiblyUnboundVariable]
        )
    else:
        weights = weights / torch.sum(weights)
        msssim_val = torch.sum((mcs[:-1] * weights[:-1].unsqueeze(1)), dim=0) + (
            ssim_val * weights[-1]  # pyright: ignore[reportPossiblyUnboundVariable]
        )

    return msssim_val


@LOSS_REGISTRY.register()
class MSSIMLoss(torch.nn.Module):
    r"""Multiscale structure similarity

    References:
        Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural similarity for image
        quality assessment." In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers,
        2003, vol. 2, pp. 1398-1402. Ieee, 2003.

    Args:
        channel: Number of channel.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.
    """

    def __init__(
        self,
        loss_weight: float,
        channels: int = 3,
        downsample: bool = False,
        test_y_channel: bool = True,
        is_prod: bool = True,
        color_space: str = "yiq",
        include_luminance: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.is_prod = is_prod
        self.data_range = 1
        self.include_luminance = include_luminance

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computation of MS-SSIM metric.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MS-SSIM metric in [0, 1] range.
        """
        assert x.shape == y.shape, (
            "Input and reference images should have the same shape, but got"
        )
        f"{x.shape} and {y.shape}"

        x = preprocess_rgb(x, self.test_y_channel, self.data_range, self.color_space)
        y = preprocess_rgb(y, self.test_y_channel, self.data_range, self.color_space)

        score = ms_ssim(
            x,
            y,
            data_range=self.data_range,
            downsample=self.downsample,
            is_prod=self.is_prod,
            include_luminance=self.include_luminance,
        )
        return 1 - score.mean().clamp(0, 1)
