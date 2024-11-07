import math
import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pyvips
import torch
from cv2.typing import MatLike
from torch import Tensor
from torchvision.utils import make_grid


def img2tensor(
    img: np.ndarray,
    color: bool = True,
    bgr2rgb: bool = True,
    float32: bool = True,
) -> Tensor:
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    # def _totensor(img: np.ndarray, color: bool, bgr2rgb: bool, float32: bool) -> Tensor:
    if color:
        if img.ndim == 2:
            # Gray to RGB and to BGR are the same.
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(img.transpose(2, 0, 1))
    else:
        if img.ndim >= 3 and img.shape[2] == 3:
            if bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        out = torch.from_numpy(img[None, ...])

    if float32:
        out = out.float()
    return out

    # if isinstance(imgs, list):
    #     return [_totensor(img, color, bgr2rgb, float32) for img in imgs]
    # else:
    #     return _totensor(imgs, color, bgr2rgb, float32)


def imgs2tensors(
    imgs: list[np.ndarray],
    color: bool = True,
    bgr2rgb: bool = True,
    float32: bool = True,
) -> list[Tensor]:
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    return [img2tensor(img, color, bgr2rgb, float32) for img in imgs]


def tensors2imgs(
    tensors: list[Tensor],
    rgb2bgr: bool = True,
    out_type: np.dtype = np.uint8,  # type: ignore
    min_max: tuple[int, int] = (0, 1),
) -> list[np.ndarray]:
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    return [tensor2img(x, rgb2bgr, out_type, min_max) for x in tensors]


def tensor2img(
    tensor: Tensor,
    rgb2bgr: bool = True,
    out_type: np.dtype = np.uint8,  # type: ignore
    min_max: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """

    _tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = _tensor.dim()
    if n_dim == 4:
        img_np = make_grid(
            _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
        ).numpy()
        img_np = img_np.transpose(1, 2, 0)
        if rgb2bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 3:
        img_np = _tensor.numpy()
        img_np = img_np.transpose(1, 2, 0)
        if img_np.shape[2] == 1:  # gray image
            img_np = np.squeeze(img_np, axis=2)
        elif rgb2bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 2:
        img_np = _tensor.numpy()
    else:
        raise TypeError(
            f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}"
        )
    if out_type == np.uint8:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)

    return img_np


def tensor2img_fast(
    tensor: Tensor, rgb2bgr: bool = True, min_max: tuple[int, int] = (0, 1)
) -> np.ndarray:
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content: bytes, flag: str = "color", float32: bool = False) -> MatLike:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }

    img = cv2.imdecode(img_np, imread_flags[flag])

    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def vipsimfrompath(path: str) -> pyvips.Image:
    img = pyvips.Image.new_from_file(path, access="sequential", fail=True)
    assert isinstance(img, pyvips.Image)
    return img


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: Sequence[int] | None = None,
    auto_mkdir: bool = True,
) -> None:
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    if params:
        cv_save_image(file_path, img, params)
    else:
        cv_save_image(file_path, img, [])


# https://github.com/chaiNNer-org/chaiNNer/blob/171a31244064e0d90b9456b8ec0aebb80fe26511/backend/src/nodes/impl/image_utils.py#L330
def split_file_path(path: Path | str) -> tuple[Path, str, str]:
    """
    Returns the base directory, file name, and extension of the given file path.
    """
    base, ext = os.path.splitext(path)
    dirname, basename = os.path.split(base)
    return Path(dirname), basename, ext


# https://github.com/chaiNNer-org/chaiNNer/blob/171a31244064e0d90b9456b8ec0aebb80fe26511/backend/src/nodes/impl/image_utils.py#L330
def cv_save_image(path: Path | str, img: np.ndarray, params: Sequence[int]) -> None:
    """
    A light wrapper around `cv2.imwrite` to support non-ASCII paths.
    """

    # We can't actually use `cv2.imwrite`, because it:
    # 1. Doesn't support non-ASCII paths
    # 2. Silently fails without doing anything if the path is invalid

    _, _, extension = split_file_path(path)
    _, buf_img = cv2.imencode(f".{extension}", img, params)
    with open(path, "wb") as outf:
        outf.write(buf_img)  # type: ignore


def crop_border(
    imgs: np.ndarray | list[np.ndarray], crop_border: int
) -> np.ndarray | list[np.ndarray]:
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    elif isinstance(imgs, list):
        return [
            v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs
        ]
    else:
        return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
