import random
from typing import overload

import cv2
import numpy as np
import pyvips
from torch import Tensor


def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[: h - h_remainder, : w - w_remainder, ...]
    else:
        raise ValueError(f"Wrong img ndim: {img.ndim}.")
    return img


@overload
def paired_random_crop(
    img_gt: np.ndarray,
    img_lq: np.ndarray,
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def paired_random_crop(
    img_gt: Tensor,
    img_lq: Tensor,
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[Tensor, Tensor]: ...


def paired_random_crop(
    img_gt: np.ndarray | Tensor,
    img_lq: np.ndarray | Tensor,
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[Tensor, Tensor]:
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if isinstance(img_gt, Tensor):
        assert isinstance(img_lq, Tensor)
        h_lq, w_lq = img_lq.size()[-2:]
        h_gt, w_gt = img_gt.size()[-2:]
    else:
        assert isinstance(img_lq, np.ndarray)
        h_lq, w_lq = img_lq.shape[0:2]
        h_gt, w_gt = img_gt.shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ",
            f"multiplication of LQ ({h_lq}, {w_lq}). {gt_path}",
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size}). "
            f"Please remove {gt_path}."
        )

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if isinstance(img_lq, Tensor):
        img_lq = img_lq[:, :, top : top + lq_patch_size, left : left + lq_patch_size]

    else:
        img_lq = img_lq[top : top + lq_patch_size, left : left + lq_patch_size, ...]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if isinstance(img_gt, Tensor):
        assert isinstance(img_lq, Tensor)
        img_gt = img_gt[
            :, :, top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size
        ]

        return img_gt, img_lq
    else:
        assert isinstance(img_lq, np.ndarray)
        img_gt = img_gt[
            top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size, ...
        ]

        return img_gt, img_lq


@overload
def paired_random_crop_list(
    img_gts: list[np.ndarray],
    img_lqs: list[np.ndarray],
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]: ...


@overload
def paired_random_crop_list(
    img_gts: list[Tensor],
    img_lqs: list[Tensor],
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[list[Tensor], list[Tensor]]: ...


def paired_random_crop_list(
    img_gts: list[np.ndarray] | list[Tensor],
    img_lqs: list[np.ndarray] | list[Tensor],
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[list[np.ndarray] | list[Tensor], list[np.ndarray] | list[Tensor]]:
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    assert len(img_gts) == len(img_lqs)

    cropped_gts, cropped_lqs = [], []
    for img_gt, img_lq in zip(img_gts, img_lqs, strict=False):
        if isinstance(img_gt, Tensor) and isinstance(img_lq, Tensor):
            cropped_gt, cropped_lq = paired_random_crop(
                img_gt, img_lq, gt_patch_size, scale, gt_path
            )
        elif isinstance(img_gt, np.ndarray) and isinstance(img_lq, np.ndarray):
            cropped_gt, cropped_lq = paired_random_crop(
                img_gt, img_lq, gt_patch_size, scale, gt_path
            )
        else:
            raise ValueError("img_gts and img_lqs must be all Tensor or all np.ndarray")
        cropped_gts.append(cropped_gt)
        cropped_lqs.append(cropped_lq)

    return cropped_gts, cropped_lqs


def paired_random_crop_vips(
    img_gt: pyvips.Image,
    img_lq: pyvips.Image,
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h_lq: int = img_lq.height  # pyright: ignore[reportAssignmentType]
    w_lq: int = img_lq.width  # pyright: ignore[reportAssignmentType]
    h_gt, w_gt = img_gt.height, img_gt.width
    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ",
            f"multiplication of LQ ({h_lq}, {w_lq}). {gt_path}",
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size}). "
            f"Please remove {gt_path}."
        )

    # randomly choose top and left coordinates for lq patch
    y = random.randint(0, h_lq - lq_patch_size)
    x = random.randint(0, w_lq - lq_patch_size)

    region_lq = pyvips.Region.new(img_lq)
    data_lq = region_lq.fetch(x, y, lq_patch_size, lq_patch_size)
    img_lq_np = np.ndarray(
        buffer=data_lq,
        dtype=np.uint8,
        shape=[lq_patch_size, lq_patch_size, img_lq.bands],  # pyright: ignore[reportAssignmentType,reportCallIssue,reportOptionalCall, reportArgumentType]
    )

    # crop corresponding gt patch
    x_gt, y_gt = int(x * scale), int(y * scale)
    region_gt = pyvips.Region.new(img_gt)
    data_gt = region_gt.fetch(x_gt, y_gt, gt_patch_size, gt_patch_size)
    img_gt_np = np.ndarray(
        buffer=data_gt,
        dtype=np.uint8,
        shape=[gt_patch_size, gt_patch_size, img_gt.bands],  # pyright: ignore[reportAssignmentType,reportCallIssue,reportOptionalCall, reportArgumentType]
    )

    return img_gt_np.astype(np.float32) / 255.0, img_lq_np.astype(np.float32) / 255.0


def augment(
    imgs: np.ndarray | list[np.ndarray],
    hflip: bool = True,
    rotation: bool = True,
    flows: list[np.ndarray] | None = None,
    return_status: bool = False,
) -> (
    np.ndarray
    | list[np.ndarray]
    | tuple[np.ndarray, np.ndarray]
    | tuple[list[np.ndarray], np.ndarray]
    | tuple[list[np.ndarray], list[np.ndarray]]
):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img: np.ndarray) -> np.ndarray:
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow: np.ndarray) -> np.ndarray:
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]  # type: ignore -- wtf is this function even? this needs to be rewritten to be less jank with what its returning
        return imgs, flows  # type: ignore -- ditto above
    elif return_status:
        return imgs, (hflip, vflip, rot90)  # type: ignore -- ditto above
    else:
        return imgs


def img_rotate(
    img: np.ndarray,
    angle: float,
    center: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
