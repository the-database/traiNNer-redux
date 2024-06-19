import random
from typing import TypeVar

import cv2
import numpy as np
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


T = TypeVar("T", np.ndarray, list[np.ndarray], Tensor, list[Tensor])


def paired_random_crop(
    img_gts: T,
    img_lqs: T,
    gt_patch_size: int,
    scale: int,
    gt_path: str | None = None,
) -> tuple[T, T]:
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

    l_img_gts = []
    l_img_lqs = []
    if not isinstance(img_gts, list):
        l_img_gts = [img_gts]
    else:
        l_img_gts = list(img_gts)
    if not isinstance(img_lqs, list):
        l_img_lqs = [img_lqs]
    else:
        l_img_lqs = list(img_lqs)

    # determine input type: Numpy array or Tensor
    input_type = "Tensor" if isinstance(l_img_gts[0], Tensor) else "Numpy"

    if input_type == "Tensor":
        first_lq = l_img_lqs[0]
        first_gt = l_img_gts[0]
        assert isinstance(first_lq, Tensor)
        assert isinstance(first_gt, Tensor)
        h_lq, w_lq = first_lq.size()[-2:]
        h_gt, w_gt = first_gt.size()[-2:]
    else:
        h_lq, w_lq = l_img_lqs[0].shape[0:2]
        h_gt, w_gt = l_img_gts[0].shape[0:2]
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
    if input_type == "Tensor":
        l_img_lqs: list[np.ndarray | Tensor] = [
            v[:, :, top : top + lq_patch_size, left : left + lq_patch_size]
            for v in l_img_lqs
        ]
    else:
        l_img_lqs: list[np.ndarray | Tensor] = [
            v[top : top + lq_patch_size, left : left + lq_patch_size, ...]
            for v in l_img_lqs
        ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == "Tensor":
        l_img_gts: list[np.ndarray | Tensor] = [
            v[:, :, top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size]
            for v in l_img_gts
        ]
    else:
        l_img_gts: list[np.ndarray | Tensor] = [
            v[top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size, ...]
            for v in l_img_gts
        ]
    output_gts = None
    output_lqs = None
    if len(img_gts) == 1:
        first_out_gt = l_img_gts[0]
        assert isinstance(first_out_gt, np.ndarray | Tensor)
        output_gts = first_out_gt
    else:
        output_gts = l_img_gts
    if len(img_lqs) == 1:
        first_out_lq = l_img_lqs[0]
        assert isinstance(first_out_lq, np.ndarray | Tensor)
        output_lqs = first_out_lq
    else:
        output_lqs = l_img_lqs
    assert output_gts is not None
    assert output_lqs is not None
    return output_gts, output_lqs  # type: ignore


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
