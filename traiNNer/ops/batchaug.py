# Modified from: https://github.com/victorca25/traiNNer/blob/master/codes/dataops/batchaug.py

import os
import random
import sys
from os import path as osp

import numpy as np
import torch
import torchvision
from torch import Size, Tensor
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils import RNG
from traiNNer.utils.redux_options import TrainOptions

MOA_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/moa"))
)


class BatchAugment:
    def __init__(self, scale: int, train_opt: TrainOptions) -> None:
        self.moa_augs = train_opt.moa_augs
        self.moa_probs = train_opt.moa_probs
        self.scale = scale
        self.debug = train_opt.moa_debug
        self.debug_limit = train_opt.moa_debug_limit

    def __call__(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the configured augmentations.
        Args:
            img1: the target image.
            img2: the input image.
        """
        return batch_aug(
            img1,
            img2,
            self.scale,
            self.moa_augs,
            self.moa_probs,
            self.debug,
            self.debug_limit,
        )


def batch_aug(
    img_gt: Tensor,
    img_lq: Tensor,
    scale: int,
    augs: list[str],
    probs: list[float],
    debug: bool,
    debug_limit: int,
) -> tuple[Tensor, Tensor]:
    """Mixture of Batch Augmentations (MoA)
    Randomly selects single augmentation from the augmentation pool
    and applies it to the batch.
    Note: most of these augmentations require batch size > 1

    References:
    https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/aug_mixup.py
    https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
    https://github.com/clovaai/cutblur/blob/master/augments.py
    """

    i = 1

    if debug:
        os.makedirs(MOA_DEBUG_PATH, exist_ok=True)
        while os.path.exists(rf"{MOA_DEBUG_PATH}/{i:06d}_preauglq.png"):
            i += 1

        if i <= debug_limit or debug_limit == 0:
            torchvision.utils.save_image(
                img_lq, os.path.join(MOA_DEBUG_PATH, f"{i:06d}_preauglq.png"), padding=0
            )
            torchvision.utils.save_image(
                img_gt, os.path.join(MOA_DEBUG_PATH, f"{i:06d}_preauggt.png"), padding=0
            )

    if len(augs) != len(probs):
        msg = "Length of 'augmentation' and aug_prob don't match!"
        raise ValueError(msg)
    if img_gt.shape[0] == 1:
        msg = "Augmentations need batch >1 to work."
        raise ValueError(msg)

    idx = random.choices(range(len(augs)), weights=probs)[0]
    aug = augs[idx]

    if aug == "none":
        return img_gt, img_lq

    if "cutmix" == aug:
        img_gt, img_lq = cutmix(img_gt, img_lq, scale)
    elif "mixup" == aug:
        img_gt, img_lq = mixup(img_gt, img_lq, scale)
    elif "resizemix" == aug:
        img_gt, img_lq = resizemix(img_gt, img_lq, scale)
    elif "cutblur" == aug:
        img_gt, img_lq = cutblur(img_gt, img_lq, scale)
    elif "downup" == aug:
        img_gt, img_lq = downup(img_gt, img_lq)
    elif "up" == aug:
        img_gt, img_lq = up(img_gt, img_lq, scale)
    else:
        raise ValueError(f"{aug} is not invalid.")

    if debug:
        if i <= debug_limit:
            torchvision.utils.save_image(
                img_lq,
                os.path.join(MOA_DEBUG_PATH, f"{i:06d}_postaug_{aug}_lqfinal.png"),
                padding=0,
            )
            torchvision.utils.save_image(
                img_gt,
                os.path.join(MOA_DEBUG_PATH, f"{i:06d}_postaug_{aug}_gtfinal.png"),
                padding=0,
            )

    return img_gt, img_lq


@torch.no_grad()
def mixup(
    img_gt: Tensor,
    img_lq: Tensor,
    scale: int,
    alpha_min: float = 0.4,
    alpha_max: float = 0.6,
) -> tuple[Tensor, Tensor]:
    r"""MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha_min/max (float): The given min/max mixing ratio.
    """

    lam = RNG.get_rng().uniform(alpha_min, alpha_max)

    # mixup process
    rand_index = torch.randperm(img_gt.size(0))
    _img_gt = img_gt[rand_index]
    _img_lq = img_lq[rand_index]

    img_gt = lam * img_gt + (1 - lam) * _img_gt
    img_lq = lam * img_lq + (1 - lam) * _img_lq

    return img_gt, img_lq


@torch.no_grad()
def cutmix(
    img_gt: Tensor, img_lq: Tensor, scale: int, alpha: float = 0.9
) -> tuple[Tensor, Tensor]:
    r"""CutMix augmentation.

    "CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features (https://arxiv.org/abs/1905.04899)". In ICCV, 2019.
        https://github.com/clovaai/CutMix-PyTorch

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha (float): The given maximum mixing ratio.
    """

    if (
        img_gt.size()[3] != img_lq.size()[3] * scale
        or img_gt.size()[2] != img_lq.size()[2] * scale
    ):
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox(size: Size, scale: int, lam: float) -> tuple[int, int, int, int]:
        """generate random box by lam"""
        w = size[2] // scale
        h = size[3] // scale
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = RNG.get_rng().integers(w, dtype=int)
        cy = RNG.get_rng().integers(h, dtype=int)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        bb = (bbx1, bby1, bbx2, bby2)

        return tuple(bbi * scale for bbi in bb)

    lam = RNG.get_rng().uniform(0, alpha)
    rand_index = torch.randperm(img_gt.size(0))

    # mixup process
    img_gt_ = img_gt[rand_index]
    img_lq_ = img_lq[rand_index]

    # bbx1, bby1, bbx2, bby2 = rand_bbox(img_gt.size(), scale, lam)

    # random box
    gt_bbox = rand_bbox(img_gt.size(), scale, lam)
    gt_bbx1, gt_bby1, gt_bbx2, gt_bby2 = gt_bbox
    lq_bbox = tuple(bbi // 4 for bbi in gt_bbox)
    lq_bbx1, lq_bby1, lq_bbx2, lq_bby2 = lq_bbox

    img_gt[:, :, gt_bbx1:gt_bbx2, gt_bby1:gt_bby2] = img_gt_[
        :, :, gt_bbx1:gt_bbx2, gt_bby1:gt_bby2
    ]
    img_lq[:, :, lq_bbx1:lq_bbx2, lq_bby1:lq_bby2] = img_lq_[
        :, :, lq_bbx1:lq_bbx2, lq_bby1:lq_bby2
    ]

    return img_gt, img_lq


@torch.no_grad()
def resizemix(
    img_gt: Tensor, img_lq: Tensor, scale: int, scope: tuple[float, float] = (0.5, 0.9)
) -> tuple[Tensor, Tensor]:
    r"""ResizeMix augmentation.

    "ResizeMix: Mixing Data with Preserved Object Information and True Labels
    (https://arxiv.org/abs/2012.11101)".

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        scope (float): The given maximum mixing ratio.
    """

    if (
        img_gt.size()[3] != img_lq.size()[3] * scale
        or img_gt.size()[2] != img_lq.size()[2] * scale
    ):
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox_tao(size: Size, scale: int, tao: float) -> tuple[int, int, int, int]:
        """generate random box by tao (scale)"""
        w = size[2] // scale
        h = size[3] // scale
        cut_w = int(w * tao)
        cut_h = int(h * tao)

        # uniform
        cx = RNG.get_rng().integers(w, dtype=int)
        cy = RNG.get_rng().integers(h, dtype=int)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        bb = (bbx1, bby1, bbx2, bby2)

        return tuple(bbi * scale for bbi in bb)

    # index
    rand_index = torch.randperm(img_gt.size(0))
    img_gt_resize = img_gt.clone()
    img_gt_resize = img_gt_resize[rand_index]
    img_lq_resize = img_lq.clone()
    img_lq_resize = img_lq_resize[rand_index]

    # generate tao
    tao = RNG.get_rng().uniform(scope[0], scope[1])

    # random box
    # bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img_gt.size(), scale, tao)

    gt_bbox = rand_bbox_tao(img_gt.size(), scale, tao)
    gt_bbx1, gt_bby1, gt_bbx2, gt_bby2 = gt_bbox
    lq_bbox = tuple(bbi // 4 for bbi in gt_bbox)
    lq_bbx1, lq_bby1, lq_bbx2, lq_bby2 = lq_bbox

    # resize
    img_gt_resize = F.interpolate(
        img_gt_resize,
        (gt_bby2 - gt_bby1, gt_bbx2 - gt_bbx1),
        mode="bicubic",
        antialias=True,
    ).clamp(0, 1)
    img_lq_resize = F.interpolate(
        img_lq_resize,
        (lq_bby2 - lq_bby1, lq_bbx2 - lq_bbx1),
        mode="bicubic",
        antialias=True,
    ).clamp(0, 1)

    # mix
    img_gt[:, :, gt_bby1:gt_bby2, gt_bbx1:gt_bbx2] = img_gt_resize
    img_lq[:, :, lq_bby1:lq_bby2, lq_bbx1:lq_bbx2] = img_lq_resize

    return img_gt, img_lq


# TODO
# def cutmixup(img1, img2, mixup_prob=1.0, mixup_alpha=1.0,
#     cutmix_prob=1.0, cutmix_alpha=1.0):  # (alpha1 / alpha2) -> 0.7 / 1.2
#     """ CutMix with the Mixup-ed image.
#     CutMix and Mixup procedure use hyper-parameter alpha1 and alpha2 respectively.
#     """
#     c = _cutmix(img2, cutmix_prob, cutmix_alpha)
#     if c is None:
#         return img1, img2
#
#     scale = img1.size(2) // img2.size(2)
#     r_index, ch, cw = c["r_index"], c["ch"], c["cw"]
#     tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]
#
#     hch, hcw = ch * scale, cw * scale
#     hfcy, hfcx, htcy, htcx = (
#         fcy * scale, fcx * scale, tcy * scale, tcx * scale)
#
#     v = RNG.get_rng().beta(mixup_alpha, mixup_alpha)
#     if mixup_alpha <= 0 or random.random() >= mixup_prob:
#         img1_aug = img1[r_index, :]
#         img2_aug = img2[r_index, :]
#     else:
#         img1_aug = v * img1 + (1-v) * img1[r_index, :]
#         img2_aug = v * img2 + (1-v) * img2[r_index, :]
#
#     # apply mixup to inside or outside
#     if RNG.get_rng().random() > 0.5:
#         img1[..., htcy:htcy+hch, htcx:htcx+hcw] = img1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
#         img2[..., tcy:tcy+ch, tcx:tcx+cw] = img2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
#     else:
#         img1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = img1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
#         img2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = img2[..., fcy:fcy+ch, fcx:fcx+cw]
#         img2, img1 = img2_aug, img1_aug
#
#     return img1, img2


@torch.no_grad()
def cutblur(
    img_gt: Tensor, img_lq: Tensor, scale: int, alpha: float = 0.7
) -> tuple[Tensor, Tensor]:
    r"""CutBlur Augmentation.

    "Rethinking Data Augmentation for Image Super-resolution:
        A Comprehensive Analysis and a New Strategy"
        (https://arxiv.org/abs/2004.00448)

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha (float): The given max mixing ratio.
    """
    if (
        img_gt.size()[3] != img_lq.size()[3] * scale
        or img_gt.size()[2] != img_lq.size()[2] * scale
    ):
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox(size: Size, scale: int, lam: float) -> tuple[int, int, int, int]:
        """generate random box by lam (scale)"""
        w = size[2] // scale
        h = size[3] // scale
        cut_w = int(w * lam)
        cut_h = int(h * lam)

        # uniform
        cx = RNG.get_rng().integers(w, dtype=int)
        cy = RNG.get_rng().integers(h, dtype=int)

        bbx1 = np.clip(cx - cut_w // 2, 0, w) * scale
        bby1 = np.clip(cy - cut_h // 2, 0, h) * scale
        bbx2 = np.clip(cx + cut_w // 2, 0, w) * scale
        bby2 = np.clip(cy + cut_h // 2, 0, h) * scale

        return (bbx1, bby1, bbx2, bby2)

    lam = RNG.get_rng().uniform(0.2, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(img_gt.size(), scale, lam)

    # cutblur inside
    img_lq[:, :, bbx1 // scale : bbx2 // scale, bby1 // scale : bby2 // scale] = (
        F.interpolate(
            img_gt[:, :, bbx1:bbx2, bby1:bby2],
            scale_factor=1 / scale,
            mode="bicubic",
            antialias=True,
        )
    )

    return img_gt, img_lq


def downup(
    img_gt: Tensor, img_lq: Tensor, scope: tuple[float, float] = (0.5, 0.9)
) -> tuple[Tensor, Tensor]:
    sampling_opts = [("bicubic", True), ("bilinear", True), ("nearest-exact", False)]

    down_sample = random.choice(sampling_opts)
    up_sample = random.choice(sampling_opts)

    # don't allow nearest for both
    if down_sample[0] == sampling_opts[2][0] and up_sample[0] == sampling_opts[2][0]:
        if RNG.get_rng().random() > 0.5:
            # change up sample
            while up_sample[0] == sampling_opts[2][0]:
                up_sample = random.choice(sampling_opts)
        else:
            # change down sample
            while down_sample[0] == sampling_opts[2][0]:
                down_sample = random.choice(sampling_opts)

    scale_factor = RNG.get_rng().uniform(scope[0], scope[1])
    img_lq_base_size = img_lq.shape

    # downscale
    img_lq = F.interpolate(
        img_lq,
        size=(
            list(np.round(np.array(img_lq_base_size[2:]) * scale_factor).astype(int))
        ),
        mode=down_sample[0],
        antialias=down_sample[1],
    )

    # upscale back to original res
    img_lq = F.interpolate(
        img_lq, size=img_lq_base_size[2:], mode=up_sample[0], antialias=up_sample[1]
    )

    return img_gt, img_lq


def up(
    img_gt: Tensor, img_lq: Tensor, scale: int, scope: tuple[float, float] = (0.5, 0.9)
) -> tuple[Tensor, Tensor]:
    sampling_opts = [("bicubic", True), ("bilinear", True), ("nearest-exact", False)]

    def rand_bbox(size: Size, scale: int, lam: float) -> tuple[int, int, int, int]:
        """generate random box by lam (scale)"""
        w = size[2] // scale
        h = size[3] // scale
        cut_w = int(w * lam)
        cut_h = int(h * lam)

        pad_w = cut_w // 2
        pad_h = cut_h // 2

        # uniform
        cx = RNG.get_rng().integers(pad_w, w - pad_w, dtype=int)
        cy = RNG.get_rng().integers(pad_h, h - pad_w, dtype=int)

        bbx1 = (cx - pad_w) * scale
        bby1 = (cy - pad_h) * scale
        bbx2 = (cx + pad_w) * scale
        bby2 = (cy + pad_h) * scale

        return (bbx1, bby1, bbx2, bby2)

    img_gt_base_size = img_gt.shape
    img_lq_base_size = img_lq.shape

    lam = RNG.get_rng().uniform(scope[0], scope[1])

    # random box
    gt_bbox = rand_bbox(img_gt.size(), scale, lam)
    gt_bbx1, gt_bby1, gt_bbx2, gt_bby2 = gt_bbox
    lq_bbox = tuple(bbi // 4 for bbi in gt_bbox)
    lq_bbx1, lq_bby1, lq_bbx2, lq_bby2 = lq_bbox

    # crop to random box
    img_gt = img_gt[:, :, gt_bbx1:gt_bbx2, gt_bby1:gt_bby2]
    img_lq = img_lq[:, :, lq_bbx1:lq_bbx2, lq_bby1:lq_bby2]

    assert img_gt.shape[2] == img_gt.shape[3], (
        f"Expected crop to be square, got shape {img_gt}"
    )

    gt_up_sample = sampling_opts[0]  # bicubic
    lq_up_sample = random.choice(sampling_opts)

    # upscale cropped HQ to original size
    img_gt = F.interpolate(
        img_gt,
        size=img_gt_base_size[2:],
        mode=gt_up_sample[0],
        antialias=gt_up_sample[1],
    )

    # upscale cropped LQ to original size
    img_lq = F.interpolate(
        img_lq,
        size=img_lq_base_size[2:],
        mode=lq_up_sample[0],
        antialias=lq_up_sample[1],
    )

    return img_gt, img_lq
