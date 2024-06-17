# Code from: https://github.com/victorca25/traiNNer/blob/master/codes/dataops/batchaug.py

import os
import random

import numpy as np
import torch
import torchvision
from torch.nn import functional as F

rng = np.random.default_rng()


class BatchAugment:
    def __init__(self, train_opt):
        self.moa_augs = train_opt.get(
            "moa_augs", ["none", "mixup", "cutmix", "resizemix"]
        )  # , "cutblur"]
        self.moa_probs = train_opt.get(
            "moa_probs", [0.4, 0.084, 0.084, 0.084, 0.348]
        )  # , 1.0]
        self.scale = train_opt.get("scale", 4)
        self.debug = train_opt.get("moa_debug", False)
        self.debug_limit = train_opt.get("moa_debug_limit", 0)

    def __call__(self, img1, img2):
        """Apply the configured augmentations.
        Args:
            img1: the target image.
            img2: the input image.
        """
        return BatchAug(
            img1,
            img2,
            self.scale,
            self.moa_augs,
            self.moa_probs,
            self.debug,
            self.debug_limit,
        )


def BatchAug(img_gt, img_lq, scale, augs, probs, debug, debug_limit):
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

    if debug:
        i = 1
        moa_debug_path = "./moa_debug"
        os.makedirs(moa_debug_path, exist_ok=True)
        while os.path.exists(rf"./moa_debug/{i:06d}_preauglq.png"):
            i += 1

        if i <= debug_limit or debug_limit == 0:
            torchvision.utils.save_image(
                img_lq, os.path.join(moa_debug_path, f"{i:06d}_preauglq.png"), padding=0
            )
            torchvision.utils.save_image(
                img_gt, os.path.join(moa_debug_path, f"{i:06d}_preauggt.png"), padding=0
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
                os.path.join(moa_debug_path, f"{i:06d}_postaug_{aug}_lqfinal.png"),
                padding=0,
            )
            torchvision.utils.save_image(
                img_gt,
                os.path.join(moa_debug_path, f"{i:06d}_postaug_{aug}_gtfinal.png"),
                padding=0,
            )

    return img_gt, img_lq


@torch.no_grad()
def mixup(img_gt, img_lq, scale, alpha_min=0.4, alpha_max=0.6):
    r"""MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha_min/max (float): The given min/max mixing ratio.
    """

    lam = rng.uniform(alpha_min, alpha_max)

    # mixup process
    rand_index = torch.randperm(img_gt.size(0))
    _img_gt = img_gt[rand_index]
    _img_lq = img_lq[rand_index]

    img_gt = lam * img_gt + (1 - lam) * _img_gt
    img_lq = lam * img_lq + (1 - lam) * _img_lq

    return img_gt, img_lq


@torch.no_grad()
def _cutmix(img2, prob=1.0, alpha=1.0):
    if alpha <= 0 or random.random() >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = img2.shape[2:]
    ch, cw = int(h * cut_ratio), int(w * cut_ratio)

    fcy = np.random.randint(0, h - ch + 1)
    fcx = np.random.randint(0, w - cw + 1)
    tcy, tcx = fcy, fcx
    r_index = torch.randperm(img2.size(0)).to(img2.device)

    return {
        "r_index": r_index,
        "ch": ch,
        "cw": cw,
        "tcy": tcy,
        "tcx": tcx,
        "fcy": fcy,
        "fcx": fcx,
    }


@torch.no_grad()
def cutmix(img_gt, img_lq, scale, alpha=0.9):
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

    def rand_bbox(size, scale, lam):
        """generate random box by lam"""
        W = size[2] // scale
        H = size[3] // scale
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        bb = (bbx1, bby1, bbx2, bby2)

        return tuple(bbi * scale for bbi in bb)

    lam = rng.uniform(0, alpha)
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
def resizemix(img_gt, img_lq, scale, scope=(0.5, 0.9)):
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

    def rand_bbox_tao(size, scale, tao):
        """generate random box by tao (scale)"""
        W = size[2] // scale
        H = size[3] // scale
        cut_w = int(W * tao)
        cut_h = int(H * tao)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        bb = (bbx1, bby1, bbx2, bby2)

        return tuple(bbi * scale for bbi in bb)

    # index
    rand_index = torch.randperm(img_gt.size(0))
    img_gt_resize = img_gt.clone()
    img_gt_resize = img_gt_resize[rand_index]
    img_lq_resize = img_lq.clone()
    img_lq_resize = img_lq_resize[rand_index]

    # generate tao
    tao = rng.uniform(scope[0], scope[1])

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
    )
    img_lq_resize = F.interpolate(
        img_lq_resize,
        (lq_bby2 - lq_bby1, lq_bbx2 - lq_bbx1),
        mode="bicubic",
        antialias=True,
    )

    # mix
    img_gt[:, :, gt_bby1:gt_bby2, gt_bbx1:gt_bbx2] = img_gt_resize
    img_lq[:, :, lq_bby1:lq_bby2, lq_bbx1:lq_bbx2] = img_lq_resize

    return img_gt, img_lq


# TODO
# def cutmixup(img1, img2, mixup_prob=1.0, mixup_alpha=1.0,
#     cutmix_prob=1.0, cutmix_alpha=1.0):  # (α1 / α2) -> 0.7 / 1.2
#     """ CutMix with the Mixup-ed image.
#     CutMix and Mixup procedure use hyper-parameter α1 and α2 respectively.
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
#     v = np.random.beta(mixup_alpha, mixup_alpha)
#     if mixup_alpha <= 0 or random.random() >= mixup_prob:
#         img1_aug = img1[r_index, :]
#         img2_aug = img2[r_index, :]
#     else:
#         img1_aug = v * img1 + (1-v) * img1[r_index, :]
#         img2_aug = v * img2 + (1-v) * img2[r_index, :]
#
#     # apply mixup to inside or outside
#     if np.random.random() > 0.5:
#         img1[..., htcy:htcy+hch, htcx:htcx+hcw] = img1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
#         img2[..., tcy:tcy+ch, tcx:tcx+cw] = img2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
#     else:
#         img1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = img1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
#         img2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = img2[..., fcy:fcy+ch, fcx:fcx+cw]
#         img2, img1 = img2_aug, img1_aug
#
#     return img1, img2


@torch.no_grad()
def cutblur(img_gt, img_lq, scale, alpha=0.7):
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

    def rand_bbox(size, scale, lam):
        """generate random box by lam (scale)"""
        W = size[2] // scale
        H = size[3] // scale
        cut_w = int(W * lam)
        cut_h = int(H * lam)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        bb = (bbx1, bby1, bbx2, bby2)

        return (bbi * scale for bbi in bb)

    lam = rng.uniform(0.2, alpha)
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


def downup(img_gt, img_lq, scope=(0.5, 0.9)):
    sampling_opts = [("bicubic", True), ("bilinear", True), ("nearest-exact", False)]

    down_sample = random.choice(sampling_opts)
    up_sample = random.choice(sampling_opts)

    # don't allow nearest for both
    if down_sample[0] == sampling_opts[2][0] and up_sample[0] == sampling_opts[2][0]:
        if rng.random() > 0.5:
            # change up sample
            while up_sample[0] == sampling_opts[2][0]:
                up_sample = random.choice(sampling_opts)
        else:
            # change down sample
            while down_sample[0] == sampling_opts[2][0]:
                down_sample = random.choice(sampling_opts)

    scale_factor = rng.uniform(scope[0], scope[1])
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


def up(img_gt, img_lq, scale, scope=(0.5, 0.9)):
    sampling_opts = [("bicubic", True), ("bilinear", True), ("nearest-exact", False)]

    def rand_bbox(size, scale, lam):
        """generate random box by lam (scale)"""
        W = size[2] // scale
        H = size[3] // scale
        cut_w = int(W * lam)
        cut_h = int(H * lam)

        pad_w = cut_w // 2
        pad_h = cut_h // 2

        # uniform
        cx = rng.integers(pad_w, W - pad_w)
        cy = rng.integers(pad_h, H - pad_w)

        bbx1 = cx - pad_w
        bby1 = cy - pad_h
        bbx2 = cx + pad_w
        bby2 = cy + pad_h

        bb = (bbx1, bby1, bbx2, bby2)

        return tuple(bbi * scale for bbi in bb)

    img_gt_base_size = img_gt.shape
    img_lq_base_size = img_lq.shape

    lam = rng.uniform(scope[0], scope[1])

    # random box
    gt_bbox = rand_bbox(img_gt.size(), scale, lam)
    gt_bbx1, gt_bby1, gt_bbx2, gt_bby2 = gt_bbox
    lq_bbox = tuple(bbi // 4 for bbi in gt_bbox)
    lq_bbx1, lq_bby1, lq_bbx2, lq_bby2 = lq_bbox

    # crop to random box
    img_gt = img_gt[:, :, gt_bbx1:gt_bbx2, gt_bby1:gt_bby2]
    img_lq = img_lq[:, :, lq_bbx1:lq_bbx2, lq_bby1:lq_bby2]

    assert (
        img_gt.shape[2] == img_gt.shape[3]
    ), f"Expected crop to be square, got shape {img_gt}"

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
