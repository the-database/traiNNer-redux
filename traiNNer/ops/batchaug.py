# Code from: https://github.com/victorca25/traiNNer/blob/master/codes/dataops/batchaug.py

import random
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import os

rng = np.random.default_rng()


class BatchAugment:
    def __init__(self, train_opt):
        self.moa_augs = train_opt.get(
            "moa_augs", ["none", "mixup", "cutmix", "resizemix"])  # , "cutblur"]
        self.moa_probs = train_opt.get(
            "moa_probs", [0.4, 0.084, 0.084, 0.084, 0.348])  # , 1.0]
        self.scale = train_opt.get("scale", 4)
        self.debug = train_opt.get("moa_debug", False)

    def __call__(self, img1, img2):
        """Apply the configured augmentations.
        Args:
            img1: the target image.
            img2: the input image.
        """
        return BatchAug(img1, img2, self.scale, self.moa_augs, self.moa_probs, self.debug)


def BatchAug(img_gt, img_lq, scale, augs, probs, debug):
    """ Mixture of Batch Augmentations (MoA)
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
        while os.path.exists(rf'./augout/{i:06d}_preauglq.png'):
            i += 1

        torchvision.utils.save_image(img_lq, rf'./augout/{i:06d}_preauglq.png', padding=0)
        torchvision.utils.save_image(img_gt, rf'./augout/{i:06d}_preauggt.png', padding=0)

    if len(augs) != len(probs):
        msg = "Length of 'augmentation' and aug_prob don't match!"
        raise ValueError(msg)
    if img_gt.shape[0] == 1:
        msg = "Augmentations need batch >1 to work."
        raise ValueError(msg)

    idx = random.choices(range(len(augs)), weights=probs)[0]
    aug = augs[idx]
    print("BatchAug", aug)
    if aug == "none":
        return img_gt, img_lq

    # match resolutions
    if scale > 1:
        img_lq = F.interpolate(img_lq, scale_factor=scale, mode="bicubic")

    if "cutmix" in aug:
        img_gt, img_lq = cutmix(img_gt, img_lq, scale)
    elif "mixup" in aug:
        img_gt, img_lq = mixup(img_gt, img_lq)
    elif "resizemix" in aug:
        img_gt, img_lq = resizemix(img_gt, img_lq, scale)
    elif "cutblur" in aug:
        img_gt, img_lq = cutblur(img_gt, img_lq, scale)
    else:
        raise ValueError("{} is not invalid.".format(aug))

    # back to original resolution
    if scale > 1:
        img_lq = F.interpolate(img_lq, scale_factor=1 / scale, mode="bicubic")

    if debug:
        torchvision.utils.save_image(img_lq, rf'./augout/{i:06d}_postaug_{aug}_lqfinal.png', padding=0)
        torchvision.utils.save_image(img_gt, rf'./augout/{i:06d}_postaug_{aug}_gtfinal.png', padding=0)

    return img_gt, img_lq


@torch.no_grad()
def mixup(img_gt, img_lq, alpha_min=0.4, alpha_max=0.6):
    r"""MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10

    Args:
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha_min/max (float): The given min/max mixing ratio.
    """

    if img_gt.size() != img_lq.size():
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    lam = rng.uniform(alpha_min, alpha_max)

    # mixup process
    rand_index = torch.randperm(img_gt.size(0))
    img_ = img_gt[rand_index]

    img_gt = lam * img_gt + (1 - lam) * img_
    img_lq = lam * img_lq + (1 - lam) * img_

    return img_gt, img_lq


@torch.no_grad()
def _cutmix(img2, prob=1.0, alpha=1.0):
    if alpha <= 0 or random.random() >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = img2.shape[2:]
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)

    fcy = np.random.randint(0, h - ch + 1)
    fcx = np.random.randint(0, w - cw + 1)
    tcy, tcx = fcy, fcx
    r_index = torch.randperm(img2.size(0)).to(img2.device)

    return {
        "r_index": r_index, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
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

    if img_gt.size() != img_lq.size():
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

        return (bbi * scale for bbi in bb)

    lam = rng.uniform(0, alpha)
    rand_index = torch.randperm(img_gt.size(0))

    # mixup process
    img_gt_ = img_gt[rand_index]
    img_lq_ = img_lq[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(img_gt.size(), scale, lam)
    img_gt[:, :, bbx1:bbx2, bby1:bby2] = img_gt_[:, :, bbx1:bbx2, bby1:bby2]
    img_lq[:, :, bbx1:bbx2, bby1:bby2] = img_lq_[:, :, bbx1:bbx2, bby1:bby2]

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

    if img_gt.size() != img_lq.size():
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

        return (bbi * scale for bbi in bb)

    # index
    rand_index = torch.randperm(img_gt.size(0))
    img_gt_resize = img_gt.clone()
    img_gt_resize = img_gt_resize[rand_index]
    img_lq_resize = img_lq.clone()
    img_lq_resize = img_lq_resize[rand_index]

    # generate tao
    tao = rng.uniform(scope[0], scope[1])

    # random box
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img_gt.size(), scale, tao)

    # resize
    img_gt_resize = F.interpolate(
        img_gt_resize, (bby2 - bby1, bbx2 - bbx1), mode="bicubic"
    )
    img_lq_resize = F.interpolate(
        img_lq_resize, (bby2 - bby1, bbx2 - bbx1), mode="bicubic"
    )

    # mix
    img_gt[:, :, bby1:bby2, bbx1:bbx2] = img_gt_resize
    img_lq[:, :, bby1:bby2, bbx1:bbx2] = img_lq_resize

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
    if img_gt.size() != img_lq.size():
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

    if rng.uniform() < 0.5:
        # cutblur inside
        img_lq[:, :, bbx1:bbx2, bby1:bby2] = img_gt[:, :, bbx1:bbx2, bby1:bby2]
    else:
        # cutblur outside
        img_lq_aug = img_gt.clone()
        img_lq_aug[:, :, bbx1:bbx2, bby1:bby2] = img_lq[:, :, bbx1:bbx2, bby1:bby2]
        img_lq = img_lq_aug

    return img_gt, img_lq
