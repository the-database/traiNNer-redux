# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py
import os
import random

import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import augment_vips_pair, paired_random_crop_vips
from traiNNer.utils.file_client import FileClient
from traiNNer.utils.img_util import img2rgb, imgs2tensors, vipsimfrompath
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedVideoDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)

        assert opt.dataroot_lq is not None
        assert opt.dataroot_gt is not None
        assert opt.clip_size is not None

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )
        assert isinstance(opt.dataroot_gt, list), (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )

        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.dataroot_lq = opt.dataroot_lq
        self.dataroot_gt = opt.dataroot_gt
        self.clip_size = opt.clip_size
        self.gt_size = opt.gt_size
        self.frames: dict[str, list[tuple[str, str]]] = {}

        logger = get_root_logger()

        for i, lq_path in enumerate(self.dataroot_lq):
            for f in sorted(os.listdir(lq_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    show_prefix = f.split("_")[0]
                    lr_path = os.path.join(lq_path, f)
                    hr_path = os.path.join(self.dataroot_gt[i], f)

                    if os.path.exists(hr_path):
                        if show_prefix not in self.frames:
                            self.frames[show_prefix] = []
                        self.frames[show_prefix].append((lr_path, hr_path))
                    else:
                        logger.warning("No matching HR file for %s", f)

        logger.info(
            "Found %d valid file pairs across %d scenes.",
            sum(len(v) for v in self.frames.values()),
            len(self.frames),
        )

        if opt.phase == "train":
            if len(self.frames) < 100:
                logger.warning(
                    "Number of scene pairs is low: %d, training quality may be impacted. Please use more scene pairs for best training results.",
                    len(self.frames),
                )

        self.index_mapping = []
        for show_prefix, clips in self.frames.items():
            n_clips = len(clips) - self.clip_size + 1
            for start_idx in range(max(n_clips, 0)):
                self.index_mapping.append((show_prefix, start_idx))

    def __len__(self) -> int:
        return sum(max(0, len(v) - self.clip_size + 1) for v in self.frames.values())

    def __getitem__(self, idx: int) -> DataFeed:
        scale = self.opt.scale
        assert scale is not None

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scene, start_idx = self.index_mapping[idx]
        clips = self.frames[scene][start_idx : start_idx + self.clip_size]

        lr_clip = []
        hr_clip = []

        assert self.gt_size is not None
        lq_size = self.gt_size // scale
        force_x = None
        force_y = None
        force_hflip = None
        force_vflip = None
        force_rot90 = None

        for i in range(self.clip_size):
            lq_path, gt_path = clips[i]

            vips_img_gt = vipsimfrompath(gt_path)
            vips_img_lq = vipsimfrompath(lq_path)

            if self.opt.phase == "train":
                if force_x is None:
                    force_rot90 = random.random() < 0.5
                    force_hflip = random.random() < 0.5
                    force_vflip = random.random() < 0.5
                    h_lq: int = vips_img_lq.height  # pyright: ignore[reportAssignmentType]
                    w_lq: int = vips_img_lq.width  # pyright: ignore[reportAssignmentType]
                    if force_rot90:
                        h_lq, w_lq = w_lq, h_lq  # swap dimensions if rotating
                    force_y = random.randint(0, h_lq - lq_size)
                    force_x = random.randint(0, w_lq - lq_size)

                # flip, rotation
                vips_img_gt, vips_img_lq = augment_vips_pair(
                    (vips_img_gt, vips_img_lq),
                    self.opt.use_hflip,
                    self.opt.use_rot,
                    self.opt.use_rot,
                    force_hflip,
                    force_vflip,
                    force_rot90,
                )

                img_gt, img_lq = paired_random_crop_vips(
                    vips_img_gt,
                    vips_img_lq,
                    self.gt_size,
                    scale,
                    lq_path,
                    gt_path,
                    force_x,
                    force_y,
                )
            else:
                img_gt = img2rgb(vips_img_gt.numpy())
                img_lq = img2rgb(vips_img_lq.numpy())

            img_gt, img_lq = imgs2tensors(
                [img_gt, img_lq], color=True, bgr2rgb=False, float32=True
            )

            lr_clip.append(img_lq)
            hr_clip.append(img_gt)

        return {
            "lq": torch.stack(lr_clip),
            "gt": hr_clip[self.clip_size // 2],
            "gt_path": clips[self.clip_size // 2][1],
            "lq_path": clips[self.clip_size // 2][0],
        }

    @property
    def label(self) -> str:
        return f"{self.clip_size}-frame sequence pairs"
