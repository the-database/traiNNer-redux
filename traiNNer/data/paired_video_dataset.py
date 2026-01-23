# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py
import os
import random

import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import (
    augment_vips,
    augment_vips_pair,
    paired_random_crop_vips,
    single_crop_vips,
)
from traiNNer.utils.file_client import FileClient
from traiNNer.utils.img_util import img2rgb, img2tensor, vipsimfrompath
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
        self.clips: list[tuple[list[str], str]] = []  # (lr_paths, hr_middle_path)
        self.frames: dict[str, list[tuple[str, str | None]]] = {}
        self.index_mapping: list[tuple[str, int]] = []

        logger = get_root_logger()

        total_clips = 0
        total_scenes = 0

        for i, lq_path in enumerate(self.dataroot_lq):
            gt_path = self.dataroot_gt[i]

            lr_files = sorted(
                f
                for f in os.listdir(lq_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )

            if not lr_files:
                continue

            hr_files = {
                f
                for f in os.listdir(gt_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            }

            scenes: dict[str, list[str]] = {}
            for f in lr_files:
                scene_prefix = f.split("_")[0]
                if scene_prefix not in scenes:
                    scenes[scene_prefix] = []
                scenes[scene_prefix].append(f)

            for scene_prefix, scene_files in scenes.items():
                full_scene_key = f"{lq_path}_{scene_prefix}"

                if full_scene_key not in self.frames:
                    self.frames[full_scene_key] = [
                        (
                            os.path.join(lq_path, f),
                            os.path.join(gt_path, f) if f in hr_files else None,
                        )
                        for f in scene_files
                    ]

                n_frames = len(scene_files)
                n_clips = n_frames - self.clip_size + 1
                scene_clips = 0

                for start_idx in range(max(n_clips, 0)):
                    middle_idx = start_idx + self.clip_size // 2
                    middle_filename = scene_files[middle_idx]

                    if middle_filename in hr_files:
                        lr_paths = [
                            os.path.join(lq_path, scene_files[start_idx + j])
                            for j in range(self.clip_size)
                        ]
                        hr_path = os.path.join(gt_path, middle_filename)
                        self.clips.append((lr_paths, hr_path))

                        self.index_mapping.append((full_scene_key, start_idx))

                        scene_clips += 1

                if scene_clips > 0:
                    total_scenes += 1
                    total_clips += scene_clips

        logger.info(
            "Found %d valid file pairs across %d scenes.",
            total_clips,
            total_scenes,
        )

        if opt.phase == "train":
            if total_scenes < 100:
                logger.warning(
                    "Number of scene pairs is low: %d, training quality may be impacted. Please use more scene pairs for best training results.",
                    total_scenes,
                )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> DataFeed:
        scale = self.opt.scale
        assert scale is not None
        assert self.clip_size > 0

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        lr_paths, hr_path = self.clips[idx]
        middle_idx = self.clip_size // 2

        lr_clip = []

        force_x = None
        force_y = None
        force_hflip = None
        force_vflip = None
        force_rot90 = None
        lq_size = 0

        if self.opt.phase == "train":
            assert self.gt_size is not None
            lq_size = self.gt_size // scale

        # middle frame only
        vips_img_gt = vipsimfrompath(hr_path)

        for i in range(self.clip_size):
            lq_path = lr_paths[i]
            vips_img_lq = vipsimfrompath(lq_path)

            if self.opt.phase == "train":
                assert self.gt_size is not None

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

                if i == middle_idx:
                    vips_img_gt_aug, vips_img_lq = augment_vips_pair(
                        (vips_img_gt, vips_img_lq),
                        self.opt.use_hflip,
                        self.opt.use_rot,
                        self.opt.use_rot,
                        force_hflip,
                        force_vflip,
                        force_rot90,
                    )
                    img_gt, img_lq = paired_random_crop_vips(
                        vips_img_gt_aug,
                        vips_img_lq,
                        self.gt_size,
                        scale,
                        lq_path,
                        hr_path,
                        force_x,
                        force_y,
                    )
                else:
                    assert force_hflip is not None
                    assert force_vflip is not None
                    assert force_rot90 is not None
                    assert force_y is not None
                    # For non-middle frames, only augment LR
                    vips_img_lq = augment_vips(
                        vips_img_lq,
                        force_hflip,
                        force_vflip,
                        force_rot90,
                        randomize=False,
                    )
                    # Crop LR only
                    img_lq = single_crop_vips(
                        vips_img_lq, lq_size, force_x, force_y, lq_path
                    )
            else:
                img_lq = img2rgb(vips_img_lq.numpy())
                if i == middle_idx:
                    img_gt = img2rgb(vips_img_gt.numpy())

            img_lq = img2tensor(
                img_lq,
                from_bgr=False,
                float32=True,
            )
            lr_clip.append(img_lq)

        # Process GT tensor
        img_gt = img2tensor(
            img_gt,  # pyright: ignore[reportPossiblyUnboundVariable]
            from_bgr=False,
            float32=True,
        )

        return {
            "lq": torch.stack(lr_clip),
            "gt": img_gt,
            "gt_path": hr_path,
            "lq_path": lr_paths[middle_idx],
        }

    @property
    def label(self) -> str:
        return f"{self.clip_size}-frame sequence pairs"
