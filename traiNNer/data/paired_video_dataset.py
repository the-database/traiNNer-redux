# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py
import os

import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import paired_random_crop
from traiNNer.utils.file_client import FileClient
from traiNNer.utils.img_util import imfrombytes, imgs2tensors
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

        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.dataroot_lq = opt.dataroot_lq
        self.dataroot_gt = opt.dataroot_gt
        self.clip_size = opt.clip_size
        self.gt_size = opt.gt_size
        self.frames: dict[str, list[tuple[str, str]]] = {}

        for f in sorted(os.listdir(self.dataroot_lq)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                show_prefix = f.split("_")[0]
                lr_path = os.path.join(self.dataroot_lq, f)
                hr_path = os.path.join(self.dataroot_gt, f)

                if os.path.exists(hr_path):
                    if show_prefix not in self.frames:
                        self.frames[show_prefix] = []
                    self.frames[show_prefix].append((lr_path, hr_path))
                else:
                    print(f"Warning: No matching HR file for {f}")

        print(
            f"Found {sum(len(v) for v in self.frames.values())} valid file pairs across {len(self.frames)} shows"
        )

    def __len__(self) -> int:
        return sum(max(0, len(v) - self.clip_size + 1) for v in self.frames.values())

    def __getitem__(self, idx: int) -> DataFeed:
        scale = self.opt.scale
        assert scale is not None

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        for _show_prefix, clips in self.frames.items():
            if idx < len(clips) - self.clip_size + 1:
                lr_clip = []
                hr_clip = []

                for i in range(self.clip_size):
                    lq_path, gt_path = clips[idx + i]

                    img_gt = imfrombytes(
                        self.file_client.get(gt_path, "gt"), float32=True
                    )
                    img_lq = imfrombytes(
                        self.file_client.get(lq_path, "lq"), float32=True
                    )

                    if self.opt.phase == "train":
                        assert self.gt_size is not None
                        img_gt, img_lq = paired_random_crop(
                            img_gt, img_lq, self.gt_size, scale, gt_path
                        )

                    img_gt, img_lq = imgs2tensors(
                        [img_gt, img_lq], color=True, bgr2rgb=True, float32=True
                    )

                    lr_clip.append(img_lq)
                    hr_clip.append(img_gt)

                return {
                    "lq": torch.stack(lr_clip),
                    "gt": hr_clip[self.clip_size // 2],
                    "gt_path": clips[self.clip_size // 2][1],
                    "lq_path": clips[self.clip_size // 2][0],
                }
            idx -= len(clips) - self.clip_size + 1

        raise IndexError("Index out of range.")
