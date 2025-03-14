# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py
import os

import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.utils.file_client import FileClient
from traiNNer.utils.img_util import img2rgb, img2tensor, vipsimfrompath
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class SingleVideoDataset(BaseDataset):
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
        self.clip_size = opt.clip_size
        self.frames: dict[str, list[str]] = {}

        logger = get_root_logger()

        for _i, lq_path in enumerate(self.dataroot_lq):
            for f in sorted(os.listdir(lq_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    show_prefix = f.split("_")[0]
                    lr_path = os.path.join(lq_path, f)

                    if show_prefix not in self.frames:
                        self.frames[show_prefix] = []
                    self.frames[show_prefix].append(lr_path)

        logger.info(
            "Found %d valid frames across %d scenes.",
            sum(len(v) for v in self.frames.values()),
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

        for i in range(self.clip_size):
            lq_path = clips[i]
            vips_img_lq = vipsimfrompath(lq_path)
            img_lq = img2rgb(vips_img_lq.numpy())
            img_lq = img2tensor(img_lq, color=True, bgr2rgb=False, float32=True)
            lr_clip.append(img_lq)

        return {
            "lq": torch.stack(lr_clip),
            "lq_path": clips[self.clip_size // 2][0],
        }

    @property
    def label(self) -> str:
        return f"{self.clip_size}-frame sequences"
