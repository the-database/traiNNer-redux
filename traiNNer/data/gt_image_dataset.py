from os import path as osp

import numpy as np

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import (
    paths_from_lmdb,
)
from traiNNer.data.transforms import augment_vips, single_random_crop_vips
from traiNNer.utils import FileClient, img2tensor
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.misc import scandir
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class GtImageDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.color = opt.color != "y"

        assert isinstance(opt.dataroot_gt, list), (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )

        self.filename_tmpl = opt.filename_tmpl
        self.gt_folder = opt.dataroot_gt

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = self.gt_folder
            self.io_backend_opt["client_keys"] = ["gt"] * len(self.gt_folder)
            self.paths = paths_from_lmdb(self.gt_folder)
        elif self.opt.meta_info is not None:
            self.paths = []
            with open(self.opt.meta_info) as fin:
                for line in fin:
                    filename = line.rstrip().split(" ")[0]
                    for folder in self.gt_folder:
                        self.paths.append(osp.join(folder, filename))

        else:
            self.paths = []
            for folder in self.gt_folder:
                self.paths.extend(
                    sorted(scandir(folder, recursive=True, full_path=True))
                )

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt.scale
        assert scale is not None

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]

        try:
            vips_img_gt = vipsimfrompath(gt_path)
        except AttributeError as err:
            raise AttributeError(gt_path) from err

        # augmentation for training
        if self.opt.phase == "train":
            assert self.opt.gt_size is not None
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            # flip, rotation
            vips_img_gt = augment_vips(
                vips_img_gt, self.opt.use_hflip, self.opt.use_rot
            )

            # random crop
            img_gt = single_random_crop_vips(vips_img_gt, self.opt.gt_size)

            assert isinstance(img_gt, np.ndarray)
        else:
            img_gt = img2rgb(vips_img_gt.numpy())

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(
            img_gt,
            color=self.color,
            bgr2rgb=False,
            float32=True,
        )

        return {"gt": img_gt, "gt_path": gt_path}

    def __len__(self) -> int:
        return len(self.paths)
