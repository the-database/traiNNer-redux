import os
import os.path as osp

import pyvips

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import augment_vips, single_random_crop_vips
from traiNNer.utils import FileClient, img2tensor, scandir
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register(suffix="traiNNer")
class SingleGtDataset(BaseDataset):
    """GT-only dataset for training workflows where LR is synthesized on the
    fly by the model (e.g. ECO). Loads HR images with flip/rotation
    augmentation and random cropping to gt_size. Returns {"gt", "gt_path",
    "lq_path"}. lq_path is a harmless placeholder so downstream logging code
    that assumes a paired API still works.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.gt_folders = opt.dataroot_gt

        assert self.gt_folders is not None, (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )
        assert isinstance(self.gt_folders, list), (
            f"dataroot_gt must be a list of folders for dataset {opt.name}"
        )
        assert opt.gt_size is not None, f"gt_size must be set for dataset {opt.name}"
        assert opt.scale is not None
        assert opt.gt_size % opt.scale == 0, (
            f"gt_size ({opt.gt_size}) must be divisible by scale ({opt.scale}) "
            f"for dataset {opt.name}"
        )

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = self.gt_folders
            self.io_backend_opt["client_keys"] = ["gt"] * len(self.gt_folders)

            for folder in self.gt_folders:
                if not folder.endswith(".lmdb"):
                    raise ValueError(
                        f"Each 'dataroot_gt' should end with '.lmdb', but received {folder}"
                    )

            self.paths = []
            for folder in self.gt_folders:
                with open(osp.join(folder, "meta_info.txt")) as fin:
                    self.paths.extend([line.split(".")[0] for line in fin])

        elif self.opt.meta_info is not None:
            self.paths = []
            for folder in self.gt_folders:
                with open(self.opt.meta_info) as fin:
                    paths = [line.strip().split(" ")[0] for line in fin]
                    self.paths.extend([os.path.join(folder, v) for v in paths])
        else:
            self.paths = []
            for folder in self.gt_folders:
                self.paths.extend(
                    sorted(scandir(folder, recursive=True, full_path=True))
                )

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]

        assert self.opt.use_hflip is not None
        assert self.opt.use_rot is not None
        assert self.opt.gt_size is not None

        vips_img_gt = vipsimfrompath(gt_path)
        vips_img_gt = augment_vips(vips_img_gt, self.opt.use_hflip, self.opt.use_rot)

        h: int = vips_img_gt.height  # type: ignore
        w: int = vips_img_gt.width  # type: ignore
        gt_size = self.opt.gt_size

        if h < gt_size or w < gt_size:
            pad_h = max(0, gt_size - h)
            pad_w = max(0, gt_size - w)
            vips_img_gt: pyvips.Image = vips_img_gt.embed(0, 0, w + pad_w, h + pad_h)  # type: ignore
            h, w = vips_img_gt.height, vips_img_gt.width  # type: ignore

        if w > gt_size or h > gt_size:
            img_gt = single_random_crop_vips(vips_img_gt, gt_size)
        else:
            img_gt = img2rgb(vips_img_gt.numpy())

        img_gt_tensor = img2tensor(img_gt, from_bgr=False, float32=True)

        out: DataFeed = {
            "gt": img_gt_tensor,
            "gt_path": gt_path,
            "lq_path": gt_path,
        }
        if self.opt.lq_resize_mode is not None:
            out["lq_resize_mode"] = self.opt.lq_resize_mode
        return out

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "GT-only images"
