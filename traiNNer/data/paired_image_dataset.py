import numpy as np
from torchvision.transforms.functional import normalize

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from traiNNer.data.transforms import augment_vips_pair, paired_random_crop_vips
from traiNNer.utils import FileClient, imgs2tensors
from traiNNer.utils.img_util import vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedImageDataset(BaseDataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.mean = opt.mean
        self.std = opt.std
        self.color = opt.color != "y"

        assert isinstance(
            opt.dataroot_lq, list
        ), f"dataroot_lq must be defined for dataset {opt.name}"
        assert isinstance(
            opt.dataroot_gt, list
        ), f"dataroot_gt must be defined for dataset {opt.name}"

        self.filename_tmpl = opt.filename_tmpl
        self.gt_folder, self.lq_folder = opt.dataroot_gt, opt.dataroot_lq

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]
            self.paths = paired_paths_from_lmdb(
                (self.lq_folder, self.gt_folder), ("lq", "gt")
            )
        elif self.opt.meta_info is not None:
            self.paths = paired_paths_from_meta_info_file(
                (self.lq_folder, self.gt_folder),
                ("lq", "gt"),
                self.opt.meta_info,
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                (self.lq_folder, self.gt_folder), ("lq", "gt"), self.filename_tmpl
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
        gt_path = self.paths[index]["gt_path"]

        try:
            vips_img_gt = vipsimfrompath(gt_path)
        except AttributeError as err:
            raise AttributeError(gt_path) from err

        lq_path = self.paths[index]["lq_path"]

        try:
            vips_img_lq = vipsimfrompath(lq_path)
        except AttributeError as err:
            raise AttributeError(lq_path) from err

        # augmentation for training
        if self.opt.phase == "train":
            assert self.opt.gt_size is not None
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            # flip, rotation
            vips_img_gt, vips_img_lq = augment_vips_pair(
                (vips_img_gt, vips_img_lq), self.opt.use_hflip, self.opt.use_rot
            )

            # random crop
            img_gt, img_lq = paired_random_crop_vips(
                vips_img_gt, vips_img_lq, self.opt.gt_size, scale, gt_path
            )

            assert isinstance(img_gt, np.ndarray)
            assert isinstance(img_lq, np.ndarray)
        else:
            img_gt = vips_img_gt.numpy().astype(np.float32) / 255.0
            img_lq = vips_img_lq.numpy().astype(np.float32) / 255.0

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt.phase != "train":
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = imgs2tensors(
            [img_gt, img_lq],
            color=self.color,
            bgr2rgb=False,
            float32=True,
        )
        # normalize
        if self.mean is not None and self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self) -> int:
        return len(self.paths)
