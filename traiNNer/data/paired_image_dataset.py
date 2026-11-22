from collections import OrderedDict
from typing import Any

import numpy as np
import pyvips

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from traiNNer.data.transforms import (
    augment_vips_pair,
    get_vips_dtype,
    paired_random_crop_vips,
    patch_wh,
)
from traiNNer.utils import FileClient, img2tensor
from traiNNer.utils.img_util import img2rgb, normalize_, vipsimfrompath
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
        self.io_backend_opt = opt.io_backend.copy()
        self._decoded: OrderedDict[str, tuple[pyvips.Image, int]] = OrderedDict()
        self._decoded_bytes = 0
        self.mean = opt.mean
        self.std = opt.std
        self.color = opt.color != "y"

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )
        assert isinstance(opt.dataroot_gt, list), (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )

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

    def __getstate__(self) -> dict[str, Any]:
        # Spawned workers open their own handles and fill their own cache.
        return {
            **vars(self),
            "file_client": None,
            "_decoded": OrderedDict(),
            "_decoded_bytes": 0,
        }

    def _read_image(self, path: str) -> pyvips.Image:
        cached = self._decoded.get(path)
        if cached is not None:
            self._decoded.move_to_end(path)
            return cached[0]

        image = vipsimfrompath(path)
        limit = self.opt.decoded_cache_mib * 2**20
        if limit and self.opt.phase == "train":
            size = (
                int(image.width)
                * int(image.height)
                * int(image.bands)
                * np.dtype(get_vips_dtype(image)).itemsize
            )
            if size <= limit:
                while self._decoded_bytes + size > limit:
                    _, (_, evicted) = self._decoded.popitem(last=False)
                    self._decoded_bytes -= evicted
                image = image.copy_memory()
                self._decoded[path] = (image, size)
                self._decoded_bytes += size
        return image

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            settings = self.io_backend_opt.copy()
            self.file_client = FileClient(settings.pop("type"), **settings)
        scale = self.opt.scale
        assert scale is not None
        gt_path = self.paths[index]["gt_path"]
        lq_path = self.paths[index]["lq_path"]
        vips_img_gt = self._read_image(gt_path)
        vips_img_lq = self._read_image(lq_path)

        # augmentation for training
        if self.opt.phase == "train":
            assert self.opt.gt_size is not None
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            # flip, rotation. Only rotate 90 degrees if the crop still fits the
            # rotated image, which can fail for rectangular or full-frame crops.
            gt_patch_w, gt_patch_h = patch_wh(self.opt.gt_size)
            rot90_fits = (
                vips_img_lq.height >= gt_patch_w // scale
                and vips_img_lq.width >= gt_patch_h // scale
            )
            vips_img_gt, vips_img_lq = augment_vips_pair(
                (vips_img_gt, vips_img_lq),
                self.opt.use_hflip,
                self.opt.use_rot,
                self.opt.use_rot and rot90_fits,
            )

            # random crop
            img_gt, img_lq = paired_random_crop_vips(
                vips_img_gt, vips_img_lq, self.opt.gt_size, scale, lq_path, gt_path
            )

            assert isinstance(img_gt, np.ndarray)
            assert isinstance(img_lq, np.ndarray)
        else:
            img_gt = img2rgb(vips_img_gt.numpy())
            img_lq = img2rgb(vips_img_lq.numpy())

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt.phase != "train":
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_tensor = img2tensor(img_gt, float32=True, from_bgr=False)
        img_lq_tensor = img2tensor(img_lq, float32=True, from_bgr=False)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize_(img_lq_tensor, self.mean, self.std)
            normalize_(img_gt_tensor, self.mean, self.std)

        return {
            "lq": img_lq_tensor,
            "gt": img_gt_tensor,
            "lq_path": lq_path,
            "gt_path": gt_path,
        }

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "image pairs"
