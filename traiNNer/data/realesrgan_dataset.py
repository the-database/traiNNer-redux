import math
import os
import os.path as osp
import random

import numpy as np
import pyvips
import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from traiNNer.data.transforms import augment_vips
from traiNNer.utils import (
    RNG,
    FileClient,
    img2tensor,
    scandir,
)
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register(suffix="traiNNer")
class RealESRGANDataset(BaseDataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.gt_folders = opt.dataroot_gt

        assert isinstance(self.gt_folders, list), (
            f"dataroot_gt must be a list of folders for dataset {opt.name}"
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
                self.paths.extend(sorted(scandir(folder, full_path=True)))

        # blur settings for the first degradation
        self.blur_kernel_size = opt.blur_kernel_size
        self.kernel_list = opt.kernel_list
        self.kernel_prob = opt.kernel_prob  # a list for each kernel probability
        self.blur_sigma = opt.blur_sigma
        self.betag_range = (
            opt.betag_range
        )  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt.betap_range  # betap used in plateau blur kernels
        self.sinc_prob = opt.sinc_prob  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt.blur_kernel_size2
        self.kernel_list2 = opt.kernel_list2
        self.kernel_prob2 = opt.kernel_prob2
        self.blur_sigma2 = opt.blur_sigma2
        self.betag_range2 = opt.betag_range2
        self.betap_range2 = opt.betap_range2
        self.sinc_prob2 = opt.sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = opt.final_sinc_prob

        self.kernel_range = list(range(opt.kernel_range[0], opt.kernel_range[1] + 1, 2))
        self.kernel_range2 = list(
            range(opt.kernel_range2[0], opt.kernel_range2[1] + 1, 2)
        )
        self.final_kernel_range = list(
            range(opt.final_kernel_range[0], opt.final_kernel_range[1] + 1, 2)
        )
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]

        assert self.opt.use_hflip is not None
        assert self.opt.use_rot is not None

        vips_img_gt = vipsimfrompath(gt_path)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        vips_img_gt = augment_vips(vips_img_gt, self.opt.use_hflip, self.opt.use_rot)

        h: int = vips_img_gt.height  # type: ignore
        w: int = vips_img_gt.width  # type: ignore
        assert self.opt.gt_size is not None

        crop_pad_size = self.opt.gt_size + 32
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            vips_img_gt: pyvips.Image = vips_img_gt.embed(0, 0, w + pad_w, h + pad_h)  # type: ignore
        # crop
        if w > crop_pad_size or h > crop_pad_size:
            y = random.randint(0, h - crop_pad_size)
            x = random.randint(0, w - crop_pad_size)
            region_gt = pyvips.Region.new(vips_img_gt)
            data_gt = region_gt.fetch(x, y, crop_pad_size, crop_pad_size)
            img_gt = img2rgb(
                np.ndarray(
                    buffer=data_gt,
                    dtype=np.uint8,
                    shape=[crop_pad_size, crop_pad_size, vips_img_gt.bands],  # pyright: ignore
                )
            )
        else:
            img_gt = img2rgb(vips_img_gt.numpy())

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if RNG.get_rng().uniform() < self.opt.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                (-math.pi, math.pi),
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if RNG.get_rng().uniform() < self.opt.sinc_prob2:
            if kernel_size < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                (-math.pi, math.pi),
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if RNG.get_rng().uniform() < self.opt.final_sinc_prob:
            kernel_size = random.choice(self.final_kernel_range)
            omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
        }

    def __len__(self) -> int:
        return len(self.paths)
