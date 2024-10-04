from os import path as osp

from torch import Tensor
from torchvision.transforms.functional import normalize

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import paths_from_lmdb
from traiNNer.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class SingleImageDataset(BaseDataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.mean = opt.mean
        self.std = opt.std
        self.lq_folder = opt.dataroot_lq

        assert self.lq_folder is not None and isinstance(
            self.lq_folder, list
        ), f"dataroot_lq must be defined as a list of paths for dataset {opt.name}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = self.lq_folder
            self.io_backend_opt["client_keys"] = ["lq"] * len(self.lq_folder)
            self.paths = paths_from_lmdb(self.lq_folder)

        elif self.opt.meta_info is not None:
            self.paths = []
            with open(self.opt.meta_info) as fin:
                for line in fin:
                    filename = line.rstrip().split(" ")[0]
                    for folder in self.lq_folder:
                        self.paths.append(osp.join(folder, filename))

        else:
            self.paths = []
            for folder in self.lq_folder:
                self.paths.extend(
                    sorted(scandir(folder, recursive=True, full_path=True))
                )

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, "lq")
        img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if self.opt.color == "y":
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        assert isinstance(img_lq, Tensor)
        # normalize
        if self.mean is not None and self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "lq_path": lq_path}

    def __len__(self) -> int:
        return len(self.paths)
