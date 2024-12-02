from copy import deepcopy

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.data.realesrgan_dataset import RealESRGANDataset
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register(suffix="traiNNer")
class RealESRGANPairedDataset(BaseDataset):
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

        self.otf_dataset = RealESRGANDataset(deepcopy(opt))
        self.paired_dataset = PairedImageDataset(deepcopy(opt))

    def __getitem__(self, index: int) -> DataFeed:
        paired = self.paired_dataset[index]
        otf = self.otf_dataset[index]

        paired = {f"paired_{k}": v for k, v in paired.items()}
        otf = {f"otf_{k}": v for k, v in otf.items()}
        return paired | otf  # type: ignore

    def __len__(self) -> int:
        return len(self.otf_dataset)
