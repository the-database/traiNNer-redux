from typing import TypedDict

from torch import Tensor


class DataFeed(TypedDict):
    lq: Tensor
    gt: Tensor
    lq_path: str
    gt_path: str
    kernel1: Tensor
    kernel2: Tensor
    sinc_kernel: Tensor
