from typing import NotRequired, TypedDict

from torch import Tensor
from torch.optim.optimizer import StateDict


class DataFeed(TypedDict):
    lq: NotRequired[Tensor]
    lq_path: NotRequired[str]
    gt: NotRequired[Tensor]
    gt_path: NotRequired[str]
    kernel1: NotRequired[Tensor]
    kernel2: NotRequired[Tensor]
    sinc_kernel: NotRequired[Tensor]


class TrainingState(TypedDict):
    epoch: int
    iter: int
    optimizers: list[StateDict]
    schedulers: list[StateDict]
    scaler_g: StateDict
    scaler_d: StateDict
