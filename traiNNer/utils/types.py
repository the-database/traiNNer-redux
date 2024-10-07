from collections.abc import Iterable
from typing import NotRequired, TypedDict

from torch import Tensor
from torch.optim.optimizer import StateDict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


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
    scaler_g: NotRequired[StateDict]
    scaler_d: NotRequired[StateDict]
    ema_n_averaged: NotRequired[Tensor]


class DataLoaderArgs(TypedDict):
    dataset: Dataset
    batch_size: NotRequired[int | None]
    shuffle: NotRequired[bool | None]
    sampler: NotRequired[Sampler | Iterable | None]
    batch_sampler: NotRequired[Sampler[list] | Iterable[list]]
    num_workers: NotRequired[int]
    collate_fn: NotRequired[_collate_fn_t | None]
    pin_memory: NotRequired[bool]
    drop_last: NotRequired[bool]
    timeout: NotRequired[float]
    worker_init_fn: NotRequired[_worker_init_fn_t | None]
    prefetch_factor: NotRequired[int | None]
    persistent_workers: NotRequired[bool]
    pin_memory_device: NotRequired[str]
