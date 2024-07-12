import importlib
import random
from functools import partial
from os import path as osp

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import Dataset

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.prefetch_dataloader import PrefetchDataLoader
from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.dist_util import get_dist_info
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.rng import RNG
from traiNNer.utils.types import DataLoaderArgs

__all__ = ["build_dataset", "build_dataloader"]

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(data_folder)
    if v.endswith("_dataset.py")
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"traiNNer.data.{file_name}")
    for file_name in dataset_filenames
]


def build_dataset(dataset_opt: DatasetOptions) -> BaseDataset:
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    logger = get_root_logger()
    logger.info("Building Dataset %s...", dataset_opt.name)

    dataset = DATASET_REGISTRY.get(dataset_opt.type)(dataset_opt)

    logger.info(
        "Dataset [bold]%s[/bold] - %s is built.",
        dataset.__class__.__name__,
        dataset_opt.name,
        extra={"markup": True},
    )
    return dataset


def build_dataloader(
    dataset: Dataset,
    dataset_opt: DatasetOptions,
    num_gpu: int = 1,
    dist: bool = False,
    sampler: EnlargedSampler | None = None,
    seed: int | None = None,
) -> PrefetchDataLoader | torch.utils.data.DataLoader:
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    rank, _ = get_dist_info()
    if dataset_opt.phase == "train":
        assert dataset_opt.batch_size_per_gpu is not None
        assert dataset_opt.num_worker_per_gpu is not None

        if dist:  # distributed training
            batch_size = dataset_opt.batch_size_per_gpu
            num_workers = dataset_opt.num_worker_per_gpu
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt.batch_size_per_gpu * multiplier
            num_workers = dataset_opt.num_worker_per_gpu * multiplier
        dataloader_args: DataLoaderArgs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "sampler": sampler,
            "drop_last": True,
        }
        if sampler is None:
            dataloader_args["shuffle"] = True
        dataloader_args["worker_init_fn"] = (
            partial(
                worker_init_fn,
                num_workers=num_workers,
                rank=rank,
                seed=seed,
            )
            if seed is not None
            else None
        )

        dataloader_args["persistent_workers"] = dataset_opt.persistent_workers
    elif dataset_opt.phase in ["val", "test"]:  # validation
        dataloader_args = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }
    else:
        raise ValueError(
            f"Wrong dataset phase: {dataset_opt.phase}. Supported ones are 'train', 'val' and 'test'."
        )

    dataloader_args["pin_memory"] = dataset_opt.pin_memory
    prefetch_mode = dataset_opt.prefetch_mode
    if prefetch_mode == "cpu":  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.num_prefetch_queue
        logger = get_root_logger()
        logger.info(
            "Use %s prefetch dataloader: num_prefetch_queue = %d",
            prefetch_mode,
            num_prefetch_queue,
        )
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args
        )
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(
    worker_id: int,
    num_workers: int,
    rank: int,
    seed: int,
) -> None:
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
    RNG.init_rng(worker_seed)
