import argparse
import logging
import math
from logging import Logger
from os import path as osp
from pprint import pformat

from rich.pretty import pretty_repr
from rich.traceback import install
from torch.utils.data.dataset import Dataset
from traiNNer.data import build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.data.single_video_dataset import SingleVideoDataset
from traiNNer.utils import (
    get_env_info,
    get_root_logger,
)
from traiNNer.utils.config import Config
from traiNNer.utils.misc import set_random_seed
from traiNNer.utils.redux_options import ReduxOptions


def create_train_val_dataloader(
    opt: ReduxOptions,
    args: argparse.Namespace,
    val_enabled: bool,
    logger: logging.Logger,
) -> tuple[Dataset | None, EnlargedSampler | None, list[Dataset], int, int]:
    assert isinstance(opt.num_gpu, int)
    assert opt.world_size is not None
    assert opt.dist is not None

    # create train and val dataloaders
    train_set, _train_sampler, val_sets, total_epochs, total_iters = (
        None,
        None,
        [],
        0,
        0,
    )
    for phase, dataset_opt in opt.datasets.items():
        if phase == "train":
            assert opt.train is not None
            assert dataset_opt.batch_size_per_gpu is not None

            if dataset_opt.gt_size is None and dataset_opt.lq_size is not None:
                dataset_opt.gt_size = dataset_opt.lq_size * opt.scale
            elif dataset_opt.lq_size is None and dataset_opt.gt_size is not None:
                dataset_opt.lq_size = dataset_opt.gt_size // opt.scale
            else:
                raise ValueError(
                    "Exactly one of gt_size or lq_size must be defined in the train dataset"
                )

            train_set = build_dataset(dataset_opt)
            dataset_enlarge_ratio = dataset_opt.dataset_enlarge_ratio
            if dataset_enlarge_ratio == "auto":
                dataset_enlarge_ratio = max(
                    2000 * dataset_opt.batch_size_per_gpu // len(train_set), 1
                )

            iter_per_epoch = (
                len(train_set)
                * dataset_enlarge_ratio
                // (
                    dataset_opt.batch_size_per_gpu
                    * dataset_opt.accum_iter
                    * opt.world_size
                )
            )

            opt.switch_iter_per_epoch = len(train_set) // (
                dataset_opt.batch_size_per_gpu * dataset_opt.accum_iter * opt.world_size
            )

            total_iters = int(opt.train.total_iter)
            total_epochs = math.ceil(total_iters / (iter_per_epoch))
            assert dataset_opt.gt_size is not None, "gt_size is required for train set"
            logger.info(
                "Training statistics for [b]%s[/b]:\n"
                "\t%-40s %10s\t%-40s %10s\n"
                "\t%-40s %10s\t%-40s %10s\n"
                "\t%-40s %10s\t%-40s %10s\n"
                "\t%-40s %10s\t%-40s %10s\n"
                "\t%-40s %10s\t%-40s %10s",
                opt.name,
                f"Number of train {train_set.label}:",
                f"{len(train_set):,}",
                "Dataset enlarge ratio:",
                f"{dataset_enlarge_ratio:,}",
                "Batch size per gpu:",
                f"{dataset_opt.batch_size_per_gpu:,}",
                "Accumulate iterations:",
                f"{dataset_opt.accum_iter:,}",
                "HR crop size:",
                f"{dataset_opt.gt_size:,}",
                "LR crop size:",
                f"{dataset_opt.lq_size:,}",
                "World size (gpu number):",
                f"{opt.world_size:,}",
                "Require iter per epoch:",
                f"{iter_per_epoch:,}",
                "Total epochs:",
                f"{total_epochs:,}",
                "Total iters:",
                f"{total_iters:,}",
            )
            if len(train_set) < 100:
                logger.warning(
                    "Number of training %s is low: %d, training quality may be impacted. Please use more training %s for best training results.",
                    train_set.label,
                    len(train_set),
                    train_set.label,
                )
        elif phase.split("_")[0] == "val":
            if val_enabled:
                val_set = build_dataset(dataset_opt)

                logger.info(
                    "Number of val images/folders in %s: %d",
                    dataset_opt.name,
                    len(val_set),
                )
                val_sets.append(val_set)
            else:
                logger.info(
                    "Validation is disabled, skip building val dataset %s.",
                    dataset_opt.name,
                )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_set, None, val_sets, total_epochs, total_iters


def log_line(dataset: Dataset, i: int, logger: Logger) -> None:
    if hasattr(dataset, "index_mapping"):
        assert isinstance(dataset, PairedVideoDataset | SingleVideoDataset)
        scene, start_idx = dataset.index_mapping[i]
        clips = dataset.frames[scene][start_idx : start_idx + dataset.clip_size]
        if isinstance(clips[0], tuple):
            logger.info("%d %s", i, pformat([c[0] for c in clips]))
        else:
            logger.info("%d %s", i, pformat(clips))
    else:
        logger.info("%d %s", i, pformat(dataset[i]["lq_path"]))


def train_pipeline(root_path: str) -> None:
    install()

    # parse options, set distributed setting, set random seed
    opt, args = Config.load_config_from_file(root_path, is_train=True)
    opt.root_path = root_path

    # assert opt.train is not None
    # assert opt.logger is not None
    assert opt.manual_seed is not None
    assert opt.rank is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)

    logger = get_root_logger(logger_name="traiNNer")
    logger.info(get_env_info())
    logger.debug(pretty_repr(opt))

    # create train and validation dataloaders
    val_enabled = False
    if opt.val:
        val_enabled = opt.val.val_enabled

    train_set, _, val_sets, _, _ = create_train_val_dataloader(
        opt, args, val_enabled, logger
    )

    if train_set is not None:
        logger.info(
            "List of distinct scene names:\n%s",
            pformat(list(train_set.frames.keys())),  # pyright: ignore[reportAttributeAccessIssue]
        )

        logger.info("List of train LRs:")

        for i in range(len(train_set)):  # pyright: ignore[reportArgumentType]
            # logger.info("%d %s", i, pformat(train_set[i]["lq_path"]))
            log_line(train_set, i, logger)

    for val_set in val_sets:
        logger.info("List of val LRs:")
        for i in range(len(val_set)):
            log_line(val_set, i, logger)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
