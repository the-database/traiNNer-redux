import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from traiNNer.check.check_dependencies import check_dependencies

if __name__ == "__main__":
    check_dependencies()


import argparse
import datetime
import logging
import math
import signal
import sys
import time
from os import path as osp
from types import FrameType
from typing import Any

import torch
from rich.pretty import pretty_repr
from rich.traceback import install
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from traiNNer.models import build_model
from traiNNer.utils import (
    AvgTimer,
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
)
from traiNNer.utils.config import Config
from traiNNer.utils.misc import free_space_gb_str, set_random_seed
from traiNNer.utils.options import copy_opt_file
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import TrainingState


def init_tb_loggers(opt: ReduxOptions) -> SummaryWriter | None:
    # initialize wandb logger before tensorboard logger to allow proper sync
    assert opt.logger is not None
    assert opt.root_path is not None

    if (
        (opt.logger.wandb is not None)
        and (opt.logger.wandb.project is not None)
        and ("debug" not in opt.name)
    ):
        assert opt.logger.use_tb_logger, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt.logger.use_tb_logger and "debug" not in opt.name:
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt.root_path, "tb_logger", opt.name)
        )
    return tb_logger


def create_train_val_dataloader(
    opt: ReduxOptions,
    args: argparse.Namespace,
    val_enabled: bool,
    logger: logging.Logger,
) -> tuple[DataLoader | None, EnlargedSampler | None, list[DataLoader], int, int]:
    assert isinstance(opt.num_gpu, int)
    assert opt.world_size is not None
    assert opt.dist is not None

    # create train and val dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = (
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
            train_sampler = EnlargedSampler(
                train_set, opt.world_size, opt.rank, dataset_enlarge_ratio
            )
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt.num_gpu,
                dist=opt.dist,
                sampler=train_sampler,
                seed=opt.manual_seed,
            )

            num_iter_per_epoch = (
                len(train_set)
                * dataset_enlarge_ratio
                // (
                    dataset_opt.batch_size_per_gpu
                    * dataset_opt.accum_iter
                    * opt.world_size
                )
            )

            total_iters = int(opt.train.total_iter)
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            assert dataset_opt.gt_size is not None, "gt_size is required for train set"
            logger.info(
                "Training statistics:\n"
                "\t%-25s %10s\t%-25s %10s\n"
                "\t%-25s %10s\t%-25s %10s\n"
                "\t%-25s %10s\t%-25s %10s\n"
                "\t%-25s %10s\t%-25s %10s\n"
                "\t%-25s %10s\t%-25s %10s",
                "Number of train images:",
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
                f"{num_iter_per_epoch:,}",
                "Total epochs:",
                f"{total_epochs:,}",
                "Total iters:",
                f"{total_iters:,}",
            )
            if len(train_set) < 100:
                logger.warning(
                    "Number of train images is low: %d, training quality may be impacted. Please use more train images for best training results.",
                    len(train_set),
                )
        elif phase.split("_")[0] == "val":
            if val_enabled:
                val_set = build_dataset(dataset_opt)
                val_loader = build_dataloader(
                    val_set,
                    dataset_opt,
                    num_gpu=opt.num_gpu,
                    dist=opt.dist,
                    sampler=None,
                    seed=opt.manual_seed,
                )
                logger.info(
                    "Number of val images/folders in %s: %d",
                    dataset_opt.name,
                    len(val_set),
                )
                val_loaders.append(val_loader)
            else:
                logger.info(
                    "Validation is disabled, skip building val dataset %s.",
                    dataset_opt.name,
                )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt: ReduxOptions) -> Any | None:
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join("experiments", opt.name, "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [
                    [int(x) for x in v.split(".state")[0].split("_")] for v in states
                ]

                resume_state_path = osp.join(
                    state_path, f"{'_'.join([str(x) for x in max(states)])}.state"
                )
                opt.path.resume_state = resume_state_path
    elif opt.path.resume_state:
        resume_state_path = opt.path.resume_state

    if resume_state_path is None:
        resume_state: TrainingState | None = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path,
            map_location=lambda storage, _: storage.cuda(device_id),  # pyright: ignore[reportAttributeAccessIssue] (https://github.com/pytorch/pytorch/issues/131765)
            weights_only=True,
        )
        assert resume_state is not None

        check_resume(
            opt,
            resume_state["iter"],
        )
    return resume_state


def train_pipeline(root_path: str) -> None:
    install()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure that you have a GPU with CUDA support "
            "and that you have installed the correct CUDA-enabled version of PyTorch. "
            "You can check the installation guide at https://pytorch.org/get-started/locally/"
        )

    # parse options, set distributed setting, set random seed
    opt, args = Config.load_config_from_file(root_path, is_train=True)
    opt.root_path = root_path

    assert opt.train is not None
    assert opt.logger is not None
    assert opt.manual_seed is not None
    assert opt.rank is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if opt.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    make_exp_dirs(opt, resume_state is not None)
    # mkdir for experiments and logger
    if resume_state is None:
        if opt.logger.use_tb_logger and "debug" not in opt.name and opt.rank == 0:
            mkdir_and_rename(osp.join(opt.root_path, "tb_logger", opt.name))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt.path.experiments_root)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt.path.log, f"train_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="traiNNer", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(pretty_repr(opt))

    if opt.deterministic:
        logger.info(
            "Training in deterministic mode with manual seed=%d. Deterministic mode has reduced training speed.",
            opt.manual_seed,
        )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    val_enabled = False
    if opt.val:
        val_enabled = opt.val.val_enabled

    train_loader, train_sampler, val_loaders, total_epochs, total_iters = (
        create_train_val_dataloader(opt, args, val_enabled, logger)
    )

    if train_loader is None or train_sampler is None:
        raise ValueError(
            "Failed to initialize training dataloader. Make sure train dataset is defined in datasets."
        )

    if opt.fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True

    # create model
    model = build_model(opt)
    if model.with_metrics:
        if not any(
            isinstance(val_loader.dataset, (PairedImageDataset | PairedVideoDataset))
            for val_loader in val_loaders
        ):
            raise ValueError(
                "Validation metrics are enabled, at least one validation dataset must have type PairedImageDataset or PairedVideoDataset."
            )

    if torch.is_anomaly_enabled():
        logger.warning(
            "!!!! Anomaly checking is enabled. This slows down training and should be used for testing only !!!!"
        )

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            "Resuming training from epoch: %d, iter: %d.",
            resume_state["epoch"],
            resume_state["iter"],
        )
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        start_epoch = 0
        current_iter = 0

    current_accum_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt.datasets["train"].prefetch_mode
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info("Use %s prefetch dataloader", prefetch_mode)
        if not opt.datasets["train"].pin_memory:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'."
        )

    # training
    logger.info("Start training from epoch: %d, iter: %d.", start_epoch, current_iter)
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    interrupt_received = False

    def handle_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupt_received
        if not interrupt_received:
            logger.info("User interrupted. Preparing to save state...")
            interrupt_received = True

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    epoch = start_epoch
    apply_gradient = False
    train_data = None

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_accum_iter += 1

            if current_accum_iter >= model.accum_iters:
                current_accum_iter = 0
                current_iter += 1
                apply_gradient = True
            else:
                apply_gradient = False

            if current_iter > total_iters:
                break
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter, current_accum_iter, apply_gradient)
            # update learning rate
            if apply_gradient:
                model.update_learning_rate(
                    current_iter, warmup_iter=opt.train.warmup_iter
                )
            iter_timer.record()
            if current_iter == msg_logger.start_iter + 1:
                # reset start time in msg_logger for more accurate eta_time
                msg_logger.reset_start_time()
            # log
            if current_iter % opt.logger.print_freq == 0 and apply_gradient:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update(
                    {
                        "time": iter_timer.get_avg_time(),
                        "data_time": data_timer.get_avg_time(),
                    }
                )
                log_vars.update(model.get_current_log())
                model.reset_current_log()
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt.logger.save_checkpoint_freq == 0 and apply_gradient:
                logger.info(
                    "Saving models and training states. Free space: %s",
                    free_space_gb_str(),
                )
                model.save(
                    epoch,
                    current_iter,
                )

            # validation
            if opt.val is not None:
                assert (
                    opt.val.val_freq is not None
                ), "val_freq must be defined under the val section"
                if current_iter % opt.val.val_freq == 0 and apply_gradient:
                    multi_val_datasets = len(val_loaders) > 1
                    for val_loader in val_loaders:
                        model.validation(
                            val_loader,
                            current_iter,
                            tb_logger,
                            opt.val.save_img,
                            multi_val_datasets,
                        )

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

            if interrupt_received:
                break

        # end of iter
        if interrupt_received:
            break
    # end of epoch

    # epoch was completed, increment it to set the correct epoch count when interrupted
    if train_data is None:
        epoch += 1

    if interrupt_received:
        # discard partially accumulated iters
        if not apply_gradient:
            current_iter -= 1

        logger.info(
            "Saving models and training states for epoch: %d, iter: %d.",
            epoch,
            current_iter,
        )
        model.save(epoch, current_iter)
        sys.exit(0)

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info("End of training. Time consumed: %s", consumed_time)
    logger.info("Save the latest model.")
    model.save(
        epoch=-1,
        current_iter=-1,
    )  # -1 stands for the latest
    if opt.val is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt.val.save_img)
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
