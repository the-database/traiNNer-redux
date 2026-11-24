"""The training loop behind train.py."""

import argparse
import datetime as dt
import logging
import math
import os
import signal
import time
from collections.abc import Generator, Iterable
from contextlib import closing, nullcontext
from os import path as osp
from types import FrameType
from typing import Any

import torch
from rich.traceback import install
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from traiNNer.models import build_model
from traiNNer.utils import (
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
from traiNNer.utils.dist_util import any_rank
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import (
    free_space_gb_str,
    set_random_seed,
)
from traiNNer.utils.options import (
    copy_opt_file,
    create_config_from_args,
    derive_train_patch_size,
    diff_user_vs_template,
    format_patch_size,
)
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed, TrainingState
from traiNNer.utils.win_qos import disable_power_throttling


def init_tb_loggers(opt: ReduxOptions) -> SummaryWriter | None:
    # initialize wandb logger before tensorboard logger to allow proper sync
    assert opt.logger is not None
    assert opt.root_path is not None

    if (opt.logger.wandb is not None) and (opt.logger.wandb.project is not None):
        assert opt.logger.use_tb_logger, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt.logger.use_tb_logger:
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

            derive_train_patch_size(dataset_opt, opt.scale, opt.high_order_degradation)

            train_set = build_dataset(dataset_opt)
            if not len(train_set):
                raise ValueError("The training dataset is empty.")
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

            batches_per_epoch = len(train_loader)
            if batches_per_epoch == 0:
                raise ValueError("Training needs at least one full batch per rank.")
            if dataset_opt.accum_iter < 1:
                raise ValueError("accum_iter must be positive.")
            assert train_loader.batch_size is not None
            opt.switch_iter_per_epoch = len(train_set) // (
                train_loader.batch_size * dataset_opt.accum_iter * opt.world_size
            )
            total_iters = int(opt.train.total_iter)
            total_epochs = math.ceil(
                total_iters * dataset_opt.accum_iter / batches_per_epoch
            )
            assert dataset_opt.gt_size is not None, "gt_size is required for train set"
            logger.info(
                "Training statistics for [b]%s[/b]:\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s",
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
                format_patch_size(dataset_opt.gt_size),
                "LR crop size:",
                format_patch_size(dataset_opt.lq_size),
                "World size (gpu number):",
                f"{opt.world_size:,}",
                "Microbatches per epoch:",
                f"{batches_per_epoch:,}",
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


def training_batches(
    loader: Iterable[DataFeed],
    sampler: EnlargedSampler,
    batch_size: int,
    epoch: int,
    samples_in_epoch: int,
) -> Generator[tuple[int, int, DataFeed], None, None]:
    """Stream sampler epochs, resuming before decoding the next unconsumed image."""
    if samples_in_epoch + batch_size > sampler.num_samples:
        epoch += 1
        samples_in_epoch = 0
    while True:
        sampler.set_epoch(epoch, start_index=samples_in_epoch)
        for batch in loader:
            samples_in_epoch += batch_size
            yield epoch, samples_in_epoch, batch
        epoch += 1
        samples_in_epoch = 0


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
        resume_state = torch.load(
            resume_state_path,
            map_location="cpu",
            weights_only=True,
        )
        assert resume_state is not None

        check_resume(
            opt,
            resume_state["iter"],
        )
    return resume_state


def train_pipeline(root_path: str) -> None:
    disable_power_throttling()
    install()

    # train.py --new creates a config and exits; it doesn't need a GPU
    if create_config_from_args(root_path):
        return

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

    torch.cuda.set_per_process_memory_fraction(fraction=1.0)

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if opt.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    make_exp_dirs(opt, resume_state is not None)
    # mkdir for experiments and logger
    if resume_state is None and opt.logger.use_tb_logger and opt.rank == 0:
        mkdir_and_rename(osp.join(opt.root_path, "tb_logger", opt.name))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt.path.experiments_root)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt.path.log, f"train_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="traiNNer", log_file=log_file)
    logger.info(get_env_info())
    logger.debug(opt.contents)
    opt.contents = None
    diff, template_name = diff_user_vs_template(args.opt)
    if diff and template_name:
        logger.info("Diff with default config (%s):\n%s", template_name, diff)

    if opt.deterministic:
        logger.info(
            "Training in deterministic mode with manual_seed=%d. Deterministic mode has reduced training speed.",
            opt.manual_seed,
        )
    else:
        logger.info(
            "Training with manual_seed=%d.",
            opt.manual_seed,
        )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    val_enabled = False
    if opt.val:
        val_enabled = opt.val.val_enabled

    train_loader, train_sampler, val_loaders, _, total_iters = (
        create_train_val_dataloader(opt, args, val_enabled, logger)
    )

    if train_loader is None or train_sampler is None:
        raise ValueError(
            "Failed to initialize training dataloader. Make sure train dataset is defined in datasets."
        )

    if opt.fast_matmul:
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    # create model
    model = build_model(opt)
    if model.with_metrics and not any(
        isinstance(
            val_loader.dataset,
            (PairedImageDataset | PairedVideoDataset),
        )
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

        model.samples_in_epoch = resume_state.get("samples_in_epoch", 0)
        del resume_state
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    train_opt = opt.datasets["train"]
    if train_opt.prefetch_mode == "cuda":
        if not train_opt.pin_memory:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
        prefetcher = CUDAPrefetcher(train_loader, opt)
    elif train_opt.prefetch_mode in (None, "cpu"):
        prefetcher = CPUPrefetcher(train_loader)
    else:
        raise ValueError(f"Unknown training prefetch mode: {train_opt.prefetch_mode}")

    if val_loaders and (
        opt.val is None or not opt.val.val_freq or opt.val.val_freq < 1
    ):
        raise ValueError("Enabled validation needs a positive val_freq.")
    assert train_loader.batch_size is not None
    assert opt.world_size is not None
    assert model.opt.path.models is not None
    batches = training_batches(
        prefetcher,
        train_sampler,
        train_loader.batch_size,
        start_epoch,
        model.samples_in_epoch,
    )
    samples_per_update = train_loader.batch_size * model.accum_iters * opt.world_size
    start_time = time.monotonic()
    epoch = start_epoch
    last_saved = current_iter
    last_validated = None
    can_recover = False
    stop_requested = False
    interrupted = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            if opt.dist:
                logger.info("Stopping together at the next log or checkpoint boundary.")
            else:
                logger.info("Finishing the current optimizer update before saving.")

    def save(step: int) -> None:
        try:
            model.save(epoch if step != -1 else -1, step)
        except Exception:
            if opt.dist:
                any_rank(True)
            raise
        if opt.dist and any_rank(False):
            raise RuntimeError(
                f"A checkpoint write failed on another rank at update {step}."
            )

    def validate(step: int) -> None:
        if opt.val is not None:
            for val_loader in val_loaders:
                model.validation(
                    val_loader, step, tb_logger, opt.val.save_img, len(val_loaders) > 1
                )

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    logger.info("Start training from epoch: %d, iter: %d.", epoch, current_iter)
    try:
        with closing(batches):
            for update in range(current_iter + 1, total_iters + 1):
                if stop_requested and not opt.dist:
                    interrupted = True
                    break
                model.set_current_iter(update)
                next_epoch, consumed = epoch, model.samples_in_epoch
                for microbatch in range(1, model.accum_iters + 1):
                    next_epoch, consumed, train_data = next(batches)
                    # A forward can mutate buffers even before the optimizer step.
                    can_recover = False
                    model.feed_data(train_data)
                    apply_gradient = microbatch == model.accum_iters
                    with (
                        model.no_sync()
                        if opt.dist and not apply_gradient
                        else nullcontext()
                    ):
                        model.optimize_parameters(
                            update, microbatch % model.accum_iters, apply_gradient
                        )

                model.update_learning_rate(update, warmup_iter=opt.train.warmup_iter)
                current_iter = update
                epoch, model.samples_in_epoch = next_epoch, consumed
                can_recover = True
                if not opt.dist:
                    interrupted = stop_requested
                elif (
                    update % opt.logger.print_freq == 0
                    or update % opt.logger.save_checkpoint_freq == 0
                    or update == total_iters
                ):
                    interrupted = any_rank(stop_requested)
                if update % opt.logger.print_freq == 0:
                    log_vars = {
                        "epoch": epoch,
                        "iter": update,
                        "lrs": model.get_current_learning_rate(),
                        "samples_per_update": samples_per_update,
                        **model.get_current_log(),
                    }
                    model.reset_current_log()
                    msg_logger(log_vars)

                if update % opt.logger.save_checkpoint_freq == 0 or interrupted:
                    logger.info(
                        "Saving update %d to %s. Free space: %s",
                        update,
                        clickable_file_path(
                            model.opt.path.models, "experiments folder"
                        ),
                        free_space_gb_str(),
                    )
                    can_recover = False
                    save(update)
                    last_saved = update
                if interrupted:
                    break
                if (
                    val_loaders
                    and opt.val
                    and opt.val.val_freq
                    and update % opt.val.val_freq == 0
                ):
                    validate(update)
                    last_validated = update

        if current_iter != last_saved:
            can_recover = False
            save(current_iter)
        if not interrupted:
            save(-1)
            if val_loaders and last_validated != current_iter:
                validate(current_iter)
    except Exception:
        if can_recover and not opt.dist:
            try:
                model.save(epoch, current_iter)
                logger.info("Recovered completed update %d.", current_iter)
            except Exception:
                logger.exception(
                    "Recovery checkpoint for update %d failed.", current_iter
                )
        logger.exception("Training failed after completed update %d.", current_iter)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if tb_logger:
            tb_logger.close()
        if model.opt.dist:
            # a CUDA graph holding NCCL collectives would make this wait forever
            model.release_cuda_graphs()
            destroy_process_group()
    elapsed = dt.timedelta(seconds=int(time.monotonic() - start_time))
    logger.info(
        "Training stopped at update %d. Time consumed: %s", current_iter, elapsed
    )
