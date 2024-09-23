import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from traiNNer.utils.check_dependencies import check_dependencies

if __name__ == "__main__":
    check_dependencies()
import argparse
import logging
from os import path as osp

import torch
from rich.pretty import pretty_repr
from rich.traceback import install
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.models import build_model
from traiNNer.utils import (
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
from traiNNer.utils.misc import set_random_seed
from traiNNer.utils.options import copy_opt_file
from traiNNer.utils.redux_options import ReduxOptions


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
            pass
        elif phase.split("_")[0] in {"val", "test"}:
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


def train_pipeline(root_path: str) -> None:
    install()
    # torch.autograd.set_detect_anomaly(True)
    # parse options, set distributed setting, set random seed
    opt, args = Config.load_config_from_file(root_path, is_train=True)
    opt.root_path = root_path

    assert opt.logger is not None
    assert opt.manual_seed is not None
    assert opt.rank is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None

    if opt.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)

    # load resume states if necessary
    make_exp_dirs(opt, opt.resume > 0)
    # mkdir for experiments and logger
    if opt.resume == 0:
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

    _, _, val_loaders, _, _ = create_train_val_dataloader(
        opt, args, val_enabled, logger
    )

    if opt.fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True

    # create model
    model = build_model(opt)
    if model.with_metrics:
        for val_loader in val_loaders:
            if not (
                isinstance(val_loader.dataset, PairedImageDataset)
                or isinstance(val_loader.dataset, PairedVideoDataset)
            ):
                raise ValueError(
                    "Validation metrics are enabled, all validation datasets must have type PairedImageDataset."
                )

    current_iter = 0
    start_iter = opt.resume

    logger.info("Start testing from iter: %d.", start_iter)

    ext = opt.logger.save_checkpoint_format

    if opt.path.pretrain_network_g_path is not None and osp.isdir(
        opt.path.pretrain_network_g_path
    ):
        nets = list(
            scandir(
                opt.path.pretrain_network_g_path,
                suffix=ext,
                recursive=False,
                full_path=False,
            )
        )
        nets = [v.split(f".{ext}")[0].split("_")[-1] for v in nets]
        nets = sorted([int(v) for v in nets if v.isnumeric()])

        for net_iter in nets:
            if net_iter < start_iter:
                continue
            net_path = osp.join(
                opt.path.pretrain_network_g_path, f"net_g_ema_{net_iter}.{ext}"
            )
            print(net_path, osp.exists(net_path))
            if not osp.exists(net_path):
                net_path = osp.join(
                    opt.path.pretrain_network_g_path, f"net_g_{net_iter}.{ext}"
                )

            assert model.net_g is not None
            current_iter = net_iter
            model.load_network(model.net_g, net_path, True, None)
            # validation
            if opt.val is not None:
                multi_val_datasets = len(val_loaders) > 1
                for val_loader in val_loaders:
                    model.validation(
                        val_loader,
                        current_iter,
                        tb_logger,
                        opt.val.save_img,
                        multi_val_datasets,
                    )
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
