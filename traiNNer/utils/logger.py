import datetime
import logging
import shutil
import subprocess
import time
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from rich.logging import RichHandler
from rich.markup import escape
from torch.utils.tensorboard.writer import SummaryWriter

from traiNNer.utils.dist_util import get_dist_info, master_only
from traiNNer.utils.misc import free_space_gb_str
from traiNNer.utils.redux_options import ReduxOptions

initialized_logger = {}


class AvgTimer:
    def __init__(self, window: int = 200) -> None:
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start_time = None
        self.tic = None
        self.toc = None
        self.start()

    def start(self) -> None:
        self.start_time = self.tic = time.time()

    def record(self) -> None:
        if self.tic is None:
            raise ValueError("Must start timing before recording")
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self) -> float:
        return self.current_time

    def get_avg_time(self) -> float:
        return self.avg_time


class MessageLogger:
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Experiment name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iterations.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iteration. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default: None.
    """

    def __init__(
        self,
        opt: ReduxOptions,
        start_iter: int = 1,
        tb_logger: SummaryWriter | None = None,
    ) -> None:
        assert opt.logger is not None
        assert opt.train is not None

        self.exp_name = opt.name
        self.interval = opt.logger.print_freq
        self.start_iter = start_iter
        self.accum_iters = opt.datasets["train"].accum_iter
        self.max_iters = opt.train.total_iter
        self.use_tb_logger = opt.logger.use_tb_logger
        self.tb_logger = tb_logger

        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self) -> None:
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars: dict[str, Any]) -> None:
        """Format logging message.

        Args:
            log_vars (dict): Contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iteration.
                lrs (list): List of learning rates.
                time (float): Iteration time.
                data_time (float): Data loading time for each iteration.
        """
        # epoch, current iteration, and learning rates
        epoch = log_vars.pop("epoch")
        current_iter = log_vars.pop("iter")
        lrs = log_vars.pop("lrs")

        # Construct the base message with epoch, iteration, and learning rates
        message = f"[epoch:{epoch:4,d}, iter:{current_iter:8,d}, lr:("
        message += ", ".join([f"{v:.3e}" for v in lrs]) + ")] "

        # performance, eta
        if "time" in log_vars.keys():
            iter_time = 1 / (log_vars.pop("time") * self.accum_iters)
            log_vars.pop("data_time")

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

            message += f"[performance: {iter_time:.3f} it/s] [eta: {eta_str}] "

        # peak VRAM
        message += (
            f"[peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB] "
        )

        # Log any additional variables (typically losses)
        for k, v in log_vars.items():
            message += f"{k}: {v:.4e} "
            if self.tb_logger is not None:
                label = k
                if label.startswith("l_"):
                    label = f"losses/{label}"
                elif label.startswith("grad_norm_"):
                    label = f"grad_norms/{label}"
                elif label.startswith("scale_"):
                    label = f"scales/{label}"

                if isinstance(v, float):
                    value = v
                else:
                    value = (
                        v.to(dtype=torch.float32) if v.dtype == torch.bfloat16 else v
                    )
                self.tb_logger.add_scalar(label, value, current_iter)

        # Log the final constructed message
        self.logger.info(message, extra={"markup": False})


@master_only
def init_tb_logger(log_dir: str) -> SummaryWriter:
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt: ReduxOptions) -> None:
    """We now only use wandb to sync tensorboard log."""
    import wandb  # type: ignore

    assert opt.logger is not None
    assert opt.logger.wandb is not None
    logger = get_root_logger()

    project = opt.logger.wandb.project
    resume_id = opt.logger.wandb.resume_id
    if resume_id:
        wandb_id = resume_id
        resume = "allow"
        logger.warning("Resume wandb logger with id=%s.", wandb_id)
    else:
        wandb_id = wandb.util.generate_id()  # type: ignore
        resume = "never"

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt.name,
        config=opt,  # type: ignore
        project=project,
        sync_tensorboard=True,
    )

    logger.info("Use wandb logger with id=%s; project=%s.", wandb_id, project)


def get_root_logger(
    logger_name: str = "traiNNer",
    log_level_console: int = logging.INFO,
    log_level_file: int = logging.DEBUG,
    log_file: str | None = None,
) -> Logger:
    """Get the root logger.
    duf_downsample
        The logger will be initialized if it has not been initialized. By default a
        StreamHandler will be added. If `log_file` is specified, a FileHandler will
        also be added.

        Args:
            logger_name (str): root logger name. Default: 'traiNNer'.
            log_file (str | None): The log filename. If specified, a FileHandler
                will be added to the root logger.
            log_level (int): The root logger level. Note that only the process of
                rank 0 is affected, while other processes will set the level to
                "Error" and be silent most of the time.

        Returns:
            logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    format_str = "%(asctime)s %(levelname)s: %(message)s"
    rich_handler = RichHandler(
        markup=True, rich_tracebacks=True, omit_repeated_times=False
    )
    rich_handler.setLevel(log_level_console)
    logger.addHandler(rich_handler)
    logger.propagate = False

    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    else:
        logger.setLevel(log_level_file)
        if log_file is not None:
            # add file handler
            file_handler = logging.FileHandler(log_file, "w")
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level_file)
            logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info() -> str:
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    device_info = torch.cuda.get_device_properties(torch.cuda.current_device())

    # from traiNNer.version import __version__
    msg = r"[italic red]:rocket:  traiNNer-redux: good luck! :rocket:[/]"
    msg += (
        "\nSystem Information: "
        f"\n\tCurrent GPU: "
        f"\n\t\tName: {device_info.name}"
        f"\n\t\tTotal VRAM: {device_info.total_memory / (1024**3):.2f} GB"
        f"\n\t\tCompute Capability: {device_info.major}.{device_info.minor}"
        f"\n\t\tMultiprocessors: {device_info.multi_processor_count}"
        f"\n\tStorage:"
        f"\n\t\tFree Space: {free_space_gb_str()}"
        "\nVersion Information: "
        f"\n\ttraiNNer-redux: {log_git_status()}"
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
    )
    return msg


def log_git_status() -> str | None:
    logger = get_root_logger()
    if shutil.which("git") is None:
        logger.warning(
            "[yellow]Git is not installed or not available in PATH. "
            "You may have downloaded this repo as a ZIP, which is not recommended. "
            "Please install git, then clone the repo using [bold]git clone[/bold] to ensure easier updates. Please see the %s for more info.[/yellow]",
            clickable_url(
                "https://trainner-redux.readthedocs.io/en/latest/getting_started.html#initial-setup",
                "documentation",
            ),
        )
        return

    # Check if we're inside a Git repo
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "[yellow]This directory is not a Git repository. "
            "You may have downloaded this repo as a ZIP file, which is not recommended. "
            "Please clone the repo using [bold]git clone[/bold] to ensure easier updates. Please see the %s for more info.[/yellow]",
            clickable_url(
                "https://trainner-redux.readthedocs.io/en/latest/getting_started.html#initial-setup",
                "documentation",
            ),
        )
        return

    msg = ""

    try:
        # Get commit hash and date
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        commit_date = (
            subprocess.check_output(
                ["git", "show", "-s", "--format=%cd", "--date=iso", commit_hash],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Check if working directory is clean
        status_output = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty_flag = " (dirty)" if status_output else ""

        # Check if local is behind remote
        try:
            subprocess.check_output(
                ["git", "remote", "update"], stderr=subprocess.DEVNULL
            )
            behind_output = subprocess.check_output(
                ["git", "status", "-uno"], stderr=subprocess.DEVNULL
            ).decode("utf-8")

            behind_flag = ""
            if "Your branch is behind" in behind_output:
                behind_flag = " ([yellow]updates available - please use [bold]git pull[/bold] to update[/yellow])"
        except Exception:
            behind_flag = ""

        msg += f"\n\t\tGit commit: {commit_hash}{dirty_flag}{behind_flag}"
        msg += f"\n\t\tCommit date: {commit_date}"
        return msg

    except Exception as e:
        logger.warning("Failed to get git information: %s", e)


def clickable_file_path(file_path: str | Path, display_text: str) -> str:
    file_path = str(file_path).replace(" ", "%20")
    out = f"[link=file:///{file_path}]{escape(display_text)}[/link]"
    # print(out)
    return out


def clickable_url(url: str, display_text: str) -> str:
    url = str(url).replace(" ", "%20")
    out = f"[link={url}]{escape(display_text)}[/link]"
    return out
