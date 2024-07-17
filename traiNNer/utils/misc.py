import os
import random
import time
from collections.abc import Generator
from os import path as osp

import torch
from rich import print

from traiNNer.utils.dist_util import master_only
from traiNNer.utils.options import struct2dict
from traiNNer.utils.redux_options import ReduxOptions


def set_random_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path: str) -> None:
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + "_archived_" + get_time_str()
        print(f"Path already exists. Rename it to {new_name}", flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt: ReduxOptions, is_resume: bool = False) -> None:
    """Make dirs for experiments."""
    path_opt = struct2dict(opt.path)
    if not is_resume:
        if opt.is_train:
            mkdir_and_rename(path_opt.pop("experiments_root"))
        else:
            mkdir_and_rename(path_opt.pop("results_root"))
    for key, path in path_opt.items():
        if (
            ("strict_load" in key)
            or ("pretrain_network" in key)
            or ("resume_state" in key)
            or ("ignore_reusme_networks" in key)
            or ("param_key" in key)
        ):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def scandir(
    dir_path: str,
    suffix: str | tuple[str] | None = None,
    recursive: bool = False,
    full_path: bool = False,
) -> Generator[str]:
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """
    root = dir_path

    def _scandir(
        dir_path: str, suffix: str | tuple[str] | None, recursive: bool
    ) -> Generator[str]:
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = (
                        entry.name if not recursive else osp.relpath(entry.path, root)
                    )

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def check_resume(opt: ReduxOptions, resume_iter: int) -> None:
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    if opt.path.resume_state:
        assert opt.train is not None
        assert opt.path.models is not None
        assert opt.path.resume_models is not None
        model_extensions = ["safetensors", "pth"]
        model_dirs = [opt.path.models, opt.path.resume_models]

        flag_pretrain = (opt.path.pretrain_network_g is not None) or (
            opt.network_d and opt.path.pretrain_network_d is not None
        )
        if flag_pretrain:
            print("pretrain_network path will be ignored during resuming.")

        # set pretrained model paths
        if opt.train.ema_decay > 0 and (
            opt.path.ignore_resume_networks is None
            or "network_g_ema" not in opt.path.ignore_resume_networks
        ):
            model_exists = False
            basepath = ""
            for ext in model_extensions:
                for net_label in ["net_g_ema", "net_g"]:
                    basepath = osp.join(opt.path.models, f"{net_label}_{resume_iter}")
                    if osp.exists(f"{basepath}.{ext}"):
                        opt.path.pretrain_network_g_ema = f"{basepath}.{ext}"
                        model_exists = True
                        print(
                            f"Set pretrain_network_g_ema to {opt.path.pretrain_network_g_ema}"
                        )
            if not model_exists:
                raise FileNotFoundError(
                    f"Unable to resume, pretrain_network_g_ema not found at path: {basepath}.{model_extensions[0]}"
                )

        if (
            opt.path.ignore_resume_networks is None
            or "network_g" not in opt.path.ignore_resume_networks
        ):
            model_exists = False
            basepath = ""
            for ext in model_extensions:
                for model_dir in model_dirs:
                    basepath = osp.join(model_dir, f"net_g_{resume_iter}")
                    if osp.exists(f"{basepath}.{ext}"):
                        opt.path.pretrain_network_g = f"{basepath}.{ext}"
                        model_exists = True
                        print(
                            f"Set pretrain_network_g to {opt.path.pretrain_network_g}"
                        )

            if not model_exists:
                raise FileNotFoundError(
                    f"Unable to resume, pretrain_network_g not found at path: {basepath}.{model_extensions[0]}",
                )

        if opt.network_d is not None and (
            opt.path.ignore_resume_networks is None
            or "network_d" not in opt.path.ignore_resume_networks
        ):
            model_exists = False
            basepath = ""
            for ext in model_extensions:
                for model_dir in model_dirs:
                    basepath = osp.join(model_dir, f"net_d_{resume_iter}")
                    if osp.exists(f"{basepath}.{ext}"):
                        opt.path.pretrain_network_d = f"{basepath}.{ext}"
                        model_exists = True
                        print(
                            f"Set pretrain_network_d to {opt.path.pretrain_network_d}",
                        )

            if not model_exists:
                raise FileNotFoundError(
                    f"Unable to resume, pretrain_network_d not found at path: {basepath}.{model_extensions[0]}"
                )

        if opt.path.param_key_g == "params_ema":
            opt.path.param_key_g = "params"
            print("Set param_key_g to params")

        if opt.path.param_key_d == "params_ema":
            opt.path.param_key_d = "params"
            print("Set param_key_d to params")


def sizeof_fmt(size: float, suffix: str = "B") -> str:
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size /= 1024.0
    return f"{size:3.1f} Y{suffix}"
