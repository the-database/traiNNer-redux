import os
import random
import time
from collections.abc import Generator, Mapping
from os import path as osp
from typing import Any

import torch

from traiNNer.utils.dist_util import master_only


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
def make_exp_dirs(opt: Mapping[str, Any]) -> None:
    """Make dirs for experiments."""
    path_opt = opt["path"].copy()
    if opt["is_train"]:
        mkdir_and_rename(path_opt.pop("experiments_root"))
    else:
        mkdir_and_rename(path_opt.pop("results_root"))
    for key, path in path_opt.items():
        if (
            ("strict_load" in key)
            or ("pretrain_network" in key)
            or ("resume" in key)
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
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def check_resume(opt: Mapping[str, Any], resume_iter: int) -> None:
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """

    model_extensions = ["safetensors", "pth"]

    if opt["path"]["resume_state"]:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith("network_")]
        flag_pretrain = False
        for network in networks:
            if opt["path"].get(f"pretrain_{network}") is not None:
                flag_pretrain = True
        if flag_pretrain:
            print("pretrain_network path will be ignored during resuming.")
        # set pretrained model paths
        for network in networks:
            name = f"pretrain_{network}"
            basename = network.replace("network_", "")
            if opt["path"].get("ignore_resume_networks") is None or (
                network not in opt["path"]["ignore_resume_networks"]
            ):
                model_exists = False
                for ext in model_extensions:
                    basepath = osp.join(
                        opt["path"]["models"], f"net_{basename}_{resume_iter}"
                    )

                    if osp.exists(f"{basepath}.{ext}"):
                        opt["path"][name] = f"{basepath}.{ext}"
                        model_exists = True
                        break

                if not model_exists:
                    raise RuntimeError(
                        f"Unable to resume, model not found at path: {basepath}.{model_extensions[0]}"
                    )

                print(f"Set {name} to {opt['path'][name]}")

        # change param_key to params in resume
        param_keys = [key for key in opt["path"].keys() if key.startswith("param_key")]
        for param_key in param_keys:
            if opt["path"][param_key] == "params_ema":
                opt["path"][param_key] = "params"
                print(f"Set {param_key} to params")


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
