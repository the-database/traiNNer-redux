import os
import random
import re
import shutil
import time
from collections.abc import Generator, Sequence
from os import path as osp
from typing import Any

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
        # set generator ema path
        if (
            opt.network_g is not None
            and opt.train.ema_decay > 0
            and (
                opt.path.ignore_resume_networks is None
                or "network_g_ema" not in opt.path.ignore_resume_networks
            )
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

        # set generator path
        if opt.network_g is not None and (
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

        # set discriminator path if gan training
        if opt.network_d is not None and (
            opt.path.ignore_resume_networks is None
            or "network_d" not in opt.path.ignore_resume_networks
        ):
            has_gan = False
            gan_opt = opt.train.gan_opt

            if not gan_opt:
                if opt.train.losses:
                    gan_opts = list(
                        filter(
                            lambda x: x["type"].lower() == "ganloss",
                            opt.train.losses,
                        )
                    )
                    if gan_opts:
                        gan_opt = gan_opts[0]

            if gan_opt:
                if gan_opt.get("loss_weight", 0) > 0:
                    has_gan = True

            if has_gan:
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

        # set ae ema path
        if opt.train.ema_decay > 0 and (
            opt.path.ignore_resume_networks is None
            or "network_ae_ema" not in opt.path.ignore_resume_networks
        ):
            model_exists = False
            basepath = ""
            for ext in model_extensions:
                for net_label in ["net_ae_ema", "net_ae"]:
                    basepath = osp.join(opt.path.models, f"{net_label}_{resume_iter}")
                    if osp.exists(f"{basepath}.{ext}"):
                        opt.path.pretrain_network_ae_ema = f"{basepath}.{ext}"
                        model_exists = True
                        print(
                            f"Set pretrain_network_ae_ema to {opt.path.pretrain_network_ae_ema}"
                        )
            if not model_exists:
                raise FileNotFoundError(
                    f"Unable to resume, pretrain_network_ae_ema not found at path: {basepath}.{model_extensions[0]}"
                )

        # set ae path
        if (
            opt.path.ignore_resume_networks is None
            or "network_ae" not in opt.path.ignore_resume_networks
        ):
            model_exists = False
            basepath = ""
            for ext in model_extensions:
                for model_dir in model_dirs:
                    basepath = osp.join(model_dir, f"net_ae_{resume_iter}")
                    if osp.exists(f"{basepath}.{ext}"):
                        opt.path.pretrain_network_ae = f"{basepath}.{ext}"
                        model_exists = True
                        print(
                            f"Set pretrain_network_ae to {opt.path.pretrain_network_ae}"
                        )

            if not model_exists:
                raise FileNotFoundError(
                    f"Unable to resume, pretrain_network_ae not found at path: {basepath}.{model_extensions[0]}",
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


def free_space_gb_str() -> str:
    return f"{shutil.disk_usage(__file__).free / (1 << 30):,.2f} GB"


# https://github.com/jpvanhal/inflection/blob/88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1/inflection/__init__.py#L400
def underscore(word: str) -> str:
    """
    Make an underscored, lowercase form from the expression in the string.

    Example::

        >>> underscore("DeviceType")
        'device_type'

    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::

        >>> camelize(underscore("IOError"))
        'IoError'

    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


def loss_type_to_label(loss_type: str) -> str:
    # label = loss_type.replace("HSLuvLoss", "HSLUVLoss")  # hack for HSLuv
    # label = underscore(label)
    label = loss_type.lower().replace("loss", "")
    return f"l_g_{label}"


def is_json_compatible(value: Any) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    elif isinstance(value, Sequence):
        return all(is_json_compatible(item) for item in value)
    elif isinstance(value, dict):
        return all(
            isinstance(k, str) and is_json_compatible(v) for k, v in value.items()
        )
    return False
