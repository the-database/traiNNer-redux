import argparse
import os
import random
from collections import OrderedDict
from os import path as osp
from typing import Any

import msgspec
import torch
import yaml
from yaml import MappingNode

from traiNNer.utils.dist_util import get_dist_info, init_dist, master_only
from traiNNer.utils.redux_options import ReduxOptions

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def ordered_yaml() -> tuple[type[Loader], type[Dumper]]:
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper: Dumper, data: dict[str, Any]) -> MappingNode:
        return dumper.represent_dict(data.items())

    def dict_constructor(loader: Loader, node: MappingNode) -> OrderedDict:
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(file_path: str) -> ReduxOptions:
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Options file does not exist: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        contents = f.read()
        return msgspec.yaml.decode(contents, type=ReduxOptions, strict=True)


def struct2dict(obj: msgspec.Struct) -> dict[str, Any]:
    return {
        field: getattr(obj, field)
        for field in obj.__struct_fields__
        if not field.startswith("_") and getattr(obj, field) is not None
    }


def dict2str(opt: dict[str, Any], indent_level: int = 1) -> str:
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg


def _postprocess_yml_value(value: str) -> None | bool | float | int | list[Any] | str:
    # None
    if value == "~" or value.lower() == "none":
        return None
    # bool
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    # !!float number
    if value.startswith("!!float"):
        return float(value.replace("!!float", ""))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
        return float(value)
    # list
    if value.startswith("["):
        return eval(value)
    # str
    return value


def parse_options(
    root_path: str, is_train: bool = True
) -> tuple[ReduxOptions, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--force_yml",
        nargs="+",
        default=None,
        help="Force to update yml files. Examples: train:ema_decay=0.999",
    )
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # distributed settings
    if args.launcher == "none":
        opt.dist = False
        print("Disable distributed.", flush=True)
    else:
        opt.dist = True
        if args.launcher == "slurm" and opt.dist_params is not None:
            init_dist(args.launcher, **opt.dist_params)
        else:
            init_dist(args.launcher)
    opt.rank, opt.world_size = get_dist_info()

    # random seed
    opt.deterministic = opt.manual_seed is not None and opt.manual_seed > 0
    if not opt.deterministic:
        opt.manual_seed = random.randint(1024, 10000)

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split("=")
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = "opt"
            for key in keys.split(":"):
                eval_str += f'["{key}"]'
            eval_str += "=value"
            # using exec function
            exec(eval_str)

    opt.auto_resume = args.auto_resume
    opt.resume = args.resume
    opt.is_train = is_train

    # debug setting
    if args.debug and not opt.name.startswith("debug"):
        opt.name = "debug_" + opt.name

    if opt.num_gpu == "auto":
        opt.num_gpu = torch.cuda.device_count()

    # datasets
    for full_phase, dataset in opt.datasets.items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = full_phase.split("_")[0]
        dataset.phase = phase
        dataset.scale = opt.scale
        if dataset.dataroot_gt is not None:
            if isinstance(dataset.dataroot_gt, str):
                dataset.dataroot_gt = [osp.expanduser(dataset.dataroot_gt)]
            else:
                dataset.dataroot_gt = [osp.expanduser(p) for p in dataset.dataroot_gt]
        if dataset.dataroot_lq is not None:
            if isinstance(dataset.dataroot_lq, str):
                dataset.dataroot_lq = [osp.expanduser(dataset.dataroot_lq)]
            else:
                dataset.dataroot_lq = [osp.expanduser(p) for p in dataset.dataroot_lq]

    if opt.path.resume_state is not None:
        opt.path.resume_state = osp.expanduser(opt.path.resume_state)
    if opt.path.pretrain_network_g is not None:
        opt.path.pretrain_network_g = osp.expanduser(opt.path.pretrain_network_g)
        opt.path.pretrain_network_g_ema = opt.path.pretrain_network_g
    if opt.path.pretrain_network_d is not None:
        opt.path.pretrain_network_d = osp.expanduser(opt.path.pretrain_network_d)
    if opt.path.pretrain_network_ae_decoder is not None:
        opt.path.pretrain_network_ae_decoder = osp.expanduser(
            opt.path.pretrain_network_ae_decoder
        )
        opt.path.pretrain_network_ae_decoder_ema = opt.path.pretrain_network_ae_decoder
    if opt.path.pretrain_network_ae is not None:
        opt.path.pretrain_network_ae = osp.expanduser(opt.path.pretrain_network_ae)

    if is_train:
        if opt.train and opt.train.losses is not None:
            for loss in opt.train.losses:
                if loss["type"].lower() == "aesoploss":
                    if opt.path.pretrain_network_ae is None:
                        raise ValueError(
                            "path.pretrain_network_ae is required for aesoploss"
                        )
                    loss["scale"] = opt.scale
                    loss["pretrain_network_ae"] = opt.path.pretrain_network_ae
        assert opt.logger is not None, "logger section must be defined when training"
        experiments_root = osp.join(root_path, "experiments", opt.name)
        opt.path.experiments_root = experiments_root
        opt.path.models = osp.join(experiments_root, "models")
        opt.path.resume_models = osp.join(opt.path.models, "resume_models")
        opt.path.training_states = osp.join(experiments_root, "training_states")
        opt.path.log = experiments_root
        opt.path.visualization = osp.join(experiments_root, "visualization")

        # change some options for debug mode
        if "debug" in opt.name:
            if opt.val is not None:
                opt.val.val_freq = 8
            opt.logger.print_freq = 1
            opt.logger.save_checkpoint_freq = 8
    else:  # test
        results_root = osp.join(root_path, "results", opt.name)
        opt.path.results_root = results_root
        opt.path.log = results_root
        opt.path.visualization = osp.join(results_root, "visualization")

    return opt, args


@master_only
def copy_opt_file(opt_file: str, experiments_root: str) -> None:
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile

    cmd = " ".join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, "r+") as f:
        lines = f.readlines()
        lines.insert(0, f"# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n")
        f.seek(0)
        f.writelines(lines)
