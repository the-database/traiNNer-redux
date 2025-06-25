import argparse
import os
import platform
import random
import shlex
import sys
from collections import OrderedDict
from os import path as osp
from pathlib import Path
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


def assert_not_using_template(opt_path: str) -> None:
    abs_path = Path(opt_path).resolve()
    template_root = Path("options/_templates").resolve()

    if template_root in abs_path.parents:
        rel_path = abs_path.relative_to(template_root)
        custom_filename = "custom_" + rel_path.name
        suggested_path = Path("options") / rel_path.parent / custom_filename

        if platform.system() == "Windows":
            copy_cmd = f'copy "{abs_path}" "{suggested_path}"'
        else:
            copy_cmd = f'cp "{abs_path}" "{suggested_path}"'

        original_args = sys.argv
        new_args = [
            str(suggested_path) if Path(arg).resolve() == abs_path else arg
            for arg in original_args
        ]
        new_cmd = " ".join(shlex.quote(arg) for arg in new_args)

        raise RuntimeError(
            f"You are attempting to use a template config.\n\n"
            f"Instead of modifying the template directly, please copy it to another folder first, "
            f"such as:\n  {Path(suggested_path).parent}\n\n"
            f"Then give your copy a unique filename. You can copy the template with the following command (change the custom filename of {custom_filename} to whatever you prefer):\n  {copy_cmd}\n\n"
            f"Then set up your options in your copied config file and re-run the command like this, for example:\n  python {new_cmd}"
        )


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


def yaml_load(file_path: str) -> tuple[ReduxOptions, str]:
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
        return msgspec.yaml.decode(contents, type=ReduxOptions, strict=True), contents


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
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--start-iter", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--manual_seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args()

    # parse yml to dict
    assert_not_using_template(args.opt)
    opt, contents = yaml_load(args.opt)
    opt.contents = contents

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

    # name override
    if args.name:
        opt.name = args.name

    # random seed
    # manual seed override
    if args.manual_seed:
        opt.manual_seed = args.manual_seed
    if not opt.manual_seed:
        opt.manual_seed = random.randint(1024, 10000)

    opt.auto_resume = args.auto_resume
    # opt.resume = args.resume
    opt.watch = args.watch
    opt.start_iter = args.start_iter
    opt.is_train = is_train

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
        # opt.path.pretrain_network_g_ema = opt.path.pretrain_network_g  # necessary with built in EMA, not with ema pytorch
    if opt.path.pretrain_network_d is not None:
        opt.path.pretrain_network_d = osp.expanduser(opt.path.pretrain_network_d)

    if is_train:
        # assert opt.logger is not None, "logger section must be defined when training"  # TODO
        experiments_root = osp.join(root_path, "experiments", opt.name)
        opt.path.experiments_root = experiments_root
        opt.path.models = osp.join(experiments_root, "models")
        opt.path.resume_models = osp.join(opt.path.models, "resume_models")
        opt.path.training_states = osp.join(experiments_root, "training_states")
        opt.path.log = experiments_root
        opt.path.visualization = osp.join(experiments_root, "visualization")

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


def find_arch_entry(variant: str) -> Any:
    from traiNNer.archs.arch_info import ALL_ARCHS

    for arch in ALL_ARCHS:
        if variant in [name.upper() for name in arch["names"]]:
            return arch
    raise ValueError(f"Unknown variant: {variant}")


def infer_template_tag(cfg: ReduxOptions) -> str:
    # Use simple heuristics
    if cfg.onnx is not None:
        return "onnx"
    elif cfg.train is None and cfg.val is not None:
        return "test"

    is_otf = cfg.high_order_degradation
    is_finetune = cfg.path.pretrain_network_g  # TODO resume?

    tag = "OTF_" if is_otf else ""
    tag += "finetune" if is_finetune else "fromscratch"
    return tag


def regenerate_template_from_cfg(cfg: ReduxOptions) -> tuple[str, str]:
    from scripts.options.generate_default_options import (
        final_template,
        template_filename,
        template_onnx,
        template_otf1,
        template_otf2,
        template_paired_finetune,
        template_paired_fromscratch,
        template_test_single,
    )

    from traiNNer.archs.arch_info import (
        OFFICIAL_SETTINGS_FINETUNE,
        OFFICIAL_SETTINGS_FROMSCRATCH,
    )

    if cfg.network_g is not None:
        variant = cfg.network_g["type"].lower()
        try:
            arch = find_arch_entry(variant.upper())

            tag = infer_template_tag(cfg)

            if tag == "onnx":
                template_base = template_onnx
                settings = None
                otf1 = template_otf1
                otf2 = template_otf2
                name_suffix = ""
            elif tag == "test":
                template_base = template_test_single
                settings = None
                otf1 = ""
                otf2 = ""
                name_suffix = ""
            else:
                finetune = tag.endswith("finetune")
                template_base = (
                    template_paired_finetune
                    if finetune
                    else template_paired_fromscratch
                )
                settings = (
                    OFFICIAL_SETTINGS_FINETUNE
                    if finetune
                    else OFFICIAL_SETTINGS_FROMSCRATCH
                )
                otf = tag.startswith("OTF_")
                otf1 = template_otf1 if otf else ""
                otf2 = template_otf2 if otf else ""
                name_suffix = tag if otf else ""

                return final_template(
                    template_base,
                    arch,
                    variant.upper(),
                    training_settings=settings,
                    template_otf1=otf1,
                    template_otf2=otf2,
                    name_suffix=name_suffix,
                ), template_filename(variant, otf=otf, fromscratch=not finetune)
        except ValueError:
            pass

    return "", ""


def recursive_diff(
    user: Any, template: Any, path: list[str] | None = None
) -> list[tuple[str, Any]]:
    path = path or []
    diffs = []

    if type(user) is not type(template):
        diffs.append((".".join(path), user))
        return diffs

    if isinstance(user, dict):
        for key in user:
            new_path = [*path, key]
            u_val = user[key]
            t_val = template.get(key)
            if u_val != t_val:
                diffs.extend(recursive_diff(u_val, t_val, new_path))
        for key in template:
            if key not in user:
                diffs.append((".".join([*path, key]), None))

    elif isinstance(user, list):
        if all(isinstance(x, dict) and "type" in x for x in user + template):
            user_by_type = {x["type"]: x for x in user if "type" in x}
            template_by_type = {x["type"]: x for x in template if "type" in x}
            all_types = [x["type"] for x in user if "type" in x]

            diff_list = []
            for t in all_types:
                u_item = user_by_type.get(t)
                t_item = template_by_type.get(t)
                if t_item is None:
                    diff_list.append(u_item)
                else:
                    subdiffs = recursive_diff(u_item, t_item, [])
                    if subdiffs:
                        merged = {"type": t}
                        for subkey, subval in subdiffs:
                            merged[subkey] = subval
                        diff_list.append(merged)

            if diff_list:
                diffs.append((".".join(path), diff_list))
        else:
            max_len = max(len(user), len(template))
            for i in range(max_len):
                u_val = user[i] if i < len(user) else None
                t_val = template[i] if i < len(template) else None
                if u_val != t_val:
                    diffs.append((".".join([*path, str(i)]), u_val))

    elif user != template:
        diffs.append((".".join(path), user))

    return diffs


def insert_nested(tree: dict, keys: list[str], value: Any) -> None:
    """Insert value into nested dict following keys path."""
    for key in keys[:-1]:
        tree = tree.setdefault(key, {})
    tree[keys[-1]] = value


def build_diff_tree_from_paths(diffs: list[tuple[str, Any]]) -> dict:
    """Convert list of dot-path diffs into nested dict."""
    result = {}
    for path, value in diffs:
        keys = path.split(".")
        insert_nested(result, keys, value)
    return result


def diff_user_vs_template(user_yaml_path: Path) -> tuple[str, str]:
    user_cfg_obj, _ = yaml_load(str(user_yaml_path))
    template_str, template_filename = regenerate_template_from_cfg(user_cfg_obj)
    if template_str and template_filename:
        template_cfg = yaml.safe_load(template_str)

        with open(user_yaml_path) as f:
            user_cfg = yaml.safe_load(f)

        diffs = recursive_diff(user_cfg, template_cfg)
        diff_tree = build_diff_tree_from_paths(diffs)
        if not diff_tree:
            return "", ""

        diff_yaml = yaml.dump(diff_tree, sort_keys=False, allow_unicode=True)
        return diff_yaml, template_filename
    return "", ""
