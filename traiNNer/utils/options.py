import argparse
import difflib
import os
import platform
import random
import re
import shlex
import sys
from os import path as osp
from pathlib import Path
from typing import Any

import msgspec
import torch
import yaml

from traiNNer.utils.dist_util import get_dist_info, init_dist, master_only
from traiNNer.utils.redux_options import ReduxOptions


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

        # Training templates are named <variant>[_OTF]_<fidelity|gan>.yml
        new_config_hint = ""
        match = re.fullmatch(r"(.+?)(_OTF)?_(fidelity|gan)", abs_path.stem)
        if rel_path.parts[0] == "train" and match is not None:
            flags = (" --otf" if match.group(2) else "") + (
                " --gan" if match.group(3) == "gan" else ""
            )
            new_config_hint = (
                "To create your own config from this template, run this command "
                "(change my_model to your model's name):\n"
                f"  python train.py --new {match.group(1).lower()}{flags} --name my_model\n\n"
                "Or copy the template yourself, "
            )
        else:
            new_config_hint = "Copy the template to another folder first, "

        raise RuntimeError(
            f"You are attempting to use a template config. Templates are updated by "
            f"git, so don't edit or train from them directly.\n\n"
            f"{new_config_hint}for example (change the file name of {custom_filename} "
            f"to whatever you prefer):\n  {copy_cmd}\n\n"
            f"Then set up your options in your copied config file and run it, "
            f"for example:\n  python {new_cmd}"
        )


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


def build_arg_parser(is_train: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Not required=True: train.py --new runs without a config file.
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
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
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name, overriding name in the config. With --new, also "
        "the file name of the new config.",
    )

    if is_train:
        group = parser.add_argument_group("create a new training config")
        group.add_argument(
            "--new",
            nargs="?",
            const="",
            metavar="VARIANT",
            help="Create a training config for an architecture variant, such as "
            "span_s, in options/train and exit. Run --new without a variant to "
            "list all variants.",
        )
        group.add_argument(
            "--scale",
            type=int,
            help="Scale for --new. Defaults to the template's scale.",
        )
        group.add_argument(
            "--gan",
            action="store_true",
            help="With --new, use the GAN fine-tuning template instead of the "
            "fidelity template for training from scratch.",
        )
        group.add_argument(
            "--otf",
            action="store_true",
            help="With --new, use the template with on-the-fly degradations, "
            "which only needs HR images, instead of a paired HR/LR dataset.",
        )

    return parser


def parse_options(
    root_path: str, is_train: bool = True
) -> tuple[ReduxOptions, argparse.Namespace]:
    parser = build_arg_parser(is_train)
    args = parser.parse_args()
    if args.opt is None:
        parser.error("the following arguments are required: -opt")

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


def list_train_variants() -> str:
    from traiNNer.archs.arch_info import ALL_ARCHS, arch_folder_name

    lines = [
        f"  {arch_folder_name(arch)}: {', '.join(n.lower() for n in arch['names'])}"
        for arch in ALL_ARCHS
    ]
    return "\n".join(sorted(lines, key=str.lower))


def create_train_config(
    root_path: Path,
    variant: str,
    scale: int | None = None,
    name: str | None = None,
    gan: bool = False,
    otf: bool = False,
) -> Path:
    """Copy a training template to options/train/<ARCH>/<name>.yml.

    Only the top-level name and scale lines are changed, so the copy keeps all
    of the template's comments.
    """
    from traiNNer.archs.arch_info import (
        ALL_ARCHS,
        arch_folder_name,
        template_filename,
    )

    variants = {n.lower(): n for arch in ALL_ARCHS for n in arch["names"]}
    canonical = variants.get(variant.lower())
    if canonical is None:
        close = difflib.get_close_matches(variant.lower(), variants, n=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(
            f"Unknown variant '{variant}'.{hint} "
            "Run `python train.py --new` to list all variants."
        )
    arch = find_arch_entry(canonical.upper())
    folder = arch_folder_name(arch)

    template_rel = (
        Path("options/_templates/train")
        / folder
        / template_filename(canonical, otf=otf, fromscratch=not gan)
    )
    text = (root_path / template_rel).read_text(encoding="utf-8")

    scale_match = re.search(r"^scale: (\d+)", text, flags=re.MULTILINE)
    name_match = re.search(r"^name: (.+)$", text, flags=re.MULTILINE)
    assert scale_match is not None and name_match is not None, template_rel
    template_scale = int(scale_match.group(1))

    if scale is None:
        scale = template_scale
    elif scale not in arch["scales"]:
        supported = ", ".join(str(s) for s in arch["scales"])
        raise ValueError(
            f"{canonical} does not support scale {scale}. Supported scales: {supported}."
        )

    if name is None:
        # Template names start with the scale, such as 4x_SPAN_S.
        name = re.sub(
            rf"^{template_scale}x_", f"{scale}x_", name_match.group(1).strip()
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name):
        raise ValueError(
            f"Invalid name '{name}'. Use letters, digits and _ . + - only, "
            "since it's also used as the file and experiment folder name."
        )

    out_path = root_path / "options" / "train" / folder / f"{name}.yml"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path.relative_to(root_path)} already exists. Choose a different "
            "--name, or edit the existing config."
        )

    text = re.sub(r"^name: .+$", f"name: {name}", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^scale: \d+", f"scale: {scale}", text, count=1, flags=re.MULTILINE)
    header = f"# Created from {template_rel.as_posix()} by train.py --new\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + text, encoding="utf-8")
    return out_path


def create_config_from_args(root_path: str) -> bool:
    """Handle train.py --new. Returns True if the caller should exit."""
    parser = build_arg_parser(is_train=True)
    args = parser.parse_args()

    if args.new is None:
        if args.scale is not None or args.gan or args.otf:
            parser.error("--scale, --gan and --otf are only used with --new")
        return False
    if args.opt is not None:
        parser.error("--new creates a new config, so it can't be combined with -opt")

    if args.new == "":
        print("Variants available for --new, by architecture:")
        print(list_train_variants())
        return True

    try:
        out_path = create_train_config(
            Path(root_path),
            args.new,
            scale=args.scale,
            name=args.name,
            gan=args.gan,
            otf=args.otf,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    rel_path = out_path.relative_to(root_path)
    to_set = ["dataroot_gt"] if args.otf else ["dataroot_gt", "dataroot_lq"]
    if args.gan:
        to_set.append("pretrain_network_g")
    to_set_str = (
        to_set[0] if len(to_set) == 1 else f"{', '.join(to_set[:-1])} and {to_set[-1]}"
    )
    print(
        f"Created {rel_path}\n\n"
        f"Open it and set {to_set_str}, then start training with:\n"
        f"  python train.py --auto_resume -opt {rel_path}"
    )
    return True


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
        template_filename,
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
