from copy import deepcopy
from importlib import import_module
from importlib.metadata import version
from os import path as osp
from typing import Any

from torch import nn
from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.config import Config
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY

__all__ = ["build_network"]
spandrel_name = "spandrel"

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_arch.py")
]
# import all the arch modules
_arch_modules = [
    import_module(f"traiNNer.archs.{file_name}") for file_name in arch_filenames
]


def build_network(opt: dict[str, Any]) -> nn.Module:
    opt = deepcopy(opt)
    network_type = opt.pop("type")
    logger = get_root_logger()

    opt["scale"] = Config.get_scale()

    # try loading from spandrel first
    try:
        net = SPANDREL_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [%s] is created from %s v%s.",
            net.__class__.__name__,
            spandrel_name,
            version(spandrel_name),
        )

    except KeyError:
        net = ARCH_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network %s is created from traiNNer-redux.", net.__class__.__name__
        )

    return net
