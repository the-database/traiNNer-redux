from copy import deepcopy
from importlib import import_module
from importlib.metadata import version
from os import path as osp

from ..utils import get_root_logger, scandir
from ..utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY

__all__ = ["build_network"]
spandrel_name = "spandrel"

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith("_arch.py")]
# import all the arch modules
_arch_modules = [import_module(f"traiNNer.archs.{file_name}") for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop("type")
    logger = get_root_logger()

    # try loading from spandrel first
    try:
        net = SPANDREL_REGISTRY.get(network_type)(**opt)
        logger.info(f"Network [{net.__class__.__name__}] is created from {spandrel_name} v{version(spandrel_name)}.")

    except KeyError:
        net = ARCH_REGISTRY.get(network_type)(**opt)
        logger.info(f"Network [{net.__class__.__name__}] is created from local.")

    return net
