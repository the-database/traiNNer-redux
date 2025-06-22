from copy import deepcopy
from importlib import import_module
from importlib.metadata import version
from os import path as osp
from typing import Any

from torch import nn

from traiNNer.utils.misc import scandir
from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY, TESTARCH_REGISTRY

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
    from traiNNer.utils.logger import get_root_logger

    opt = deepcopy(opt)
    network_type = opt.pop("type")
    logger = get_root_logger()

    # try loading from spandrel first
    if network_type in SPANDREL_REGISTRY:
        net = SPANDREL_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]%s[/bold] v%s.",
            net.__class__.__name__,
            opt,
            spandrel_name,
            version(spandrel_name),
            extra={"markup": True},
        )
    elif network_type in ARCH_REGISTRY:
        net = ARCH_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]traiNNer-redux[/bold].",
            net.__class__.__name__,
            opt,
            extra={"markup": True},
        )
    else:
        net = TESTARCH_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]traiNNer-redux-test[/bold].",
            net.__class__.__name__,
            opt,
            extra={"markup": True},
        )

    return net
