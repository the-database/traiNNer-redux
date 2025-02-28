from copy import deepcopy
from importlib import import_module
from os import path as osp
from typing import Any

from torch.optim import Adam, AdamW, NAdam, Optimizer
from torch.optim.optimizer import ParamsT

from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.registry import OPTIMIZER_REGISTRY

__all__ = ["build_optimizer"]

# automatically scan and import arch modules for registry
# scan all the files under the 'optimizers' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_optim.py")
]
# import all the arch modules
_arch_modules = [
    import_module(f"traiNNer.optimizers.{file_name}") for file_name in arch_filenames
]


# register built in optimizers
for o in [Adam, AdamW, NAdam]:
    OPTIMIZER_REGISTRY.register(o)


def build_optimizer(params: ParamsT, opt: dict[str, Any]) -> Optimizer:
    opt = deepcopy(opt)
    optimizer_type = opt.pop("type")
    logger = get_root_logger()

    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)(params, **opt)
    logger.info(
        "Optimizer [bold]%s[/bold](%s) is created.",
        optimizer.__class__.__name__,
        opt,
        extra={"markup": True},
    )

    return optimizer
