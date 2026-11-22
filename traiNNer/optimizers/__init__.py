import inspect
from copy import deepcopy
from importlib import import_module
from os import path as osp
from typing import Any

from pytorch_optimizer import ADOPT, SOAP, StableAdamW
from torch import Tensor
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
for o in [Adam, AdamW, NAdam, StableAdamW, SOAP, ADOPT]:
    OPTIMIZER_REGISTRY.register(o)


# Optimizers with a fused CUDA step, used unless the config sets fused or foreach. The
# fused step is one kernel per parameter group instead of several per tensor, and
# GradScaler passes it found_inf so skipping a non-finite step needs no GPU sync.
FUSED_BY_DEFAULT = (Adam, AdamW)


def build_optimizer(
    params: ParamsT, opt: dict[str, Any], capturable: bool = False
) -> Optimizer:
    """Build the optimizer named by ``opt["type"]``. With ``capturable``, its step can
    be recorded in a CUDA graph, when the optimizer supports that."""
    opt = deepcopy(opt)
    optimizer_type = opt.pop("type")
    logger = get_root_logger()

    optimizer_cls = OPTIMIZER_REGISTRY.get(optimizer_type)
    if (
        capturable
        and "capturable" not in opt
        and "capturable" in inspect.signature(optimizer_cls).parameters
    ):
        opt["capturable"] = True
    if (
        optimizer_cls in FUSED_BY_DEFAULT
        and "fused" not in opt
        and "foreach" not in opt
    ):
        param_list: list[Any] = list(params)
        params = param_list
        if param_list and all(
            isinstance(p, Tensor) and p.is_cuda and p.is_floating_point()
            for p in param_list
        ):
            opt["fused"] = True

    optimizer = optimizer_cls(params, **opt)
    logger.info(
        "Optimizer [bold]%s[/bold](%s) is created.",
        optimizer.__class__.__name__,
        opt,
        extra={"markup": True},
    )

    return optimizer
