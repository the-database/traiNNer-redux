import importlib
from os import path as osp

from torch import nn

from traiNNer.losses.gan_loss import (
    g_path_regularize,
    gradient_penalty_loss,
    r1_penalty,
)
from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.options import struct2dict
from traiNNer.utils.optionsfile import LossOptions
from traiNNer.utils.registry import LOSS_REGISTRY

__all__ = ["build_loss", "gradient_penalty_loss", "r1_penalty", "g_path_regularize"]

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(loss_folder)
    if v.endswith("_loss.py")
]
# import all the loss modules
_model_modules = [
    importlib.import_module(f"traiNNer.losses.{file_name}")
    for file_name in loss_filenames
]


def build_loss(loss_opt: LossOptions) -> nn.Module:
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = struct2dict(loss_opt)
    loss_type = opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info("Loss [%s] is created.", loss.__class__.__name__)
    return loss
