from copy import deepcopy
from typing import Any

from traiNNer.models.base_model import BaseModel

from ..utils import get_root_logger
from .realesrgan_model import RealESRGANModel
from .sr_model import SRModel

__all__ = ["build_model"]


def build_model(opt: dict[str, Any]) -> BaseModel:
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    print(opt)
    opt = deepcopy(opt)

    if opt["high_order_degradation"]:
        model = RealESRGANModel(opt)
    else:
        model = SRModel(opt)

    logger = get_root_logger()
    logger.info("Model [%s] is created.", model.__class__.__name__)
    return model
