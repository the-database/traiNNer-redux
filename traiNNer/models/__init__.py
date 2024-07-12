from copy import deepcopy

from traiNNer.models.base_model import BaseModel
from traiNNer.models.realesrgan_model import RealESRGANModel
from traiNNer.models.sr_model import SRModel
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions

__all__ = ["build_model"]


def build_model(opt: ReduxOptions) -> BaseModel:
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)

    if opt.high_order_degradation:
        model = RealESRGANModel(opt)
    else:
        model = SRModel(opt)

    logger = get_root_logger()
    logger.info(
        "Model [bold]%s[/bold] is created.",
        model.__class__.__name__,
        extra={"markup": True},
    )
    return model
