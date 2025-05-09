from copy import deepcopy

from traiNNer.models.ae_model import AEModel
from traiNNer.models.base_model import BaseModel
from traiNNer.models.realesrgan_model import RealESRGANModel
from traiNNer.models.realesrgan_paired_model import (
    RealESRGANPairedModel,
)
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
    logger = get_root_logger()
    if opt.high_order_degradation:
        if opt.dataroot_lq_prob > 0:
            model = RealESRGANPairedModel(opt)
        else:
            model = RealESRGANModel(opt)
    elif opt.network_ae and not opt.network_g:
        model = AEModel(opt)
    else:
        model = SRModel(opt)

    logger.info(
        "Model [bold]%s[/bold] is created.",
        model.__class__.__name__,
        extra={"markup": True},
    )
    return model
