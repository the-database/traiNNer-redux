from copy import deepcopy
from typing import Any

from ..utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ["calculate_psnr", "calculate_ssim"]


def calculate_metric(data: dict[str, Any], opt: dict[str, Any]) -> float:
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
